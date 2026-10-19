#!/usr/bin/env python3
"""Fermiwork: offline Fermi-estimate worksheets written in plain text.

Name the inputs of an estimate once, give each a range you can defend, and
Fermiwork returns the point answer, guaranteed interval bounds, a seeded
Monte Carlo spread, the inputs that move the answer most, and plausibility
gates for CI. Python standard library only; nothing touches the network.

Worksheet grammar (one statement per line, '#' starts a comment):

    title TEXT                      optional heading
    estimate NAME                   the quantity you are answering for (required)
    truth NUMBER                    optional known answer, used by `score`
    expect NAME in [LOW, HIGH]      plausibility bounds, used by `check`
    NAME = EXPR                     derived quantity
    NAME = NUMBER in [LOW, HIGH]    input with a range (uniform)
    NAME = NUMBER in [LOW, HIGH] log  input with a log-uniform range

EXPR may use + - * / **, parentheses, numbers, earlier names, pi, e and the
functions sqrt, ln, log10, exp, ceil, floor, abs, min, max.
"""

import argparse
import ast
import bisect
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

DEFAULT_TRIALS = 10000
DEFAULT_SEED = 1

CONSTANTS = {"pi": math.pi, "e": math.e}
RESERVED = {"title", "truth", "estimate", "expect"}

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
ASSIGN_RE = re.compile(rf"^({IDENT})\s*=(?!=)\s*(.+)$")
RANGE_RE = re.compile(r"^(.*?)\s+in\s+\[([^,\]]*),([^\]]*)\]\s*(log)?$")
EXPECT_RE = re.compile(rf"^({IDENT})\s+in\s+\[([^,\]]*),([^\]]*)\]$")


class FermiError(Exception):
    def __init__(self, message, line=None):
        super().__init__(message)
        self.message = message
        self.line = line


@dataclass
class Var:
    name: str
    text: str
    line: int
    tree: ast.Expression
    refs: list
    lo: float = None
    hi: float = None
    log: bool = False
    point: float = None

    @property
    def is_input(self):
        return self.lo is not None


@dataclass
class Expect:
    name: str
    lo: float
    hi: float
    line: int


@dataclass
class Model:
    title: str = ""
    estimate: str = None
    estimate_line: int = None
    truth: float = None
    truth_line: int = None
    expects: list = field(default_factory=list)
    vars: dict = field(default_factory=dict)
    order: list = field(default_factory=list)


# ---------------------------------------------------------------- arithmetic

def _finite(iv):
    if not all(math.isfinite(x) for x in iv):
        raise FermiError("result is not a finite number")
    return iv


def _iv_mul(a, b):
    corners = [x * y for x in a for y in b]
    return (min(corners), max(corners))


def _iv_div(a, b):
    if b[0] <= 0 <= b[1]:
        raise FermiError("division by zero, or by a range that includes zero")
    corners = [x / y for x in a for y in b]
    return (min(corners), max(corners))


def _iv_pow(a, b):
    if a[0] == a[1] and b[0] == b[1]:
        value = a[0] ** b[0]
        if isinstance(value, complex):
            raise FermiError("power gives a complex number")
        return (value, value)
    if a[0] <= 0:
        raise FermiError("a power with a varying base needs the base above zero")
    corners = [x ** y for x in a for y in b]
    return (min(corners), max(corners))


BINARY = {
    ast.Add: lambda a, b: (a[0] + b[0], a[1] + b[1]),
    ast.Sub: lambda a, b: (a[0] - b[1], a[1] - b[0]),
    ast.Mult: _iv_mul,
    ast.Div: _iv_div,
    ast.Pow: _iv_pow,
}


def _one(name, args):
    if len(args) != 1:
        raise FermiError(f"{name}() takes exactly one argument")
    return args[0]


def _increasing(name, fn, floor=None, strict=False):
    def apply(args):
        lo, hi = _one(name, args)
        if floor is not None and (lo < floor or (strict and lo == floor)):
            bound = "above" if strict else "at or above"
            raise FermiError(f"{name}() needs its input {bound} {floor:g}")
        return (float(fn(lo)), float(fn(hi)))
    return apply


def _fn_abs(args):
    lo, hi = _one("abs", args)
    if lo >= 0:
        return (lo, hi)
    if hi <= 0:
        return (-hi, -lo)
    return (0.0, max(-lo, hi))


def _fn_extreme(name, pick):
    def apply(args):
        if not args:
            raise FermiError(f"{name}() needs at least one argument")
        return (pick(a[0] for a in args), pick(a[1] for a in args))
    return apply


FUNCTIONS = {
    "sqrt": _increasing("sqrt", math.sqrt, floor=0),
    "ln": _increasing("ln", math.log, floor=0, strict=True),
    "log10": _increasing("log10", math.log10, floor=0, strict=True),
    "exp": _increasing("exp", math.exp),
    "ceil": _increasing("ceil", math.ceil),
    "floor": _increasing("floor", math.floor),
    "abs": _fn_abs,
    "min": _fn_extreme("min", min),
    "max": _fn_extreme("max", max),
}


def _eval_node(node, env):
    """Evaluate a validated expression tree over intervals (lo, hi)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        value = float(node.value)
        return (value, value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        value = CONSTANTS[node.id]
        return (value, value)
    if isinstance(node, ast.UnaryOp):
        lo, hi = _eval_node(node.operand, env)
        return (-hi, -lo) if isinstance(node.op, ast.USub) else (lo, hi)
    if isinstance(node, ast.BinOp):
        return BINARY[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env))
    args = [_eval_node(arg, env) for arg in node.args]
    return FUNCTIONS[node.func.id](args)


def _safe_eval(tree, env):
    try:
        return _finite(_eval_node(tree, env))
    except FermiError:
        raise
    except (ArithmeticError, ValueError) as exc:
        raise FermiError(f"cannot evaluate ({exc})") from None


def _compile_expr(text, line):
    """Parse an expression, reject anything outside the allowed grammar, return (tree, names)."""
    source = text.strip()
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError, ValueError):
        raise FermiError(f"syntax error in '{source}'", line) from None
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    refs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.expr_context, ast.operator, ast.unaryop)):
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            continue
        if isinstance(node, ast.Name):
            if id(node) not in called:
                refs[node.id] = None
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                name = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
                raise FermiError(f"unknown function '{name}'", line)
            continue
        raise FermiError(f"unsupported syntax ({type(node).__name__})", line)
    return tree, list(refs)


def _const(text, line):
    tree, refs = _compile_expr(text, line)
    if refs:
        raise FermiError(f"'{refs[0]}' is a name, but a plain number is needed here", line)
    return _safe_eval(tree, {})[0]


def _bounds(low_text, high_text, line):
    low = _const(low_text, line)
    high = _const(high_text, line)
    if low > high:
        raise FermiError("range is reversed: lower bound is above upper bound", line)
    return low, high


# ------------------------------------------------------------------- parsing

def _parse_assignment(model, name, body, lineno):
    if name in RESERVED or name in CONSTANTS or name in FUNCTIONS:
        raise FermiError(f"'{name}' is a reserved name", lineno)
    if name in model.vars:
        raise FermiError(f"'{name}' is already defined on line {model.vars[name].line}", lineno)
    match = RANGE_RE.match(body)
    expr_text = match.group(1) if match else body
    tree, refs = _compile_expr(expr_text, lineno)
    var = Var(name, expr_text.strip(), lineno, tree, refs)
    if match:
        if refs:
            raise FermiError(f"'{name}' has a range, so its value must be a plain number", lineno)
        var.lo, var.hi = _bounds(match.group(2), match.group(3), lineno)
        var.log = bool(match.group(4))
        if var.log and var.lo <= 0:
            raise FermiError("a log range needs a lower bound above zero", lineno)
        var.point = _const(expr_text, lineno)
        if not var.lo <= var.point <= var.hi:
            raise FermiError(f"value {fmt(var.point)} is outside its range", lineno)
    model.vars[name] = var


def _parse_line(model, line, lineno):
    match = ASSIGN_RE.match(line)
    if match:
        _parse_assignment(model, match.group(1), match.group(2), lineno)
        return
    parts = line.split(None, 1)
    word = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""
    if word == "title":
        if not rest:
            raise FermiError("title needs some text", lineno)
        model.title = rest
    elif word == "estimate":
        if not re.fullmatch(IDENT, rest):
            raise FermiError("estimate needs exactly one variable name", lineno)
        if model.estimate is not None:
            raise FermiError(f"estimate is already set on line {model.estimate_line}", lineno)
        model.estimate, model.estimate_line = rest, lineno
    elif word == "truth":
        if model.truth is not None:
            raise FermiError(f"truth is already set on line {model.truth_line}", lineno)
        value = _const(rest, lineno)
        if value <= 0:
            raise FermiError("truth must be above zero", lineno)
        model.truth, model.truth_line = value, lineno
    elif word == "expect":
        match = EXPECT_RE.match(rest)
        if not match:
            raise FermiError("expected: expect NAME in [LOW, HIGH]", lineno)
        lo, hi = _bounds(match.group(2), match.group(3), lineno)
        model.expects.append(Expect(match.group(1), lo, hi, lineno))
    else:
        raise FermiError(f"unrecognised line starting with '{word}'", lineno)


def parse(text):
    """Parse worksheet text. Returns (model, errors); errors are (line, message)."""
    model = Model()
    errors = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            _parse_line(model, line, lineno)
        except FermiError as exc:
            errors.append((exc.line or lineno, exc.message))
    return model, errors


def _order(model, errors):
    """Topologically sort variables (dependencies first) and report cycles."""
    order, state = [], {}

    def visit(name, path):
        state[name] = "open"
        for ref in model.vars[name].refs:
            if ref not in model.vars:
                continue
            if state.get(ref) == "open":
                cycle = path[path.index(ref):] + [ref]
                errors.append((model.vars[name].line, "circular reference: " + " -> ".join(cycle)))
            elif ref not in state:
                visit(ref, path + [ref])
        state[name] = "done"
        order.append(name)

    for name in model.vars:
        if name not in state:
            visit(name, [name])
    model.order = order


def _validate(model):
    errors = []
    if model.estimate is None:
        errors.append((None, "no 'estimate' line names the answer"))
    elif model.estimate not in model.vars:
        errors.append((model.estimate_line, f"estimate '{model.estimate}' is not defined"))
    for var in model.vars.values():
        for ref in var.refs:
            if ref not in model.vars and ref not in CONSTANTS:
                errors.append((var.line, f"unknown name '{ref}'"))
    for ex in model.expects:
        if ex.name not in model.vars:
            errors.append((ex.line, f"expect refers to undefined '{ex.name}'"))
    _order(model, errors)
    if errors:
        return errors
    _, run_errors = _run(model, interval_inputs(model))
    errors.extend(run_errors)
    return errors


def load_text(text):
    """Parse and validate a worksheet. Returns (model, sorted errors)."""
    model, errors = parse(text)
    errors.extend(_validate(model))
    errors.sort(key=lambda e: (e[0] is None, e[0] or 0))
    return model, errors


# ---------------------------------------------------------------- evaluation

def _inputs(model):
    return [v for v in model.vars.values() if v.is_input]


def point_inputs(model):
    return {v.name: (v.point, v.point) for v in _inputs(model)}


def interval_inputs(model):
    return {v.name: (v.lo, v.hi) for v in _inputs(model)}


def _run(model, inputs):
    """Evaluate every variable in dependency order. Returns (env, errors)."""
    env, errors = {}, []
    for name in model.order:
        var = model.vars[name]
        if var.is_input:
            env[name] = inputs[name]
            continue
        if any(ref in model.vars and ref not in env for ref in var.refs):
            continue  # a dependency already failed; that error is reported instead
        try:
            env[name] = _safe_eval(var.tree, env)
        except FermiError as exc:
            errors.append((var.line, f"'{name}': {exc.message}"))
    return env, errors


def _envs(model):
    point_env, _ = _run(model, point_inputs(model))
    interval_env, _ = _run(model, interval_inputs(model))
    return point_env, interval_env


def _estimate(model, inputs):
    env, errors = _run(model, inputs)
    if errors:
        line, message = errors[0]
        raise FermiError(message, line)
    return env[model.estimate]


def _point(model):
    return _estimate(model, point_inputs(model))[0]


def _draw(var, u):
    if var.log:
        value = var.lo * (var.hi / var.lo) ** u
    else:
        value = var.lo + (var.hi - var.lo) * u
    return min(max(value, var.lo), var.hi)


def percentile(sorted_values, p):
    """Linear-interpolated percentile of an already sorted list."""
    k = (len(sorted_values) - 1) * p / 100
    low = math.floor(k)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (k - low)


def simulate(model, trials=DEFAULT_TRIALS, seed=DEFAULT_SEED):
    """Seeded Monte Carlo over the input ranges. Failed trials are counted, not hidden."""
    rng = random.Random(seed)
    inputs = _inputs(model)
    samples, failures, first_error = [], 0, None
    for _ in range(trials):
        values = {}
        for var in inputs:
            value = _draw(var, rng.random())
            values[var.name] = (value, value)
        env, errors = _run(model, values)
        if errors:
            failures += 1
            first_error = first_error or errors[0][1]
            continue
        samples.append(env[model.estimate][0])
    if not samples:
        raise FermiError(f"every trial failed; first error: {first_error}")
    samples.sort()
    n = len(samples)
    geo_mean = None
    if samples[0] > 0:
        geo_mean = math.exp(sum(math.log(x) for x in samples) / n)
    return {
        "trials": trials,
        "seed": seed,
        "completed": n,
        "failures": failures,
        "first_error": first_error,
        "samples": samples,
        "p5": percentile(samples, 5),
        "p25": percentile(samples, 25),
        "p50": percentile(samples, 50),
        "p75": percentile(samples, 75),
        "p95": percentile(samples, 95),
        "geo_mean": geo_mean,
    }


def share_inside(samples, lo, hi):
    return (bisect.bisect_right(samples, hi) - bisect.bisect_left(samples, lo)) / len(samples)


def order_shares(samples):
    """Share of trials per power of ten, as [(exponent or None, share)]; None means <= 0."""
    counts = {}
    for x in samples:
        key = math.floor(math.log10(x)) if x > 0 else None
        counts[key] = counts.get(key, 0) + 1
    n = len(samples)
    ordered = sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    return [(key, count / n) for key, count in ordered]


def tornado(model):
    """One-at-a-time sensitivity: move each input across its range, others at their points."""
    base = point_inputs(model)
    base_value = _estimate(model, base)[0]
    rows = []
    for var in _inputs(model):
        low = _estimate(model, {**base, var.name: (var.lo, var.lo)})[0]
        high = _estimate(model, {**base, var.name: (var.hi, var.hi)})[0]
        rows.append({
            "name": var.name,
            "low_input": var.lo,
            "high_input": var.hi,
            "output_at_low": low,
            "output_at_high": high,
            "swing": abs(high - low),
        })
    total = sum(row["swing"] for row in rows)
    for row in rows:
        row["share"] = row["swing"] / total if total else 0.0
    rows.sort(key=lambda row: (-row["swing"], row["name"]))
    return base_value, rows


def score_model(model, trials=DEFAULT_TRIALS, seed=DEFAULT_SEED):
    """Compare the point estimate with the `truth` line."""
    point = _point(model)
    truth = model.truth
    sim = simulate(model, trials, seed)
    return {
        "truth": truth,
        "point": point,
        "factor": max(point / truth, truth / point),
        "log10_error": math.log10(point / truth),
        "truth_percentile": 100 * bisect.bisect_left(sim["samples"], truth) / sim["completed"],
        "inside_p5_p95": sim["p5"] <= truth <= sim["p95"],
    }


def calibration_summary(scores):
    """Aggregate score_model() results over many worksheets with known answers."""
    factors = sorted(s["factor"] for s in scores)
    n = len(scores)
    return {
        "count": n,
        "median_factor": percentile(factors, 50),
        "within_2x": sum(1 for s in scores if s["factor"] <= 2) / n,
        "inside_p5_p95": sum(1 for s in scores if s["inside_p5_p95"]) / n,
    }


def _expect_verdict(ex, point_env, interval_env):
    value = point_env[ex.name][0]
    low, high = interval_env[ex.name]
    if not ex.lo <= value <= ex.hi:
        return value, "FAIL", False
    if ex.lo <= low and high <= ex.hi:
        return value, "PASS (guaranteed)", True
    return value, "PASS (point only)", True


def _tree(model, env, name, depth, seen, out):
    var = model.vars[name]
    value = env[name][0]
    indent = "  " * depth
    if name in seen:
        out.append(f"{indent}{name} (see above)")
        return
    seen.add(name)
    if var.is_input:
        scale = " log" if var.log else ""
        note = f"input{scale}, range {fmt(var.lo)} to {fmt(var.hi)}"
    else:
        note = f"= {var.text}"
    out.append(f"{indent}{name} = {fmt(value)}  [{note}]")
    for ref in sorted(ref for ref in var.refs if ref in model.vars):
        _tree(model, env, ref, depth + 1, seen, out)


# ---------------------------------------------------------------- formatting

def _trim(text):
    return text.rstrip("0").rstrip(".") if "." in text else text


def fmt(x, digits=4):
    """Format with `digits` significant figures, thousands separators or x 10^n."""
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    if -3 <= exp < 6:
        return _trim(f"{x:,.{max(0, digits - 1 - exp)}f}")
    mantissa = _trim(f"{x / 10 ** exp:.{digits - 1}f}")
    return f"{mantissa} × 10^{exp}"


def _load(path):
    """Return (model, exit_code); model is None when the file cannot be used."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        print(f"fermiwork: cannot read '{path}': {exc.strerror}", file=sys.stderr)
        return None, EXIT_USAGE
    model, errors = load_text(text)
    if errors:
        for line, message in errors:
            where = f"{path}:{line}" if line else path
            print(f"{where}: {message}", file=sys.stderr)
        return None, EXIT_FAIL
    return model, EXIT_OK


# ------------------------------------------------------------------ commands

def cmd_check(model, args):
    point_env, interval_env = _envs(model)
    lo, hi = interval_env[model.estimate]
    status = EXIT_OK
    print(f"OK: {len(_inputs(model))} inputs, {len(model.vars) - len(_inputs(model))} derived, "
          f"estimate '{model.estimate}'")
    print(f"point {fmt(point_env[model.estimate][0])} | guaranteed {fmt(lo)} to {fmt(hi)}")
    for ex in model.expects:
        value, verdict, ok = _expect_verdict(ex, point_env, interval_env)
        status = status if ok else EXIT_FAIL
        print(f"  line {ex.line}: expect {ex.name} in [{fmt(ex.lo)}, {fmt(ex.hi)}]: "
              f"point {fmt(value)} {verdict}")
    if args.max_spread is not None:
        spread_ok = lo > 0 and hi / lo <= args.max_spread
        spread = f"{fmt(hi / lo, 3)}x" if lo > 0 else "unbounded"
        status = status if spread_ok else EXIT_FAIL
        print(f"  spread {spread} vs limit {fmt(args.max_spread, 3)}x: {'PASS' if spread_ok else 'FAIL'}")
    print("checks passed" if status == EXIT_OK else "checks FAILED")
    return status


def cmd_eval(model, args):
    point_env, interval_env = _envs(model)
    value = point_env[model.estimate][0]
    lo, hi = interval_env[model.estimate]
    if args.json:
        payload = {
            "title": model.title,
            "estimate": {"name": model.estimate, "point": value, "low": lo, "high": hi},
            "variables": {
                name: {
                    "point": point_env[name][0],
                    "low": interval_env[name][0],
                    "high": interval_env[name][1],
                    "line": model.vars[name].line,
                }
                for name in model.order
            },
        }
        print(json.dumps(payload, indent=2))
        return EXIT_OK
    print(model.title or "Fermi estimate")
    print(f"estimate {model.estimate} = {fmt(value)} (about {fmt(value, 1)})")
    print(f"guaranteed range {fmt(lo)} to {fmt(hi)}")
    if args.tree:
        print()
        lines = []
        _tree(model, point_env, model.estimate, 0, set(), lines)
        print("\n".join(lines))
        return EXIT_OK
    print()
    print(f"{'name':<22}{'point':>16}{'low':>16}{'high':>16}")
    for name in model.order:
        low, high = interval_env[name]
        print(f"{name:<22}{fmt(point_env[name][0]):>16}{fmt(low):>16}{fmt(high):>16}")
    return EXIT_OK


def cmd_sim(model, args):
    sim = simulate(model, args.trials, args.seed)
    if args.json:
        payload = {k: v for k, v in sim.items() if k != "samples"}
        payload["expects"] = {
            ex.name: share_inside(sim["samples"], ex.lo, ex.hi) for ex in model.expects
        }
        print(json.dumps(payload, indent=2))
        return EXIT_OK
    print(f"Monte Carlo: {sim['completed']} of {sim['trials']} trials (seed {sim['seed']}, "
          f"{sim['failures']} failed)")
    if sim["first_error"]:
        print(f"  first failure: {sim['first_error']}")
    print(f"  P5 {fmt(sim['p5'])} | P25 {fmt(sim['p25'])} | median {fmt(sim['p50'])} | "
          f"P75 {fmt(sim['p75'])} | P95 {fmt(sim['p95'])}")
    if sim["geo_mean"] is not None:
        print(f"  geometric mean {fmt(sim['geo_mean'])}")
    orders = ", ".join(
        f"{'<=0' if key is None else '10^' + str(key)} {share * 100:.1f}%"
        for key, share in order_shares(sim["samples"])
    )
    print(f"  order of magnitude: {orders}")
    for ex in model.expects:
        print(f"  expect {ex.name} in [{fmt(ex.lo)}, {fmt(ex.hi)}]: "
              f"{share_inside(sim['samples'], ex.lo, ex.hi) * 100:.1f}% of trials")
    return EXIT_OK


def cmd_tornado(model, args):
    base_value, rows = tornado(model)
    if args.json:
        print(json.dumps({"point": base_value, "inputs": rows}, indent=2))
        return EXIT_OK
    print(f"point estimate {fmt(base_value)}")
    print(f"{'input':<22}{'low':>14}{'high':>14}{'output at low':>16}{'output at high':>16}{'swing':>14}{'share':>8}")
    for row in rows:
        print(f"{row['name']:<22}{fmt(row['low_input']):>14}{fmt(row['high_input']):>14}"
              f"{fmt(row['output_at_low']):>16}{fmt(row['output_at_high']):>16}"
              f"{fmt(row['swing']):>14}{row['share'] * 100:>7.1f}%")
    return EXIT_OK


def cmd_score(model, args):
    if model.truth is None:
        print(f"{args.file}: no 'truth' line to score against", file=sys.stderr)
        return EXIT_FAIL
    result = score_model(model, args.trials, args.seed)
    status = EXIT_OK
    if args.max_factor is not None and result["factor"] > args.max_factor:
        status = EXIT_FAIL
    if args.json:
        result["passed"] = status == EXIT_OK
        print(json.dumps(result, indent=2))
        return status
    direction = "over" if result["point"] > result["truth"] else "under"
    print(f"estimate {fmt(result['point'])} vs truth {fmt(result['truth'])}: "
          f"{direction} by {fmt(result['factor'], 3)}x (log10 error {result['log10_error']:+.3f})")
    print(f"truth sits at simulated percentile {result['truth_percentile']:.1f}; "
          f"inside P5-P95: {'yes' if result['inside_p5_p95'] else 'no'}")
    if args.max_factor is not None:
        print(f"factor limit {fmt(args.max_factor, 3)}x: {'PASS' if status == EXIT_OK else 'FAIL'}")
    return status


def cmd_calibrate(worksheets, args):
    """worksheets: list of (path, model) pairs; only those with a truth line are scored."""
    scored = []
    print(f"{'worksheet':<28}{'estimate':>14}{'truth':>14}{'factor':>10}{'in P5-P95':>12}")
    for path, model in worksheets:
        if model.truth is None:
            print(f"{path:<28}  skipped: no truth line")
            continue
        result = score_model(model, args.trials, args.seed)
        scored.append(result)
        name = path if len(path) <= 28 else "..." + path[-25:]
        print(f"{name:<28}{fmt(result['point']):>14}{fmt(result['truth']):>14}"
              f"{fmt(result['factor'], 3) + 'x':>10}{'yes' if result['inside_p5_p95'] else 'no':>12}")
    if not scored:
        print("no worksheet has a truth line to calibrate against", file=sys.stderr)
        return EXIT_FAIL
    summary = calibration_summary(scored)
    print(f"{summary['count']} scored: median factor {fmt(summary['median_factor'], 3)}x, "
          f"{summary['within_2x'] * 100:.0f}% within 2x, "
          f"{summary['inside_p5_p95'] * 100:.0f}% with truth inside P5-P95")
    if args.max_median_factor is not None and summary["median_factor"] > args.max_median_factor:
        print(f"median factor limit {fmt(args.max_median_factor, 3)}x: FAIL")
        return EXIT_FAIL
    return EXIT_OK


def cmd_compare(first, second, args):
    point_a, interval_a = _envs(first)
    point_b, interval_b = _envs(second)
    a = point_a[first.estimate][0]
    b = point_b[second.estimate][0]
    la, ha = interval_a[first.estimate]
    lb, hb = interval_b[second.estimate]
    print(f"{'':<14}{'A':>26}{'B':>26}")
    print(f"{'title':<14}{(first.title or '-')[:24]:>26}{(second.title or '-')[:24]:>26}")
    print(f"{'point':<14}{fmt(a):>26}{fmt(b):>26}")
    print(f"{'guaranteed':<14}{(fmt(la) + ' to ' + fmt(ha)):>26}{(fmt(lb) + ' to ' + fmt(hb)):>26}")
    if a != 0:
        print(f"B / A = {fmt(b / a)}")
    return EXIT_OK


def render_markdown(model, sim, rows):
    point_env, interval_env = _envs(model)
    point = point_env[model.estimate][0]
    lo, hi = interval_env[model.estimate]
    out = [
        f"# {model.title or 'Fermi estimate'}",
        "",
        f"**Estimate:** {fmt(point)} (about {fmt(point, 1)})  ",
        f"**Guaranteed range:** {fmt(lo)} to {fmt(hi)}  ",
        f"**Simulated P5 to P95:** {fmt(sim['p5'])} to {fmt(sim['p95'])}, "
        f"median {fmt(sim['p50'])} ({sim['completed']} trials, seed {sim['seed']})",
        "",
        "## Inputs",
        "",
        "| Input | Point | Low | High | Scale |",
        "|---|---:|---:|---:|---|",
    ]
    for var in _inputs(model):
        scale = "log" if var.log else "linear"
        out.append(f"| `{var.name}` | {fmt(var.point)} | {fmt(var.lo)} | {fmt(var.hi)} | {scale} |")
    out += [
        "",
        "## Sensitivity",
        "",
        "| Input | Output at low | Output at high | Swing | Share |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(f"| `{row['name']}` | {fmt(row['output_at_low'])} | {fmt(row['output_at_high'])} "
                   f"| {fmt(row['swing'])} | {row['share'] * 100:.1f}% |")
    if model.expects:
        out += ["", "## Plausibility checks", "", "| Check | Point | Verdict | Simulated inside |",
                "|---|---:|---|---:|"]
        for ex in model.expects:
            value, verdict, _ = _expect_verdict(ex, point_env, interval_env)
            inside = share_inside(sim["samples"], ex.lo, ex.hi) * 100
            out.append(f"| `{ex.name}` in [{fmt(ex.lo)}, {fmt(ex.hi)}] | {fmt(value)} | {verdict} "
                       f"| {inside:.1f}% |")
    if model.truth is not None:
        result = score_model(model, sim["trials"], sim["seed"])
        out += ["", "## Against the known answer", "",
                f"Truth {fmt(model.truth)}; estimate is off by {fmt(result['factor'], 3)}x; "
                f"truth sits at simulated percentile {result['truth_percentile']:.1f}."]
    return "\n".join(out) + "\n"


def cmd_report(model, args):
    sim = simulate(model, args.trials, args.seed)
    _, rows = tornado(model)
    text = render_markdown(model, sim, rows)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            print(f"fermiwork: cannot write '{args.output}': {exc.strerror}", file=sys.stderr)
            return EXIT_USAGE
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fermiwork",
        description="Offline Fermi-estimate worksheets: ranges, intervals, Monte Carlo and gates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_sim_options(p):
        p.add_argument("-n", "--trials", type=int, default=DEFAULT_TRIALS,
                       help=f"Monte Carlo trials (default {DEFAULT_TRIALS})")
        p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default 1)")

    p = sub.add_parser("check", help="lint the worksheet and run its gates")
    p.add_argument("file")
    p.add_argument("--max-spread", type=float, help="fail if the guaranteed high/low ratio exceeds this")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("eval", help="point values and guaranteed ranges")
    p.add_argument("file")
    p.add_argument("--tree", action="store_true", help="show the derivation of the estimate")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("sim", help="seeded Monte Carlo over the input ranges")
    p.add_argument("file")
    add_sim_options(p)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("tornado", help="rank inputs by how much they move the estimate")
    p.add_argument("file")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_tornado)

    p = sub.add_parser("score", help="compare the estimate with the truth line")
    p.add_argument("file")
    add_sim_options(p)
    p.add_argument("--max-factor", type=float, help="fail if the estimate is off by more than this factor")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("compare", help="compare two worksheets side by side")
    p.add_argument("first")
    p.add_argument("second")
    p.set_defaults(func=None)

    p = sub.add_parser("calibrate", help="score many worksheets with known answers")
    p.add_argument("files", nargs="+")
    add_sim_options(p)
    p.add_argument("--max-median-factor", type=float,
                   help="fail if the median error factor across worksheets exceeds this")
    p.set_defaults(func=None)

    p = sub.add_parser("report", help="write a Markdown worksheet")
    p.add_argument("file")
    add_sim_options(p)
    p.add_argument("-o", "--output", help="write to this file instead of stdout")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        worksheets = []
        for path in args.files:
            model, code = _load(path)
            if model is None:
                return code
            worksheets.append((path, model))
        try:
            return cmd_calibrate(worksheets, args)
        except FermiError as exc:
            print(f"fermiwork: {exc.message}", file=sys.stderr)
            return EXIT_FAIL
    if args.command == "compare":
        first, code = _load(args.first)
        if first is None:
            return code
        second, code = _load(args.second)
        if second is None:
            return code
        return cmd_compare(first, second, args)
    model, code = _load(args.file)
    if model is None:
        return code
    try:
        return args.func(model, args)
    except FermiError as exc:
        print(f"{args.file}: {exc.message}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
