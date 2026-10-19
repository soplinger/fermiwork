# Fermiwork

**Napkin Fermi estimates with honest ranges, and a score against known answers so your guessing measurably improves.**

Fermiwork takes the back-of-the-envelope estimate you would sketch on a napkin and makes it checkable. Write the guess as a worksheet, give each input a range you can defend (log-scale for quantities that span orders of magnitude), and Fermiwork shows the answer, its spread and the inputs that drive it. Record the true answer in the same file and `calibrate` tells you how well your estimates have actually landed across a batch of them.

It is one Python file, standard library only. No network, no accounts, no API keys.

## Features

- **Plain-text worksheets.** One statement per line: `estimate`, `title`, `truth`, `expect`, inputs with ranges, and derived quantities.
- **Uniform and log-uniform inputs.** Use `in [low, high]` for uniform ranges and add `log` for quantities that span orders of magnitude.
- **Safe expression language.** Expressions are parsed with `ast` and checked against a whitelist. Nothing is passed to `eval`.
- **Line-numbered errors.** Syntax errors, unknown names, duplicate definitions, circular references, reversed ranges and points outside their range all report `file:line`.
- **Guaranteed intervals.** Every quantity carries interval bounds through `+ - * / **` and `sqrt ln log10 exp ceil floor abs min max`.
- **Division and power safety.** Division by a range that contains zero, or a power with a varying base that is not above zero, is rejected rather than silently wrong.
- **Seeded Monte Carlo.** `sim` samples the input ranges with a fixed seed, so runs are reproducible. Failed trials are counted and reported, not hidden.
- **Percentiles and orders of magnitude.** P5 to P95, quartiles, geometric mean, and the share of trials in each power of ten.
- **Sensitivity ranking.** `tornado` swings each input across its range with the others held at their points, and ranks inputs by swing and share.
- **Derivation tree.** `eval --tree` shows how the estimate is built from its inputs, with each value.
- **Plausibility checks.** `expect NAME in [LOW, HIGH]` is checked on the point value. The report notes whether the guaranteed interval keeps it inside the range too.
- **Known-answer scoring.** A `truth` line lets `score` report the error factor, the log10 error, where the truth falls in the simulation, and whether it sits inside P5 to P95.
- **Calibration across estimates.** `calibrate` scores a batch of worksheets that have known answers. It reports the median error factor, the share within 2x, and the share whose truth falls inside P5 to P95, so you can see whether your ranges are too narrow or too wide.
- **CI gates.** `check --max-spread`, `score --max-factor` and expectation failures exit with code 1. Lint errors also exit 1. An unreadable file exits 2.
- **Side-by-side comparison.** `compare` lines up two worksheets and shows the ratio of their point estimates.
- **Markdown report.** `report` writes a worksheet with inputs, sensitivity, checks and the truth comparison.
- **JSON output.** `eval`, `sim`, `tornado` and `score` accept `--json` for other tools.

## Quick start

Save this as `piano-tuners.fw` (it is also in `examples/`):

```text
# Fermi estimate: how many piano tuners work in Chicago?
title Piano tuners in Chicago
estimate tuners

population = 2.7e6 in [2.5e6, 2.9e6]        # residents
people_per_home = 2.4 in [2.0, 2.8]         # people per household
homes = population / people_per_home        # households
piano_share = 0.05 in [0.02, 0.10] log      # share of homes with a piano
tunings_per_piano = 1 in [0.5, 1.5]         # tunings per piano per year
hours_per_tuning = 2 in [1.5, 3]            # hours per visit, travel included
work_hours = 1800 in [1400, 2000]           # paid hours a tuner works per year

tuners = ceil(homes * piano_share * tunings_per_piano * hours_per_tuning / work_hours)

expect tuners in [5, 500]
```

Then run:

```sh
python3 fermiwork.py check examples/piano-tuners.fw --max-spread 100
python3 fermiwork.py eval examples/piano-tuners.fw --tree
python3 fermiwork.py sim examples/piano-tuners.fw -n 20000 --seed 42
```

Working the numbers by hand from the worksheet gives a point estimate of **63** (`2.7e6 / 2.4 × 0.05 × 1 × 2 / 1800 = 62.5`, rounded up by `ceil`). The guaranteed range is **7 to 467**, which is inside `expect tuners in [5, 500]`, so that check passes on the interval as well as the point. `sim` then shows where the seeded trials fall within that range.

## Worksheet reference

| Line | Meaning |
|---|---|
| `title TEXT` | Heading for reports |
| `estimate NAME` | The quantity being answered for (required, once) |
| `truth NUMBER` | Known answer, used by `score` |
| `expect NAME in [LOW, HIGH]` | Plausibility bounds, checked by `check` |
| `NAME = EXPR` | Derived quantity |
| `NAME = NUMBER in [LOW, HIGH]` | Input, uniform over the range |
| `NAME = NUMBER in [LOW, HIGH] log` | Input, log-uniform over the range |

Expressions may use `+ - * / **`, parentheses, numbers, earlier names, `pi`, `e`, and the functions `sqrt ln log10 exp ceil floor abs min max`. A `#` starts a comment.

## Commands

```text
fermiwork check FILE [--max-spread R]
fermiwork eval FILE [--tree] [--json]
fermiwork sim FILE [-n TRIALS] [--seed S] [--json]
fermiwork tornado FILE [--json]
fermiwork score FILE [-n TRIALS] [--seed S] [--max-factor F] [--json]
fermiwork calibrate FILE... [-n TRIALS] [--seed S] [--max-median-factor F]
fermiwork compare A B
fermiwork report FILE [-n TRIALS] [--seed S] [-o OUT.md]
```

Exit codes: `0` success, `1` lint error or failed gate, `2` unreadable file.

## How to run

Requires Python 3 (developed against 3.12). There is nothing to install for the tool itself.

```sh
python3 fermiwork.py --help
```

## How to test

The tests use `pytest`, the only third-party dependency, and cover the parser, the interval arithmetic, the simulation, the sensitivity ranking, the scoring and the CLI exit codes.

```sh
python3 -m pip install pytest
python3 -m pytest tests -q
```

## Limits

- Interval bounds are conservative. Repeating a name in an expression can widen the range (`x - x` is not zero over a range). That is the price of a guarantee.
- Monte Carlo samples the inputs independently and uniformly or log-uniformly. Model correlations between inputs are not represented.
- Units are not checked. Keep your own units consistent in the worksheet.

## Layout

```text
fermiwork.py            the whole tool
examples/piano-tuners.fw  worked example
tests/test_fermiwork.py   pytest suite
```
