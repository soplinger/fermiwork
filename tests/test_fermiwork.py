import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import fermiwork  # noqa: E402
from fermiwork import FermiError, fmt, load_text, main, percentile, score_model, simulate, tornado  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "piano-tuners.fw")


def model_of(text):
    model, errors = load_text(text)
    assert not errors, errors
    return model


def errors_of(text):
    _, errors = load_text(text)
    return errors


def test_parse_reads_inputs_derived_and_estimate():
    model = model_of(
        "title Demo\n"
        "estimate total\n"
        "a = 2 in [1, 3]\n"
        "b = a * 10\n"
        "total = b + 1\n"
    )
    assert model.title == "Demo"
    assert model.estimate == "total"
    assert model.vars["a"].is_input
    assert not model.vars["b"].is_input
    assert model.order.index("a") < model.order.index("b") < model.order.index("total")


def test_syntax_error_reports_its_line_number():
    errors = errors_of("estimate x\nx = 1\ny = 2 +* 3\n")
    assert any(line == 3 and "syntax error" in msg for line, msg in errors)


def test_unknown_name_reports_line_of_the_reference():
    errors = errors_of("estimate x\nx = 1\ny = x + missing\n")
    assert any(line == 3 and "unknown name 'missing'" in msg for line, msg in errors)


def test_circular_reference_is_detected():
    errors = errors_of("estimate a\na = b + 1\nb = a + 1\n")
    assert any("circular reference" in msg for _, msg in errors)


def test_point_outside_its_range_is_rejected():
    errors = errors_of("estimate x\nx = 5 in [1, 3]\n")
    assert any(line == 2 and "outside its range" in msg for line, msg in errors)


def test_reversed_range_is_rejected():
    errors = errors_of("estimate x\nx = 2 in [3, 1]\n")
    assert any(line == 2 and "reversed" in msg for line, msg in errors)


def test_point_and_interval_arithmetic():
    model = model_of("estimate y\na = 2 in [1, 3]\ny = a * 10\n")
    assert fermiwork._point(model) == pytest.approx(20)
    low, high = fermiwork._estimate(model, fermiwork.interval_inputs(model))
    assert (low, high) == pytest.approx((10, 30))


def test_division_by_a_range_containing_zero_fails_validation():
    errors = errors_of("estimate y\nx = 1 in [-1, 1]\ny = 5 / x\n")
    assert any(line == 3 and "division by zero" in msg for line, msg in errors)


def test_power_with_varying_base_needs_positive_base():
    errors = errors_of("estimate y\nx = 1 in [-1, 2]\ny = x ** 2\n")
    assert any("needs the base above zero" in msg for _, msg in errors)


def test_unknown_function_and_unsafe_syntax_are_rejected():
    errors = errors_of("estimate y\ny = __import__('os')\n")
    assert any(line == 2 and "unknown function" in msg for line, msg in errors)
    errors = errors_of("estimate y\ny = 2 < 3\n")
    assert any(line == 2 and "unsupported syntax" in msg for line, msg in errors)


def test_ceil_and_negation_evaluate_over_intervals():
    model = model_of("estimate n\nx = 2.5 in [1.5, 2.5]\nn = ceil(-x * 2)\n")
    low, high = fermiwork._estimate(model, fermiwork.interval_inputs(model))
    assert (low, high) == (-5, -3)


def test_simulation_is_reproducible_and_percentiles_are_ordered():
    model = model_of("estimate y\na = 1 in [1, 5]\nb = 2 in [1, 9] log\ny = a * b\n")
    first = simulate(model, trials=500, seed=7)
    second = simulate(model, trials=500, seed=7)
    other = simulate(model, trials=500, seed=8)
    assert first["samples"] == second["samples"]
    assert first["samples"] != other["samples"]
    assert first["p5"] <= first["p50"] <= first["p95"]
    low, high = fermiwork._estimate(model, fermiwork.interval_inputs(model))
    assert low <= first["samples"][0] and first["samples"][-1] <= high


def test_log_uniform_draws_stay_inside_their_range():
    var = fermiwork.Var("x", "1", 1, None, [], lo=0.02, hi=0.1, log=True)
    draws = [fermiwork._draw(var, u / 100) for u in range(101)]
    assert min(draws) >= 0.02 and max(draws) <= 0.1
    assert draws == sorted(draws)


def test_tornado_ranks_the_widest_input_first():
    model = model_of(
        "estimate out\n"
        "x = 10 in [1, 100]\n"
        "y = 10 in [9, 11]\n"
        "out = x * y\n"
    )
    base, rows = tornado(model)
    assert base == pytest.approx(100)
    assert [row["name"] for row in rows] == ["x", "y"]
    assert rows[0]["swing"] == pytest.approx(990)
    assert rows[1]["swing"] == pytest.approx(20)
    assert sum(row["share"] for row in rows) == pytest.approx(1)


def test_percentile_interpolates_linearly():
    assert percentile([0, 10], 50) == pytest.approx(5)
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


def test_fmt_uses_separators_and_scientific_notation():
    assert fmt(2700000) == "2.7 × 10^6"
    assert fmt(2100) == "2,100"
    assert fmt(0.5) == "0.5"
    assert fmt(0) == "0"


def test_score_reports_factor_against_truth():
    model = model_of("estimate x\ntruth 50\nx = 100\n")
    result = score_model(model, trials=200, seed=1)
    assert result["factor"] == pytest.approx(2)
    assert result["log10_error"] == pytest.approx(0.30103, rel=1e-4)


def test_cli_check_exit_codes(tmp_path, capsys):
    good = tmp_path / "good.fw"
    good.write_text("estimate x\nx = 2 in [1, 3]\nexpect x in [0, 10]\n", encoding="utf-8")
    assert main(["check", str(good)]) == 0

    failing_expect = tmp_path / "expect.fw"
    failing_expect.write_text("estimate x\nx = 2 in [1, 3]\nexpect x in [5, 10]\n", encoding="utf-8")
    assert main(["check", str(failing_expect)]) == 1
    assert "FAIL" in capsys.readouterr().out

    broken = tmp_path / "broken.fw"
    broken.write_text("estimate x\nx = 2 +* 3\n", encoding="utf-8")
    assert main(["check", str(broken)]) == 1
    assert "broken.fw:2:" in capsys.readouterr().err

    assert main(["check", str(tmp_path / "missing.fw")]) == 2


def test_cli_spread_gate_and_score_gate(tmp_path):
    wide = tmp_path / "wide.fw"
    wide.write_text("estimate x\nx = 1 in [1, 100]\n", encoding="utf-8")
    assert main(["check", str(wide), "--max-spread", "10"]) == 1

    scored = tmp_path / "scored.fw"
    scored.write_text("estimate x\ntruth 50\nx = 100\n", encoding="utf-8")
    assert main(["score", str(scored), "-n", "100", "--max-factor", "1.5"]) == 1
    assert main(["score", str(scored), "-n", "100", "--max-factor", "3"]) == 0


def test_report_writes_markdown_sections(tmp_path):
    out = tmp_path / "report.md"
    assert main(["report", EXAMPLE, "-n", "300", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    for heading in ("# Piano tuners in Chicago", "## Inputs", "## Sensitivity", "## Plausibility checks"):
        assert heading in text


def test_example_worksheet_passes_its_own_checks():
    assert main(["check", EXAMPLE, "--max-spread", "1000"]) == 0


def test_calibration_summary_aggregates_scores():
    summary = fermiwork.calibration_summary([
        {"factor": 2.0, "inside_p5_p95": True},
        {"factor": 1.2, "inside_p5_p95": False},
    ])
    assert summary["count"] == 2
    assert summary["median_factor"] == pytest.approx(1.6)
    assert summary["within_2x"] == pytest.approx(1.0)
    assert summary["inside_p5_p95"] == pytest.approx(0.5)


def test_cli_calibrate_median_gate_and_missing_truth(tmp_path, capsys):
    high = tmp_path / "high.fw"
    high.write_text("estimate x\ntruth 50\nx = 100\n", encoding="utf-8")   # off by 2x
    close = tmp_path / "close.fw"
    close.write_text("estimate x\ntruth 100\nx = 100\n", encoding="utf-8")  # exact
    assert main(["calibrate", str(high), str(close), "-n", "50", "--max-median-factor", "1.2"]) == 1
    assert main(["calibrate", str(high), str(close), "-n", "50", "--max-median-factor", "2"]) == 0
    assert "2 scored" in capsys.readouterr().out

    plain = tmp_path / "plain.fw"
    plain.write_text("estimate x\nx = 100\n", encoding="utf-8")
    assert main(["calibrate", str(plain), "-n", "50"]) == 1
