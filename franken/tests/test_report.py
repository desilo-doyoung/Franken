import pytest

from franken.scripts.qwen3 import report
from franken.scripts.qwen3.run_experiments import deficits, render


def test_relative_delta_is_signed_against_the_teacher():
    assert report.relative_delta(0.5, 0.4) == pytest.approx(-20.0)
    assert report.relative_delta(0.5, 0.6) == pytest.approx(20.0)


def test_relative_delta_survives_a_zero_teacher():
    assert report.relative_delta(0.0, 0.4) == 0.0


def test_quality_columns_align_with_the_blank_form():
    # A suppressed nDCG row must occupy the same width as a scored one or the table shears.
    assert len(report.quality(0.5, 0.4)) == len(report.BLANK_QUALITY)


def test_task_row_keeps_the_name_column_width():
    line = report.task_row("gooaq", "pair", 500, 5000, report.quality(0.5, 0.4), "q -> a")
    assert line.startswith(f"{'gooaq':>18}")
    assert line.endswith("q -> a")


def test_macro_row_reports_the_group_size():
    assert report.macro_row("MACRO", 0.5, 0.4, 3).lstrip().startswith("MACRO(3)")


def test_fidelity_block_names_the_metric_it_prints():
    block = report.fidelity_block(
        {
            "pool": 500,
            "recall@10": 0.9,
            "embed_dist": 0.01,
            "stsb_teacher": 0.8,
            "stsb_student": 0.79,
        }
    )
    assert "recall@10" in block and "not MTEB's recall" in block
    assert "-0.0100" in block  # the STS-B delta, signed


# --------------------------------------------------------------------- run_experiments


def run_row(**over):
    row = {
        "stem": "d19_quad",
        "depth": 19,
        "softmax": "exact",
        "activation": "quad_silu",
        "minutes": 90.0,
        "trace": [],
        "k": 10,
        "recall": 0.9,
        "ndcg": 0.4,
        "ndcg_teacher": 0.5,
        "embed_dist": 0.01,
        "stsb_teacher": 0.8,
        "stsb_student": 0.76,
        "ndcg_tasks": {},
    }
    return row | over


def test_deficits_are_relative_to_the_teacher():
    d = deficits(run_row())
    assert d["recall_def"] == pytest.approx(0.1)
    assert d["ndcg_def"] == pytest.approx(0.2)
    assert d["ratio"] == pytest.approx(2.0)
    assert d["stsb_delta"] == pytest.approx(-0.04)
    assert d["stsb_rel"] == pytest.approx(-5.0)


def test_ratio_is_undefined_when_the_student_matched_the_teacher():
    assert deficits(run_row(recall=1.0))["ratio"] is None


def test_render_puts_a_successful_run_in_the_table():
    lines = render([run_row()])
    assert any(line.startswith("| d19_quad | 19 | exact/quad_silu") for line in lines)


def test_render_keeps_failures_out_of_the_table_but_names_them():
    lines = render([run_row(), {"stem": "d11_quad", "error": "distill exit 1", "log": "x.log"}])
    table = [ln for ln in lines if ln.startswith("| d")]
    assert len(table) == 1 and "d11_quad" not in "".join(table)
    assert any("FAILED d11_quad: distill exit 1" in ln for ln in lines)


def test_render_warns_when_the_teacher_differs_across_runs():
    lines = render([run_row(), run_row(stem="other", stsb_teacher=0.7)])
    assert any("WARNING" in ln for ln in lines)
    assert not any("WARNING" in ln for ln in render([run_row(), run_row(stem="other")]))


def test_render_pairs_the_two_gold_free_fidelity_columns():
    # recall@10 catches a moved ranking, embed_dist a moved geometry; the table needs both.
    line = next(ln for ln in render([run_row(embed_dist=0.022520)]) if ln.startswith("| d19_quad"))
    assert "0.9000" in line  # recall@10
    assert "0.022520" in line  # embed_dist, at full precision -- it lives in the 3rd decimal


def test_render_omits_in_dist_and_the_coverage_gap():
    # Both are ~constant across every run measured, so gap == external minus a constant. They stay
    # in results.json; a column that never moves is not a column.
    row = run_row(macro_pair={"teacher": 0.9, "student": 0.8982, "n": 9}, coverage_gap=-8.3)
    line = next(ln for ln in render([row]) if ln.startswith("| d19_quad"))
    assert "-8.3%" not in line
    assert line.count("|") == 8  # run, depth, ops, recall, embed_dist, external, min


def test_render_shows_per_task_external_as_relative_deltas():
    # The macro averages a destroyed task with a flat one; only the per-task row shows that.
    row = run_row(
        ndcg_tasks={
            "code_apps": {"teacher": 0.7317, "student": 0.4920},
            "scifact": {"teacher": 0.7015, "student": 0.7019},
        }
    )
    lines = render([row])
    task_line = [ln for ln in lines if ln.startswith("| d19_quad")][-1]
    assert "-32.8%" in task_line and "+0.1%" in task_line


def test_render_survives_a_run_with_no_external_suite():
    line = next(
        ln for ln in render([run_row(ndcg=None, ndcg_teacher=None)]) if ln.startswith("| d19_quad")
    )
    assert line.count("—") >= 1


def test_render_is_pure():
    results = [run_row()]
    before = repr(results)
    render(results)
    assert repr(results) == before
