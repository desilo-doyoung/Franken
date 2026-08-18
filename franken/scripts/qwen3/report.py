"""Console layout for `eval.py`, kept apart from the scoring so a table can be produced and tested
without a model, and so the scorers can run with no console side effects at all."""

from __future__ import annotations

from franken.metrics import K

# The four quality columns, blanked where a source declares `scores_ndcg: false` -- a printed
# number gets read as one.
BLANK_QUALITY = f"{'-':>9} {'-':>9} {'-':>9} {'-':>8}"


def silent(_line: str) -> None:
    """Default sink: compute, print nothing."""


def relative_delta(teacher: float, student: float) -> float:
    """Student's loss against the teacher, in percent -- the reference every claim is about."""
    return 100 * (student - teacher) / teacher if teacher else 0.0


def quality(teacher: float, student: float) -> str:
    return f"{teacher:>9.4f} {student:>9.4f} {student - teacher:>+9.4f} {relative_delta(teacher, student):>7.1f}%"  # noqa: E501


def header(what: str, metric: str) -> str:
    # The metric always goes in the header: `recall@10` here means teacher-neighbour agreement,
    # MTEB's means something else, so an unlabelled column is a number waiting to be misread.
    return (
        f"\n== {what} -- {metric} ==\n{'task':>18} {'kind':>6} {'q':>5} {'docs':>6} "
        f"{'teacher':>9} {'student':>9} {'delta':>9} {'rel':>8}   retrieves"
    )


def corpus_header(corpus: str, split: str) -> str:
    return (
        f"\n== corpus: held-out rows of {corpus}, split={split} ==\n"
        f"   quality = nDCG@{K} (teacher/student/delta/rel);  fidelity = recall@{K} + embed_dist\n"
        f"   both over the task's whole doc pool. Read across MODELS: `docs` differs per task and\n"
        f"   both metrics are pool-size dependent, so task-to-task is not comparable\n"
        f"{'task':>18} {'kind':>6} {'q':>5} {'docs':>6} {'teacher':>9} {'student':>9} "
        f"{'delta':>9} {'rel':>8} {f'recall@{K}':>9} {'dist':>8}   retrieves"
    )


def empty_row(name: str, kind: str = "") -> str:
    return f"{name:>18} {kind:>6}   no queries"


def task_row(name: str, kind: str, queries: int, docs: int, cells: str, shape: str = "") -> str:
    return f"{name:>18} {kind:>6} {queries:>5} {docs:>6} {cells}   {shape}"


def macro_row(label: str, teacher: float, student: float, n: int) -> str:
    return f"{f'{label}({n})':>18} {'':>14} {'':>5} {'':>6} {quality(teacher, student)}"


def domain_row(domain: str, n: int, teacher: float, student: float) -> str:
    return f"  {domain:<14} n={n}  {teacher:.4f} -> {student:.4f}  {student - teacher:+.4f}"


def unscored_note(blind: list[str], share: float) -> str:
    return (
        f"nDCG not scored ({len(blind)}, {share:.0%} of the corpus): {', '.join(blind)}\n"
        f"  gold is one arbitrary member of an equally valid set; read their recall@{K}."
    )


def fidelity_block(out: dict) -> str:
    return "\n".join(
        [
            f"\n== agreement: {out['pool']} held-out corpus texts -- recall@{K} vs THIS teacher "
            f"(not MTEB's recall) ==",
            f"  recall@{K}     {out[f'recall@{K}']:.4f}   of the teacher's top-{K} neighbours "
            f"found; teacher = 1.0 by construction",
            f"  embed_dist    {out['embed_dist']:.6f}   (per-vector; logging only, it misranks)",
            f"  STS-B         teacher {out['stsb_teacher']:.4f}  student {out['stsb_student']:.4f}"
            f"  delta {out['stsb_student'] - out['stsb_teacher']:+.4f}",
        ]
    )


def coverage_gap_block(in_dist: float, off_dist: float) -> str:
    return (
        f"\nin-distribution {in_dist:+.1f}%   external {off_dist:+.1f}%   "
        f"coverage gap {off_dist - in_dist:+.1f}%\n"
        f"  a large gap means the fix is corpus coverage; both large means capacity."
    )
