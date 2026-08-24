"""Score an LM student against its teacher: perplexity, top-1 agreement, KL.

Needs no pools, no judgements and no external benchmarks -- the teacher's own distribution is the
reference. Run with no --student-ckpt on a full-depth exact-op config for the identity self-test:
agreement must read 1.0000 and ppl must equal teacher_ppl.

    uv run python -m franken.scripts.llama.lm_eval --config configs/llama/smoke.yaml
"""

from __future__ import annotations

import json

from franken.scripts import common


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    # Validation selects the checkpoint, test reports it; never score the model on the split that
    # picked it.
    p.add_argument("--split", default="validation", choices=("validation", "test"))
    args = p.parse_args(argv)

    m = common.load(args)
    metrics = m.task.evaluate(
        m.backend, m.student, m.tokenizer, m.cfg, split=args.split, teacher=m.teacher
    )

    print(f"\n{args.split}  ({m.cfg.train.corpus})")
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.4f}")
    print(f"  {'ppl delta':<14} {metrics['ppl'] / metrics['teacher_ppl'] - 1:+.2%}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nwrote {args.json}")
