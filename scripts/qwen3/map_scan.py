"""Where should the removed block of layers sit? Zero training — seed + score only.

Depth reduction removes `teacher_depth - depth` layers. Dropping them as one contiguous run beat
uniform striding by 4-6x, but "the middle" was a guess. This scans every position for that run and
prints the resulting init quality, so the choice is measured rather than assumed. Uniform stride is
included as the reference the repo defaults to.

Cheap (one student forward per candidate over the eval pool), and worth running before any
distillation at a new depth: the map mattered more than the depth did.

Usage:
    uv run python scripts/qwen3/map_scan.py --config configs/qwen3/depth19.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F
from embed_eval import K, _embed, _neighbour_agreement
from franken.config import Config
from franken.distill.layer_map import resolve_layer_map
from franken.models import build_backend
from franken.tasks import build_task
from torch.utils.data import DataLoader


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen3/depth19.yaml")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backend, task = build_backend(cfg.model.backend), build_task(cfg.train.task)
    tokenizer = task.build_tokenizer(cfg)

    teacher = backend.load_teacher(cfg).to(device)
    full, depth = teacher.config.num_hidden_layers, cfg.model.num_hidden_layers
    removed = full - depth

    data = task.datasets(tokenizer, cfg)
    ds = data["validation"].with_format("torch", columns=task.torch_columns())
    batches = list(DataLoader(ds, batch_size=16, collate_fn=data["collator"]))
    t_emb = _embed(backend, teacher, task, batches, device)

    def score(layer_map):
        cfg.distill.hidden_layer_map = layer_map
        student = backend.build_student(cfg)
        backend.seed_student(student, teacher, cfg)
        student = student.to(device).eval()
        s_emb = _embed(backend, student, task, batches, device)
        dist = (1 - F.cosine_similarity(s_emb, t_emb, dim=-1)).mean().item()
        recall, rho = _neighbour_agreement(s_emb, t_emb)
        del student, s_emb
        torch.cuda.empty_cache()
        return dist, recall, rho

    print(f"\ndepth {depth}/{full}: {removed} layers removed | pool {t_emb.size(0)} texts")
    print(f"{'removed block':>16} {'embed_dist':>11} {f'recall@{K}':>10} {'sim-rho':>9}")

    results = []
    for start in range(full - removed + 1):
        drop = set(range(start, start + removed))
        layer_map = [i for i in range(full) if i not in drop]
        dist, recall, rho = score(layer_map)
        results.append((recall, start, dist, rho))
        print(f"{f'{start}-{start + removed - 1}':>16} {dist:>11.4f} {recall:>10.4f} {rho:>9.4f}")

    dist, recall, rho = score(resolve_layer_map(full, depth))
    print(f"{'uniform stride':>16} {dist:>11.4f} {recall:>10.4f} {rho:>9.4f}   (repo default)")

    # Do the three metrics rank candidates the same way? If rho tracks recall it is the better
    # selection metric (same ranking, ~50x more pairs so far less noise); if it tracks embed_dist
    # it is measuring the same wrong thing.
    for name, key in (("embed_dist (min)", lambda r: -r[2]), ("recall (max)", lambda r: r[0]), ("rho (max)", lambda r: r[3])):
        best = max(results, key=key)
        print(f"  best by {name:<18} -> block {best[1]}-{best[1] + removed - 1}")

    best_recall, best_start, best_dist, _ = max(results)
    mid = depth // 2
    print(f"\nbest: remove {best_start}-{best_start + removed - 1} -> recall@{K} {best_recall:.4f}")
    print(f"drop-the-middle ({mid}-{mid + removed - 1}) is what the configs use today")
    print(f"kept layers for best: {[i for i in range(full) if not (best_start <= i < best_start + removed)]}\n")


if __name__ == "__main__":
    main()
