"""Per-model output path layout.

All output subdirectories live under a per-model namespace so different models
never collide: ``<output_dir>/<namespace>/{teacher,student,...}``. The namespace is
``cfg.train.run_name`` when set, else the model backend name (``cfg.model.backend``).
So BERT writes to ``outputs/bert/...`` and Qwen3 to ``outputs/qwen3/...`` by default,
while ``run_name`` lets a specific experiment carve out its own subtree.
"""

from __future__ import annotations

import os

from franken.config import Config


class RunPaths:
    def __init__(self, cfg: Config):
        namespace = cfg.train.run_name or cfg.model.backend
        self.base = os.path.join(cfg.train.output_dir, namespace)

    @property
    def teacher(self) -> str:
        return os.path.join(self.base, "teacher")

    @property
    def student(self) -> str:
        return os.path.join(self.base, "student")

    def student_bin(self) -> str:
        return os.path.join(self.student, "pytorch_model.bin")

    def subdir(self, name: str) -> str:
        """Arbitrary named subdir under the run base (e.g. stageA_quad, seed_sweep)."""
        return os.path.join(self.base, name)
