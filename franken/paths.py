"""Output paths, namespaced per run so two models never collide."""

from __future__ import annotations

import os

from franken.config import Config

# Repo root, so a cache identity does not move with the process CWD.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    @property
    def student_bin(self) -> str:
        return os.path.join(self.student, "pytorch_model.bin")

    def subdir(self, name: str) -> str:
        return os.path.join(self.base, name)
