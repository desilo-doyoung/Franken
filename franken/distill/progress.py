"""Periodic progress reporting for long distillation runs.

The training loop otherwise emits one line per epoch, which on a multi-thousand-step run is hours
of silence — long enough that a healthy run is indistinguishable from a hung one.
"""

import time

import torch

# Steps between lines. Output is indented, so run_experiments' epoch-line trace ignores it.
LOG_EVERY = 50


class ProgressLogger:
    """Throughput + finish estimate every `every` steps.

    Loss and token counts accumulate on-device and are read once per line: a per-step `.item()`
    would sync the GPU on every iteration. Disabled off rank 0, which owns both the sync and the
    output.

    Deliberately not a tqdm bar: this output's destination is a redirected log file, where tqdm's
    carriage returns collapse into one unreadable line."""

    def __init__(self, total_steps, world_size, device, log, enabled=True, every=LOG_EVERY):
        self.total_steps, self.world_size, self.log = total_steps, world_size, log
        self.enabled, self.every = enabled, every
        self.step_i = 0
        self._loss = torch.zeros((), device=device)
        self._tokens = torch.zeros((), device=device)
        self._t0 = time.monotonic()

    def step(self, loss, batch):
        self.step_i += 1
        if not self.enabled:
            return
        self._loss += loss.detach()
        mask = batch.get("attention_mask")
        if mask is not None:
            self._tokens += mask.sum()
        if self.step_i % self.every:
            return

        elapsed = time.monotonic() - self._t0
        per_step = elapsed / self.every
        remaining = (self.total_steps - self.step_i) * per_step
        # Rank 0 sees only its own shard, so scale its token count to the whole job.
        tok_s = self._tokens.item() * self.world_size / elapsed
        loss_avg = self._loss.item() / self.every
        finish = time.strftime("%H:%M", time.localtime(time.time() + remaining))
        self.log(
            f"  step {self.step_i}/{self.total_steps} "
            f"({100 * self.step_i / self.total_steps:.0f}%) "
            f"loss {loss_avg:.4f} | {tok_s / 1000:.1f}k tok/s | "
            f"{per_step:.2f}s/step | eta {remaining / 3600:.1f}h -> {finish}"
        )
        self._loss.zero_()
        self._tokens.zero_()
        self._t0 = time.monotonic()
