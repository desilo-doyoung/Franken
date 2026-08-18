"""Periodic progress lines: one line per epoch leaves hours in which a healthy run looks hung."""

import time

import torch

LOG_EVERY = 50  # indented output, so run_experiments' epoch-line trace ignores it


class ProgressLogger:
    # Not tqdm: carriage returns collapse into one line in a redirected log.
    def __init__(self, total_steps, world_size, device, log, enabled=True):
        self.total_steps, self.world_size, self.log = total_steps, world_size, log
        self.enabled, self.every = enabled, LOG_EVERY
        self.step_i = 0
        self._loss = torch.zeros((), device=device)
        self._tokens = torch.zeros((), device=device)
        self._t0 = time.monotonic()

    def step(self, loss, batch):
        self.step_i += 1
        # Reading the accumulators off rank 0 would sync the GPU for nothing.
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
        # Rank 0 sees only its shard; scale to the whole job.
        tok_s = self._tokens.item() * self.world_size / elapsed
        finish = time.strftime("%H:%M", time.localtime(time.time() + remaining))
        self.log(
            f"  step {self.step_i}/{self.total_steps} "
            f"({100 * self.step_i / self.total_steps:.0f}%) "
            f"loss {self._loss.item() / self.every:.4f} | {tok_s / 1000:.1f}k tok/s | "
            f"{per_step:.2f}s/step | eta {remaining / 3600:.1f}h -> {finish}"
        )
        self._loss.zero_()
        self._tokens.zero_()
        self._t0 = time.monotonic()
