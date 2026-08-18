import math
import random
from contextlib import nullcontext

import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup, set_seed

from franken.config import PRECISIONS, Config
from franken.distill.batching import plan_batches, shard
from franken.distill.dist import barrier, init_distributed, per_rank_batch
from franken.distill.progress import ProgressLogger
from franken.models import build_backend
from franken.tasks import build_task

# Reference for sqrt-batch LR scaling (`lr: null`): bs32/lr2e-5, fidelity-verified at depth 24
# (recall@10 0.9070, identical to fp32/bs8/lr1e-5). sqrt not linear because Adam-family.
BASE_LR, BASE_BATCH = 2e-5, 32
# Cap on `warmup_ratio`: warmup covers early-Adam instability, which is a fixed number of steps, not
# a share of the run -- uncapped, changing `epochs` moves it and the two runs stop being comparable.
MAX_WARMUP_STEPS = 2000


def _apply_precision(precision: str) -> None:
    # Training loop only; eval stays fp32. TF32 lowers the *teacher's targets* too, not just the
    # student's math, so check against an fp32 reference before trusting the speedup.
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}; use {' | '.join(PRECISIONS)}")
    if precision in ("tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _autocast(precision: str):
    # STUDENT forward + loss only. Never the teacher: it makes the targets, and a bf16 teacher
    # shifts them 0.0076 recall@10, ~2x the comparison band.
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def _maybe_compile(model, cfg: Config):
    # Callers keep the eager module too: its state_dict has no `_orig_mod.` prefix, and eval must
    # stay eager (see evaluate).
    return torch.compile(model) if cfg.train.compile else model


def resolve_lr(opt, global_batch: float, log) -> float:
    """`lr: null` derives the rate by sqrt-batch scaling. Under token budgeting the batch floats to
    fill the budget and is only known once the plan exists, so a hardcoded `lr` goes stale when the
    corpus or rank count moves -- the last ladder ran 5.6% hot exactly that way."""
    if opt.lr is not None:
        return opt.lr
    lr = BASE_LR * math.sqrt(global_batch / BASE_BATCH)
    log(
        f"lr {lr:.4e} = {BASE_LR:g} * sqrt({global_batch:.0f} / {BASE_BATCH})"
        f" [sqrt-batch scaling from the tuned reference]"
    )
    return lr


class BatchLoader:
    """Token-budgeted plan, DistributedSampler, or plain shuffle. Owns the per-epoch loader:
    steps/epoch must not move, or the LR schedule stops matching the recorded runs."""

    def __init__(self, cfg: Config, dist, dataset, collator, log):
        self.opt = cfg.train.distill
        self.dist, self.dataset, self.collator = dist, dataset, collator
        self.seed = cfg.train.seed
        self.sampler = self.plan = None

        if self.opt.token_budget:
            lengths = pc.list_value_length(dataset.data.column("input_ids")).to_numpy(
                zero_copy_only=False
            )
            self.plan = shard(
                plan_batches(lengths, self.opt.token_budget, self.opt.max_seqs, self.seed),
                dist.rank,
                dist.world_size,
            )
            log(
                f"token-budgeted batching: {len(self.plan):,} steps/epoch/rank, "
                f"{self.opt.token_budget:,} tokens x {dist.world_size} ranks per step"
            )
        elif dist.enabled:
            # Single-process keeps the literal shuffle=True path: DistributedSampler draws a
            # different permutation even at world_size 1, and batch composition is worth ~0.004
            # recall -- the width of the comparison band itself.
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=dist.world_size,
                rank=dist.rank,
                shuffle=True,
                seed=self.seed,
            )

        self._loader = self._build(0)
        # DistributedSampler pads to divide evenly, so matching single-process steps/epoch is a
        # happy accident, not a guarantee -- and a mismatch silently rescales the LR schedule.
        # (`shard` gives the token-budgeted path the same guarantee by construction.)
        if dist.enabled and self.plan is None:
            expected = -(-len(dataset) // self.opt.batch_size)
            if len(self._loader) != expected:
                raise RuntimeError(
                    f"steps/epoch {len(self._loader)} != single-process {expected}; the LR "
                    "schedule and recorded results would not be comparable."
                )

    def _build(self, epoch: int) -> DataLoader:
        if self.plan is None:
            return DataLoader(
                self.dataset,
                batch_size=per_rank_batch(self.opt.batch_size, self.dist),
                shuffle=self.sampler is None,
                sampler=self.sampler,
                collate_fn=self.collator,
            )
        # One plan, reordered per epoch. Re-planning would vary steps/epoch with the shuffle and
        # drift the LR schedule; order alone is enough to decorrelate the epochs.
        order = list(self.plan)
        random.Random(self.seed + epoch).shuffle(order)
        return DataLoader(self.dataset, batch_sampler=order, collate_fn=self.collator)

    def loader(self, epoch: int) -> DataLoader:
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        if self.plan is not None and epoch:
            self._loader = self._build(epoch)
        return self._loader

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def global_batch(self) -> float:
        """Sequences per step across all ranks -- what sqrt-batch LR scaling reads."""
        if self.plan is None:
            return float(self.opt.batch_size)  # already the GLOBAL batch
        seqs = sum(len(b) for b in self.plan) / max(len(self.plan), 1)
        return seqs * self.dist.world_size


class RangePenalty:
    """Pulls FFN pre-activations into the activation op's domain, via forward hooks that live
    only for the `with` body. Falsy unless `range_penalty > 0` and the op exposes a `domain`."""

    def __init__(self, backend, student, cfg: Config, log):
        self.weight = cfg.distill.range_penalty
        acts = backend.activation_ops(student)
        first = acts[0] if acts else None
        self.domain = getattr(first, "domain", None) if (self.weight > 0 and first) else None
        self._preacts, self._hooks, self._epoch = [], [], []
        self._targets = []
        if self.domain is None:
            return
        # Hook only the penalized layers: constraining one costs accuracy, so the default (all of
        # them) is usually wrong -- see range_penalty_layers. Bounds checked in Config.validate.
        mods = backend.ffn_preact_modules(student)
        which = cfg.distill.range_penalty_layers
        self._targets = mods if which is None else [mods[i] for i in which]
        if which is not None:
            log(
                f"range penalty on student layers {sorted(which)} "
                f"of {len(mods)}, domain {self.domain}"
            )

    def __bool__(self) -> bool:
        return self.domain is not None

    def __enter__(self):
        self._hooks = [m.register_forward_hook(self._capture) for m in self._targets]
        return self

    def __exit__(self, *exc) -> bool:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        return False

    def _capture(self, module, _inp, out) -> None:
        if module.training:
            self._preacts.append(out)

    def clear(self) -> None:
        self._preacts.clear()

    def measure(self):
        """Squared distance past +/-domain for the batch just forwarded, meaned over the
        OUT-OF-RANGE elements only -- averaging over all of them would let the in-range bulk dilute
        the gradient on the rare outliers. None if everything is in range."""
        terms = []
        for x in self._preacts:
            # fp32 always: a tail statistic, and bf16's coarse grid up there shifts it +7.6%.
            x = x.float()
            over, under = F.relu(x - self.domain), F.relu(-self.domain - x)
            outside = (over > 0) | (under > 0)
            if outside.any():
                terms.append(((over**2 + under**2)[outside]).mean())
        if not terms:
            return None
        value = torch.stack(terms).mean()
        self._epoch.append(value.detach())
        return value

    def epoch_mean(self) -> float | None:
        """Mean over the epoch, then reset. The only visible evidence the penalty is doing anything
        (it should fall). Verify the end state with franken/scripts/qwen3/act_range.py."""
        if not self._epoch:
            return None
        mean = torch.stack(self._epoch).mean().item()
        self._epoch.clear()
        return mean


class BestCheckpoint:
    """Best-scoring student weights seen so far, by the task's own selection metric."""

    def __init__(self, metric_name: str, higher_is_better: bool):
        self.metric_name, self.higher_is_better = metric_name, higher_is_better
        self.best = float("-inf") if higher_is_better else float("inf")
        self.state = None

    def consider(self, metrics: dict, student) -> bool:
        value = metrics[self.metric_name]
        if not (value > self.best if self.higher_is_better else value < self.best):
            return False
        self.best = value
        # Clone off-device: state_dict() hands back live references that the next optimizer.step()
        # would mutate in place.
        self.state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        return True

    def restore(self, student) -> None:
        if self.state is not None:
            student.load_state_dict(self.state)


class Distiller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dist = init_distributed()
        if self.dist.enabled:
            self.device = torch.device(f"cuda:{self.dist.local_rank}")
        else:
            self.device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        self.backend = build_backend(cfg.model.backend)
        self.task = build_task(cfg.train.task)
        self.teacher = None
        self.student = None
        self.tokenizer = None

    def log(self, *args):
        # Rank 0 only (run_experiments.py parses these lines), and flushed -- stdout is
        # block-buffered into a log file, so an unflushed print leaves `tail -f` empty all run.
        if self.dist.is_main:
            print(*args, flush=True)

    def setup(self):
        self.teacher = self.backend.load_teacher(self.cfg).to(self.device)
        self.student = self.backend.build_student(self.cfg)
        self.tokenizer = self.task.build_tokenizer(self.cfg)

        # strided weight init (backend owns the model-specific remapping)
        self.backend.seed_student(self.student, self.teacher, self.cfg)
        self.student.to(self.device)

    def _wrap_for_training(self):
        # Training only; self.student/self.teacher stay eager for evaluate() and the save.
        # DDP wraps FIRST: compiling the wrapper lets Dynamo's DDPOptimizer split the graph so
        # allreduce overlaps backward. The other order fuses one graph whose grads all become
        # ready at once, exposing the full ~2.4 GB of comms.
        student = self.student
        if self.dist.enabled:
            student = DistributedDataParallel(
                student, device_ids=[self.dist.local_rank], gradient_as_bucket_view=True
            )
        return _maybe_compile(student, self.cfg), _maybe_compile(self.teacher, self.cfg)

    def _run_epoch(self, loader, student, teacher, optimizer, scheduler, penalty, progress):
        """One pass over the loader; returns the last batch's loss components for the epoch line."""
        components = {}
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            inputs = self.task.model_inputs(batch)

            # Outside the autocast region: see _autocast.
            with torch.no_grad():
                teacher_outputs = self.backend.forward(teacher, inputs)

            penalty.clear()
            with _autocast(self.cfg.train.precision):
                student_outputs = self.backend.forward(student, inputs)
                loss, components = self.task.compute_loss(
                    student_outputs, teacher_outputs, batch, self.cfg
                )

            if penalty:
                term = penalty.measure()
                if term is not None:
                    loss = loss + penalty.weight * term

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress.step(loss, batch)
        return components

    def train(self):
        set_seed(self.cfg.train.seed)
        data = self.task.datasets(self.tokenizer, self.cfg)
        train_data = data["train"].with_format("torch", columns=self.task.torch_columns())
        opt = self.cfg.train.distill

        batches = BatchLoader(self.cfg, self.dist, train_data, data["collator"], self.log)
        total_steps = len(batches) * opt.epochs
        lr = resolve_lr(opt, batches.global_batch, self.log)
        optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=opt.weight_decay)
        warmup = min(int(total_steps * opt.warmup_ratio), MAX_WARMUP_STEPS)
        self.log(f"schedule: {total_steps:,} steps, {warmup:,} warmup")
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

        best = BestCheckpoint(*self.task.select_metric())

        # Baseline before any update: the student starts from teacher weights, so "did training
        # help?" is only answerable against it. Not a checkpoint candidate.
        self.log(f"init: {self.evaluate()}")

        self.student.train()
        _apply_precision(self.cfg.train.precision)
        student, teacher = self._wrap_for_training()
        progress = ProgressLogger(
            total_steps, self.dist.world_size, self.device, self.log, self.dist.is_main
        )

        with RangePenalty(self.backend, self.student, self.cfg, self.log) as penalty:
            for epoch in range(opt.epochs):
                components = self._run_epoch(
                    batches.loader(epoch),
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    penalty,
                    progress,
                )
                # Rank 0 scores and tracks the checkpoint; the others wait. Replicas are identical
                # after DDP's allreduce, so scoring on every rank would compute the same numbers.
                if self.dist.is_main:
                    metrics = self.evaluate()
                    best.consider(metrics, self.student)
                    comp_str = " ".join(f"{k}={float(v):.3f}" for k, v in components.items())
                    if (mean := penalty.epoch_mean()) is not None:
                        comp_str += f" penalty={mean:.1f}"
                    self.log(f"epoch {epoch}: {metrics} | {comp_str}")
                barrier(self.dist)
                self.student.train()

        # Should stay ~2. A count climbing per epoch means Dynamo will hit cache_size_limit and
        # silently revert to eager mid-run.
        if self.cfg.train.compile:
            graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            self.log(f"dynamo unique_graphs: {graphs}")

        # Only rank 0 tracked `best`, so only rank 0 holds and restores the selected checkpoint.
        # Teardown happens in cli.cmd_distill after that save: destroy_process_group is collective,
        # so tearing down here would race it.
        best.restore(self.student)

    @torch.no_grad()
    def evaluate(self):
        # Always fp32. `allow_tf32` is GLOBAL, so without this the per-epoch metrics inherit it
        # while the `init:` baseline -- measured before it is set -- does not, leaving the two
        # incomparable and checkpoint selection precision-dependent.
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            # Eager on purpose: compiling would add a graph per precision state and train/eval
            # toggle.
            return self.task.evaluate(
                self.backend, self.student, self.tokenizer, self.cfg, teacher=self.teacher
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
