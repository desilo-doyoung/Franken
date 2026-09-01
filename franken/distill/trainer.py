import math
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup, set_seed

from franken.config import PRECISIONS, Config
from franken.distill.batching import row_plan, shard
from franken.distill.dist import (
    barrier,
    init_distributed,
    max_tokens_per_rank,
    per_rank_batch,
)
from franken.distill.progress import ProgressLogger
from franken.models import build_backend
from franken.tasks import build_task

# Tuned reference for sqrt-batch LR scaling; sqrt not linear because Adam-family.
BASE_LR, BASE_BATCH = 2e-5, 32
# Warmup covers early-Adam instability -- a fixed number of steps, not a share of the run.
MAX_WARMUP_STEPS = 2000


def _apply_precision(precision: str) -> None:
    # Training only; eval stays fp32. TF32 lowers the teacher's TARGETS too, not just the
    # student's math.
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}; use {' | '.join(PRECISIONS)}")
    if precision in ("tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _autocast(precision: str):
    # STUDENT forward + loss only: a bf16 teacher shifts the targets by ~2x the comparison band.
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def _maybe_compile(model, cfg: Config):
    # Callers keep the eager module: its state_dict has no `_orig_mod.` prefix.
    return torch.compile(model) if cfg.train.compile else model


def resolve_lr(opt, global_batch: float, log, packed: bool = False) -> float:
    """`lr: null` derives the rate by sqrt-batch scaling. The token-budgeted batch is only known
    once the plan exists, so a hardcoded `lr` goes stale when the corpus or rank count moves."""
    if opt.lr is not None:
        return opt.lr
    if packed:
        raise ValueError(
            "train.pack with distill.lr: null. sqrt-batch scaling reads SEQUENCES, but a packed "
            "block is an arbitrary container: halving max_seq_len doubles the count at identical "
            f"tokens/step and would move the rate by sqrt(2). BASE_BATCH ({BASE_BATCH}) is a "
            "sequence count calibrated on unpacked embed runs besides. Set distill.lr explicitly."
        )
    lr = BASE_LR * math.sqrt(global_batch / BASE_BATCH)
    log(
        f"lr {lr:.4e} = {BASE_LR:g} * sqrt({global_batch:.0f} / {BASE_BATCH})"
        f" [sqrt-batch scaling from the tuned reference]"
    )
    return lr


def _no_sync(student):
    """DDP-only; a bare module has no no_sync."""
    return student.no_sync() if hasattr(student, "no_sync") else nullcontext()


def _split_step(tokens_per_step: int, world_size: int) -> tuple[int, int]:
    """GLOBAL tokens/step -> (per-rank micro-batch tokens, accumulation steps). The machine ceiling
    only decides how the step is chopped up, never how big it is."""
    micro = min(tokens_per_step // world_size, max_tokens_per_rank())
    if micro < 1:
        raise ValueError(
            f"tokens_per_step {tokens_per_step:,} is below world_size {world_size}: "
            "fewer than one token per rank."
        )
    if tokens_per_step % (micro * world_size):
        raise ValueError(
            f"tokens_per_step {tokens_per_step:,} is not divisible by "
            f"{micro:,} tokens x {world_size} ranks; the global batch must be exact or runs on "
            f"different machines stop being comparable. Try a multiple of {micro * world_size:,}."
        )
    return micro, tokens_per_step // (micro * world_size)


class BatchLoader:
    """Token-budgeted plan, DistributedSampler, or plain shuffle. Owns the per-epoch loader:
    steps/epoch must not move, or the LR schedule stops matching the recorded runs."""

    def __init__(self, cfg: Config, dist, dataset, collator, log):
        self.opt = cfg.train.distill
        self.dist, self.dataset, self.collator = dist, dataset, collator
        self.seed = cfg.train.seed
        self._block = cfg.train.max_seq_len if cfg.train.pack else None
        self.sampler = self.plan = None
        self.accum_steps = 1

        if self.opt.tokens_per_step:
            self.micro_tokens, self.accum_steps = _split_step(
                self.opt.tokens_per_step, dist.world_size
            )
            self.plan = shard(
                row_plan(dataset, self.micro_tokens, self.seed, self._block),
                dist.rank,
                dist.world_size,
            )
            log(
                f"token-budgeted batching: {self.opt.tokens_per_step:,} tokens/step "
                f"= {self.micro_tokens:,} x {dist.world_size} ranks x {self.accum_steps} accum; "
                f"{len(self.plan) // self.accum_steps:,} optimizer steps/epoch"
            )
        elif dist.enabled:
            # Single-process keeps the literal shuffle=True path: DistributedSampler permutes
            # differently even at world_size 1, and batch composition is worth ~0.004 recall.
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=dist.world_size,
                rank=dist.rank,
                shuffle=True,
                seed=self.seed,
            )

        self._loader = self._build(0)
        # DistributedSampler pads to divide evenly, so a steps/epoch match is an accident, not a
        # guarantee -- and a mismatch silently rescales the LR schedule.
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
        # Reordered, not re-planned: re-planning would vary steps/epoch and drift the schedule.
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
        """Micro-batches per epoch. `optimizer_steps` is what the LR schedule counts."""
        return len(self._loader)

    @property
    def optimizer_steps(self) -> int:
        return len(self._loader) // self.accum_steps

    @property
    def global_batch(self) -> float:
        """Sequences per step across all ranks -- what sqrt-batch LR scaling reads."""
        if self.plan is None:
            return float(self.opt.batch_size)  # already the GLOBAL batch
        seqs = sum(len(b) for b in self.plan) / max(len(self.plan), 1)
        return seqs * self.dist.world_size * self.accum_steps


class RangePenalty:
    """Pulls one site's pre-activations into the domain its FHE consumer's polynomial can cover,
    via forward hooks that live only for the `with` body."""

    def __init__(self, site: str, modules, domain: float, weight: float):
        self.site = site
        self.modules = modules
        self.domain = float(domain)
        self.weight = weight
        self._preacts, self._hooks, self._epoch = [], [], []

    def __enter__(self):
        self._hooks = [m.register_forward_hook(self._capture) for m in self.modules]
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
        """Squared distance past +/-domain, meaned over the OUT-OF-RANGE elements only: the
        in-range bulk would otherwise dilute the gradient on the rare outliers. Unweighted, so
        the epoch line stays comparable to `act_range.py`; `RangePenalties` applies the weight."""
        terms = []
        for x in self._preacts:
            # fp32 always: bf16's coarse grid shifts this tail statistic +7.6%.
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
        """Mean over the epoch, then reset. It should fall; verify the end state with
        `act_range.py`."""
        if not self._epoch:
            return None
        mean = torch.stack(self._epoch).mean().item()
        self._epoch.clear()
        return mean


class RangePenalties:
    """Every constrained site, driven as one: enter once, clear once a step, one loss term.
    Falsy -- and every method a no-op -- when nothing is constrained."""

    def __init__(self, penalties: list[RangePenalty]):
        self.penalties = penalties

    def __bool__(self) -> bool:
        return bool(self.penalties)

    def __enter__(self):
        for penalty in self.penalties:
            penalty.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        for penalty in self.penalties:
            penalty.__exit__(*exc)
        return False

    def clear(self) -> None:
        for penalty in self.penalties:
            penalty.clear()

    def loss_term(self):
        """The weighted sum over sites, or None when everything sat inside its domain."""
        terms = [
            penalty.weight * measured
            for penalty in self.penalties
            if (measured := penalty.measure()) is not None
        ]
        return sum(terms) if terms else None

    def epoch_summary(self) -> str:
        """`ffn=... pooler=...` for the epoch line, empty when there is nothing to report."""
        parts = [
            f"{penalty.site}={mean:.1f}"
            for penalty in self.penalties
            if (mean := penalty.epoch_mean()) is not None
        ]
        return " ".join(parts)


# TODO: Refactor
def build_penalties(backend, student, cfg: Config, log) -> RangePenalties:
    """One penalty per constrained site. The FFN's domain comes off the activation op; the pooler's
    is explicit, because its wall belongs to the consumer's tanh fit and not to any op the student
    holds. Empty when nothing is constrained."""
    penalties = []
    acts = backend.activation_ops(student)
    first = acts[0] if acts else None
    domain = getattr(first, "domain", None) if first else None
    if cfg.distill.range_penalty > 0 and domain is not None:
        # Constraining a layer costs accuracy, so hooking all of them is usually wrong.
        mods = backend.ffn_preact_modules(student)
        which = cfg.distill.range_penalty_layers
        targets = mods if which is None else [mods[i] for i in which]
        penalties.append(RangePenalty("ffn", targets, domain, cfg.distill.range_penalty))
        layers = "all" if which is None else f"{sorted(which)} of {len(mods)}"
        log(f"ffn penalty {cfg.distill.range_penalty} on student layers {layers}, domain {domain}")
    if cfg.distill.pooler_penalty > 0 and cfg.distill.pooler_domain is not None:
        pooler = backend.pooler_preact_modules(student)
        if pooler:
            penalties.append(
                RangePenalty(
                    "pooler", pooler, cfg.distill.pooler_domain, cfg.distill.pooler_penalty
                )
            )
            log(f"pooler penalty {cfg.distill.pooler_penalty}, domain {cfg.distill.pooler_domain}")
    return RangePenalties(penalties)


class BestCheckpoint:
    """Best-scoring student weights seen so far, by the task's own selection metric. Scored once
    per epoch, so at `distill.epochs: 1` there is one candidate and this selects nothing."""

    def __init__(self, metric_name: str, higher_is_better: bool):
        self.metric_name, self.higher_is_better = metric_name, higher_is_better
        self.best = float("-inf") if higher_is_better else float("inf")
        self.state = None

    def consider(self, metrics: dict, student) -> bool:
        value = metrics[self.metric_name]
        if not (value > self.best if self.higher_is_better else value < self.best):
            return False
        self.best = value
        # Clone off-device: state_dict() hands back references the next step() would mutate.
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
        self._cuda = self.device.type == "cuda"
        self.backend = build_backend(cfg.model.backend)
        self.task = build_task(cfg.train.task)
        self.teacher = None
        self.student = None
        self.tokenizer = None

    def log(self, *args):
        # Flushed: stdout is block-buffered into a log file. run_experiments parses these lines.
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
        # DDP wraps FIRST so Dynamo's DDPOptimizer can split the graph and overlap allreduce with
        # backward; the other order exposes the full ~2.4 GB of comms. self.student stays eager.
        student = self.student
        if self.dist.enabled:
            student = DistributedDataParallel(
                student,
                device_ids=[self.dist.local_rank],
                gradient_as_bucket_view=True,
                # The default re-broadcasts every buffer from rank 0 each forward, for buffers
                # that mutate in forward. RoPE inv_freq does not -- it is already identical.
                broadcast_buffers=False,
            )
        return _maybe_compile(student, self.cfg), _maybe_compile(self.teacher, self.cfg)

    def _run_epoch(
        self, batches, loader, student, teacher, optimizer, scheduler, penalties, progress
    ):
        """Returns the last batch's loss components, for the epoch line."""
        components = {}
        # A trailing group of <accum micro-batches never reaches a step; its gradients are dropped
        # here, and `optimizer_steps` already floors to match.
        accum = batches.accum_steps
        optimizer.zero_grad()
        for micro, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            inputs = self.task.model_inputs(batch)

            # Outside the autocast region: see _autocast.
            with torch.no_grad():
                teacher_outputs = self.backend.forward(teacher, inputs)

            penalties.clear()
            with _autocast(self.cfg.train.precision):
                student_outputs = self.backend.forward(student, inputs)
                loss, components = self.task.compute_loss(
                    student_outputs, teacher_outputs, batch, self.cfg
                )

            if (term := penalties.loss_term()) is not None:
                loss = loss + term

            # no_sync or DDP allreduces every micro-batch and accumulation costs more than it
            # saves. `loss / accum` weights micro-batches equally, not per sequence: zero-mean
            # given the shuffled plan, and moot at accum 1.
            boundary = (micro + 1) % accum == 0
            with nullcontext() if boundary else _no_sync(student):
                (loss / accum).backward()
            if boundary:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            progress.step(loss, batch, advance=boundary)
        return components

    def train(self):
        set_seed(self.cfg.train.seed)
        data = self.task.datasets(self.tokenizer, self.cfg)
        train_data = data["train"].with_format("torch", columns=self.task.torch_columns())
        opt = self.cfg.train.distill

        batches = BatchLoader(self.cfg, self.dist, train_data, data["collator"], self.log)
        total_steps = batches.optimizer_steps * opt.epochs  # LR schedule counts OPTIMIZER steps
        lr = resolve_lr(opt, batches.global_batch, self.log, self.cfg.train.pack)
        optimizer = AdamW(
            self.student.parameters(), lr=lr, weight_decay=opt.weight_decay, fused=True
        )
        warmup = min(int(total_steps * opt.warmup_ratio), MAX_WARMUP_STEPS)
        self.log(f"schedule: {total_steps:,} steps, {warmup:,} warmup")
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

        best = BestCheckpoint(*self.task.select_metric())

        # Baseline before any update -- the student starts from teacher weights. Not a candidate.
        # Rank 0 only: `log` filtered the output, but every rank still paid for the eval.
        if self.dist.is_main:
            self.log(f"init: {self.evaluate()}")
        barrier(self.dist)

        self.student.train()
        # Before wrapping, so DDP and compile see an already-configured module.
        self.student.grad_checkpoint = self.cfg.train.grad_checkpoint
        _apply_precision(self.cfg.train.precision)
        student, teacher = self._wrap_for_training()
        progress = ProgressLogger(
            total_steps, self.dist.world_size, self.device, self.log, self.dist.is_main
        )
        # Reset after the baseline eval, so the reported peak is the training step's alone --
        # that is the number `tokens_per_step` has to be sized against.
        if self._cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        with build_penalties(self.backend, self.student, self.cfg, self.log) as penalties:
            for epoch in range(opt.epochs):
                components = self._run_epoch(
                    batches,
                    batches.loader(epoch),
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    penalties,
                    progress,
                )
                # Rank 0 scores; replicas are identical after allreduce, so the others just wait.
                if self.dist.is_main:
                    metrics = self.evaluate()
                    best.consider(metrics, self.student)
                    comp_str = " ".join(f"{k}={float(v):.3f}" for k, v in components.items())
                    if summary := penalties.epoch_summary():
                        comp_str += f" {summary}"
                    self.log(f"epoch {epoch}: {metrics} | {comp_str}")
                barrier(self.dist)
                self.student.train()

        if self._cuda:
            self.log(
                f"peak GPU: {torch.cuda.max_memory_allocated(self.device) / 2**30:.2f} GiB "
                f"allocated, {torch.cuda.max_memory_reserved(self.device) / 2**30:.2f} reserved"
            )

        # Should stay ~2; climbing per epoch means Dynamo will silently revert to eager mid-run.
        if self.cfg.train.compile:
            graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            self.log(f"dynamo unique_graphs: {graphs}")

        # Only rank 0 tracked `best`. Teardown is in cli.cmd_distill, after the save.
        best.restore(self.student)

    @torch.no_grad()
    def evaluate(self):
        # Always fp32: `allow_tf32` is GLOBAL, so otherwise the per-epoch metrics inherit it and
        # the `init:` baseline does not, making checkpoint selection precision-dependent.
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            # Eager: compiling would add a graph per precision state and train/eval toggle.
            return self.task.evaluate(
                self.backend, self.student, self.tokenizer, self.cfg, teacher=self.teacher
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
