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

from franken.config import Config
from franken.distill.batching import plan_batches, shard
from franken.distill.dist import barrier, init_distributed, per_rank_batch
from franken.distill.progress import ProgressLogger
from franken.models import build_backend
from franken.tasks import build_task

PRECISIONS = ("fp32", "tf32", "bf16")


# Reference for sqrt-batch LR scaling (`lr: null`): bs32/lr2e-5, fidelity-verified at depth 24
# (recall@10 0.9070, identical to fp32/bs8/lr1e-5). sqrt not linear because Adam-family.
BASE_LR, BASE_BATCH = 2e-5, 32
# Cap on `warmup_ratio`: warmup covers early-Adam instability, which is a fixed number of steps, not
# a share of the run -- uncapped, changing `epochs` moves it and the two runs stop being comparable.
MAX_WARMUP_STEPS = 2000


def _range_penalty(preacts, domain):
    """Squared distance past +/-domain, meaned over the OUT-OF-RANGE elements only -- averaging
    over all of them would let the in-range bulk dilute the gradient on the rare outliers.
    Training-only; None if everything is in range."""
    terms = []
    for x in preacts:
        # fp32 always: a tail statistic, and bf16's coarse grid up there shifts it +7.6%.
        x = x.float()
        over, under = F.relu(x - domain), F.relu(-domain - x)
        outside = (over > 0) | (under > 0)
        if outside.any():
            terms.append(((over**2 + under**2)[outside]).mean())
    return torch.stack(terms).mean() if terms else None


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

    def train(self):
        set_seed(self.cfg.train.seed)
        data = self.task.datasets(self.tokenizer, self.cfg)
        train_data = data["train"].with_format("torch", columns=self.task.torch_columns())
        opt = self.cfg.train.distill

        sampler = batch_plan = None
        if opt.token_budget:
            lengths = pc.list_value_length(train_data.data.column("input_ids")).to_numpy(
                zero_copy_only=False
            )
            batch_plan = shard(
                plan_batches(lengths, opt.token_budget, opt.max_seqs, self.cfg.train.seed),
                self.dist.rank,
                self.dist.world_size,
            )
            self.log(
                f"token-budgeted batching: {len(batch_plan):,} steps/epoch/rank, "
                f"{opt.token_budget:,} tokens x {self.dist.world_size} ranks per step"
            )
        elif self.dist.enabled:
            # Single-process keeps the literal shuffle=True path: DistributedSampler draws a
            # different permutation even at world_size 1, and batch composition is worth ~0.004
            # recall -- the width of the comparison band itself.
            sampler = DistributedSampler(
                train_data,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
                seed=self.cfg.train.seed,
            )

        def build_loader(epoch: int) -> DataLoader:
            if batch_plan is None:
                return DataLoader(
                    train_data,
                    batch_size=per_rank_batch(opt.batch_size, self.dist),
                    shuffle=sampler is None,
                    sampler=sampler,
                    collate_fn=data["collator"],
                )
            # One plan, reordered per epoch. Re-planning would vary steps/epoch with the shuffle
            # and drift the LR schedule; order alone is enough to decorrelate the epochs.
            order = list(batch_plan)
            random.Random(self.cfg.train.seed + epoch).shuffle(order)
            return DataLoader(train_data, batch_sampler=order, collate_fn=data["collator"])

        loader = build_loader(0)

        # DistributedSampler pads to divide evenly, so matching single-process steps/epoch is a
        # happy accident, not a guarantee -- and a mismatch silently rescales the LR schedule.
        # (`shard` gives the token-budgeted path the same guarantee by construction.)
        if self.dist.enabled and batch_plan is None:
            expected = -(-len(train_data) // opt.batch_size)
            if len(loader) != expected:
                raise RuntimeError(
                    f"steps/epoch {len(loader)} != single-process {expected}; the LR schedule "
                    "and recorded results would not be comparable."
                )

        # `lr: null` derives the rate by sqrt-batch scaling. Under token budgeting the batch floats
        # to fill the budget and is only known once the plan exists, so a hardcoded `lr` goes stale
        # when the corpus or rank count moves -- the last ladder ran 5.6% hot exactly that way.
        total_steps = len(loader) * opt.epochs
        lr = opt.lr
        if lr is None:
            if batch_plan is None:
                batch = float(opt.batch_size)  # already the GLOBAL batch
            else:
                seqs = sum(len(b) for b in batch_plan) / max(len(batch_plan), 1)
                batch = seqs * self.dist.world_size
            lr = BASE_LR * math.sqrt(batch / BASE_BATCH)
            self.log(
                f"lr {lr:.4e} = {BASE_LR:g} * sqrt({batch:.0f} / {BASE_BATCH})"
                f" [sqrt-batch scaling from the tuned reference]"
            )
        optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=opt.weight_decay)
        warmup = min(int(total_steps * opt.warmup_ratio), MAX_WARMUP_STEPS)
        self.log(f"schedule: {total_steps:,} steps, {warmup:,} warmup")
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

        # Range penalty (FHE): pull FFN pre-activations, read via forward hooks on backend-supplied
        # modules, into the activation op's domain. Engages only for ops exposing `domain`.
        penalty_weight = self.cfg.distill.range_penalty
        acts = self.backend.activation_ops(self.student)
        first_act = acts[0] if acts else None
        domain = getattr(first_act, "domain", None) if (penalty_weight > 0 and first_act) else None
        preacts, hooks = [], []
        if domain is not None:

            def _capture(module, _inp, out):
                if module.training:
                    preacts.append(out)

            # Hook only the penalized layers: constraining one costs accuracy, so the default
            # (all of them) is usually wrong -- see range_penalty_layers.
            mods = self.backend.ffn_preact_modules(self.student)
            which = self.cfg.distill.range_penalty_layers
            if which is None:
                targets = mods
            else:
                bad = [i for i in which if not 0 <= i < len(mods)]
                if bad:
                    raise ValueError(
                        f"range_penalty_layers {bad} out of range for a {len(mods)}-layer "
                        f"student (valid 0..{len(mods) - 1}; STUDENT indices, not teacher's)"
                    )
                targets = [mods[i] for i in which]
                self.log(
                    f"range penalty on student layers {sorted(which)} "
                    f"of {len(mods)}, domain {domain}"
                )
            hooks = [m.register_forward_hook(_capture) for m in targets]

        metric_name, higher_is_better = self.task.select_metric()
        best = float("-inf") if higher_is_better else float("inf")
        best_state = None

        # Baseline before any update: the student starts from teacher weights, so "did training
        # help?" is only answerable against it. Not a checkpoint candidate.
        self.log(f"init: {self.evaluate()}")

        self.student.train()
        _apply_precision(self.cfg.train.precision)

        # Training only; self.student/self.teacher stay eager for evaluate() and the save.
        # DDP wraps FIRST: compiling the wrapper lets Dynamo's DDPOptimizer split the graph so
        # allreduce overlaps backward. The other order fuses one graph whose grads all become
        # ready at once, exposing the full ~2.4 GB of comms.
        student = self.student
        if self.dist.enabled:
            student = DistributedDataParallel(
                student, device_ids=[self.dist.local_rank], gradient_as_bucket_view=True
            )
        student = _maybe_compile(student, self.cfg)
        teacher = _maybe_compile(self.teacher, self.cfg)

        progress = ProgressLogger(
            total_steps, self.dist.world_size, self.device, self.log, self.dist.is_main
        )

        for epoch in range(opt.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if batch_plan is not None and epoch:
                loader = build_loader(epoch)
            # Mean penalty over the epoch: the only visible evidence it is doing anything (it should
            # fall). Verify the end state with scripts/qwen3/act_range.py on the checkpoint.
            penalties = []
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                inputs = self.task.model_inputs(batch)

                # Outside the autocast region: see _autocast.
                with torch.no_grad():
                    teacher_outputs = self.backend.forward(teacher, inputs)

                preacts.clear()
                with _autocast(self.cfg.train.precision):
                    student_outputs = self.backend.forward(student, inputs)
                    total, components = self.task.compute_loss(
                        student_outputs, teacher_outputs, batch, self.cfg
                    )

                loss = total
                if domain is not None:
                    penalty = _range_penalty(preacts, domain)
                    if penalty is not None:
                        loss = total + penalty_weight * penalty
                        penalties.append(penalty.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress.step(loss, batch)

            # Rank 0 scores and tracks the checkpoint; the others wait. Replicas are identical
            # after DDP's allreduce, so scoring on every rank would compute the same numbers.
            if self.dist.is_main:
                metrics = self.evaluate()
                value = metrics[metric_name]
                improved = value > best if higher_is_better else value < best
                if improved:
                    best = value
                    # Clone off-device: state_dict() hands back live references that the next
                    # optimizer.step() would mutate in place.
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.student.state_dict().items()
                    }
                comp_str = " ".join(f"{k}={float(v):.3f}" for k, v in components.items())
                if penalties:
                    comp_str += f" penalty={torch.stack(penalties).mean().item():.1f}"
                self.log(f"epoch {epoch}: {metrics} | {comp_str}")
            barrier(self.dist)
            self.student.train()

        for h in hooks:
            h.remove()

        # Should stay ~2. A count climbing per epoch means Dynamo will hit cache_size_limit and
        # silently revert to eager mid-run.
        if self.cfg.train.compile:
            graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            self.log(f"dynamo unique_graphs: {graphs}")

        # Only rank 0 tracked `best`, so only rank 0 holds and saves the selected checkpoint.
        # Teardown happens in cli.cmd_distill after that save: destroy_process_group is collective,
        # so tearing down here would race it.
        if best_state is not None:
            self.student.load_state_dict(best_state)

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
