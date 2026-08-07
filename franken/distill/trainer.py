from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup, set_seed

from franken.config import Config
from franken.distill.dist import barrier, init_distributed, per_rank_batch
from franken.distill.progress import ProgressLogger
from franken.models import build_backend
from franken.tasks import build_task

PRECISIONS = ("fp32", "tf32", "bf16")


def _range_penalty(preacts, domain):
    """Squared distance past +/-domain, meaned over the OUT-OF-RANGE elements only
    (averaging over all elements would let the in-range bulk dilute the gradient on
    the rare outliers). Pulls FFN pre-activations into the polynomial op's valid
    domain so the deployed bare poly is FHE-safe. Training-only. None if all in range."""
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
    """Arithmetic for the training loop; evaluation stays fp32. TF32 lowers the precision of the
    *teacher's targets* too, not just the student's math, so check a run against an fp32 reference
    before trusting the speedup. bf16 additionally opens an autocast region (see _autocast)."""
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}; use {' | '.join(PRECISIONS)}")
    if precision in ("tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _autocast(precision: str):
    """Autocast region for the STUDENT forward + loss only. Never the teacher: it makes the
    targets, and a bf16 teacher shifts them 0.0076 recall@10, ~2x the comparison band."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def _maybe_compile(model, cfg: Config):
    """Compiled callable for training. Callers keep the eager module too: its state_dict has no
    `_orig_mod.` prefix, and eval must stay eager (see evaluate)."""
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
        """Rank 0 only: run_experiments.py parses these lines, so N ranks would emit N rows.

        Flushed: stdout is block-buffered when redirected to a log file, so an unflushed
        print leaves `tail -f` empty for the whole run."""
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
        # Single-process keeps the literal shuffle=True path: DistributedSampler draws a
        # different permutation even at world_size 1, and batch composition is worth ~0.004
        # recall -- the width of the comparison band itself.
        sampler = None
        if self.dist.enabled:
            sampler = DistributedSampler(
                train_data,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
                seed=self.cfg.train.seed,
            )
        loader = DataLoader(
            train_data,
            batch_size=per_rank_batch(self.cfg.train.distill.batch_size, self.dist),
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=data["collator"],
        )

        # DistributedSampler pads the index list to divide evenly, so steps-per-epoch happens to
        # match single-process arithmetic rather than being guaranteed to. Assert it: a mismatch
        # silently rescales the LR schedule and invalidates every comparison.
        if self.dist.enabled:
            expected = -(-len(train_data) // self.cfg.train.distill.batch_size)
            if len(loader) != expected:
                raise RuntimeError(
                    f"steps/epoch {len(loader)} != single-process {expected}; the LR schedule "
                    "and recorded results would not be comparable."
                )

        optimizer = AdamW(
            self.student.parameters(),
            lr=self.cfg.train.distill.lr,
            weight_decay=self.cfg.train.distill.weight_decay,
        )
        total_steps = len(loader) * self.cfg.train.distill.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * self.cfg.train.distill.warmup_ratio), total_steps
        )

        # Range penalty (FHE): pull FFN pre-activations into the activation op's
        # valid domain so the deployed bare polynomial never sees out-of-range
        # inputs. Engages only for ops that expose `domain` (e.g. cheb_gelu); each
        # FFN pre-activation is read off via a forward hook. Module paths come from
        # the backend so this is model-agnostic.
        range_penalty = self.cfg.distill.range_penalty
        acts = self.backend.activation_ops(self.student)
        first_act = acts[0] if acts else None
        domain = getattr(first_act, "domain", None) if (range_penalty > 0 and first_act) else None

        # Softmax range penalty (FHE): the op computes its own term, because the constrained
        # quantity (cgf's per-row log-sum) is internal to it -- no module boundary to hook.
        softmax_range_penalty = self.cfg.distill.softmax_range_penalty
        ranged_softmax_ops = (
            [op for op in self.backend.softmax_ops(self.student) if hasattr(op, "range_loss")]
            if softmax_range_penalty > 0
            else []
        )
        preacts, hooks = [], []
        if domain is not None:

            def _capture(module, _inp, out):
                if module.training:
                    preacts.append(out)

            # Hook only the layers being penalized. Constraining a layer costs accuracy, so
            # the default (all layers) is usually the wrong choice — see range_penalty_layers.
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

        # Baseline before any update: the student starts from teacher weights, so "did
        # training help?" is only answerable against it. Not a checkpoint candidate.
        self.log(f"init: {self.evaluate()}")

        self.student.train()

        # Training-loop arithmetic (evaluation stays fp32 whatever this is).
        _apply_precision(self.cfg.train.precision)

        # Training only; self.student/self.teacher stay eager for evaluate() and the save.
        # DDP wraps first: compiling the wrapper lets Dynamo's DDPOptimizer split the graph so
        # gradient allreduce overlaps backward. The other order emits one fused graph whose
        # grads all become ready at once, exposing the full ~2.4 GB of comms.
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

        for epoch in range(self.cfg.train.distill.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            # Mean range penalty over the epoch. Logged because it is the only visible evidence that
            # the penalty is doing anything: it should fall as pre-activations are squashed into the
            # op's domain. Verify the end state with scripts/qwen3/act_range.py on the checkpoint.
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
                    pen_act = _range_penalty(preacts, domain)
                    if pen_act is not None:
                        loss = loss + range_penalty * pen_act
                        components["pen_act"] = pen_act.detach()
                # None when an op has no band configured, so the weight alone cannot enable it.
                terms = [op.range_loss for op in ranged_softmax_ops if op.range_loss is not None]
                if terms:
                    pen_sftmx = torch.stack(terms).mean()
                    loss = loss + softmax_range_penalty * pen_sftmx
                    components["pen_sftmx"] = pen_sftmx.detach()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress.step(loss, batch)

                for op in ranged_softmax_ops:
                    op.range_loss = None  # else the step's graph stays alive through evaluate()

            # Rank 0 scores and tracks the checkpoint; the others wait. Replicas are identical
            # after DDP's allreduce, so scoring on every rank would compute the same numbers.
            if self.dist.is_main:
                metrics = self.evaluate()
                # Select on the task's headline metric (max F1 for MRPC; min distance for
                # embedding self-distill). The student is deterministic, so the argmax/argmin
                # is stable run-to-run.
                value = metrics[metric_name]
                improved = value > best if higher_is_better else value < best
                if improved:
                    best = value
                    # Clone off-device: state_dict() returns live references that the
                    # next optimizer.step() would mutate in place.
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

        # Only rank 0 tracked `best`, so only rank 0's student is the selected checkpoint --
        # and only rank 0 saves it. Teardown happens in cli.cmd_distill, after that save:
        # destroy_process_group is itself collective, so tearing down here would race the save.
        if best_state is not None:
            self.student.load_state_dict(best_state)

    @torch.no_grad()
    def evaluate(self):
        # Always fp32, whatever the training precision. `allow_tf32` is a *global* flag, so
        # without this the per-epoch metrics would inherit it while the `init:` baseline —
        # measured before it is set — would not, leaving them incomparable and making
        # checkpoint selection precision-dependent.
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            # Eager on purpose: a compiled callable would add a graph per precision state and
            # per train/eval toggle.
            return self.task.evaluate(
                self.backend, self.student, self.tokenizer, self.cfg, teacher=self.teacher
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
