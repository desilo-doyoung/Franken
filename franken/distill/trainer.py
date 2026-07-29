import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed

from franken.config import Config
from franken.models import build_backend
from franken.tasks import build_task


def _range_penalty(preacts, domain):
    """Squared distance past +/-domain, meaned over the OUT-OF-RANGE elements only
    (averaging over all elements would let the in-range bulk dilute the gradient on
    the rare outliers). Pulls FFN pre-activations into the polynomial op's valid
    domain so the deployed bare poly is FHE-safe. Training-only. None if all in range."""
    terms = []
    for x in preacts:
        over, under = F.relu(x - domain), F.relu(-domain - x)
        outside = (over > 0) | (under > 0)
        if outside.any():
            terms.append(((over**2 + under**2)[outside]).mean())
    return torch.stack(terms).mean() if terms else None


def _apply_precision(precision: str) -> None:
    """Arithmetic for the training loop; evaluation stays fp32. TF32 lowers the precision of the
    *teacher's targets* too, not just the student's math, so check a run against an fp32 reference
    before trusting the speedup."""
    if precision not in ("fp32", "tf32"):
        raise ValueError(f"Unknown precision {precision!r}; use fp32 | tf32")
    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class Distiller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        self.backend = build_backend(cfg.model.backend)
        self.task = build_task(cfg.train.task)
        self.teacher = None
        self.student = None
        self.tokenizer = None

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
        loader = DataLoader(
            train_data,
            batch_size=self.cfg.train.distill.batch_size,
            shuffle=True,
            collate_fn=data["collator"],
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
        penalty_weight = self.cfg.distill.range_penalty
        acts = self.backend.activation_ops(self.student)
        first_act = acts[0] if acts else None
        domain = getattr(first_act, "domain", None) if (penalty_weight > 0 and first_act) else None
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
                print(
                    f"range penalty on student layers {sorted(which)} "
                    f"of {len(mods)}, domain {domain}"
                )
            hooks = [m.register_forward_hook(_capture) for m in targets]

        metric_name, higher_is_better = self.task.select_metric()
        best = float("-inf") if higher_is_better else float("inf")
        best_state = None

        # Baseline before any update: the student starts from teacher weights, so "did
        # training help?" is only answerable against it. Not a checkpoint candidate.
        print(f"init: {self.evaluate()}")

        self.student.train()

        # Training-loop arithmetic (evaluation stays fp32 whatever this is).
        _apply_precision(self.cfg.train.precision)

        for epoch in range(self.cfg.train.distill.epochs):
            # Mean range penalty over the epoch. Logged because it is the only visible evidence that
            # the penalty is doing anything: it should fall as pre-activations are squashed into the
            # op's domain. Verify the end state with scripts/qwen3/act_range.py on the checkpoint.
            penalties = []
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                inputs = self.task.model_inputs(batch)

                with torch.no_grad():
                    teacher_outputs = self.backend.forward(self.teacher, inputs)

                preacts.clear()
                student_outputs = self.backend.forward(self.student, inputs)

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
            print(f"epoch {epoch}: {metrics} | {comp_str}")
            self.student.train()

        for h in hooks:
            h.remove()

        if best_state is not None:
            self.student.load_state_dict(best_state)

    @torch.no_grad()
    def evaluate(self):
        # Always fp32, whatever the training precision. `allow_tf32` is a *global* flag, so
        # without this the per-epoch metrics would inherit it while the `init:` baseline —
        # measured before it is set — would not, leaving them incomparable and making
        # checkpoint selection precision-dependent.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self.task.evaluate(
                self.backend, self.student, self.tokenizer, self.cfg, teacher=self.teacher
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
