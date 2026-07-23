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

            hooks = [
                m.register_forward_hook(_capture)
                for m in self.backend.ffn_preact_modules(self.student)
            ]

        self.student.train()

        metric_name, higher_is_better = self.task.select_metric()
        best = float("-inf") if higher_is_better else float("inf")
        best_state = None

        for epoch in range(self.cfg.train.distill.epochs):
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
            print(f"epoch {epoch}: {metrics} | {comp_str}")
            self.student.train()

        for h in hooks:
            h.remove()

        if best_state is not None:
            self.student.load_state_dict(best_state)

    @torch.no_grad()
    def evaluate(self):
        return self.task.evaluate(self.backend, self.student, self.tokenizer, self.cfg)
