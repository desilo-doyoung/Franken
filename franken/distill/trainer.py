import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from franken.config import Config
from franken.data.mrpc import compute_metrics, load_mrpc
from franken.distill.layer_map import resolve_layer_map
from franken.distill.loss import DistillationLoss
from franken.model.bert import BertForClassification
from franken.model.loader import init_student_from_teacher
from franken.teacher import load_teacher


class Distiller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        self.loss_fn = DistillationLoss(cfg.distill)
        self.teacher = None
        self.student = None
        self.tokenizer = None

    def setup(self):
        self.teacher = load_teacher(self.cfg).to(self.device)
        self.student = BertForClassification(self.cfg.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.train.teacher_model)

        # strided weight init
        teacher_state_dict = self.teacher.state_dict()
        n_teacher = self.teacher.config.num_hidden_layers
        layer_map = resolve_layer_map(
            n_teacher, self.cfg.model.num_hidden_layers, self.cfg.distill.hidden_layer_map
        )

        init_student_from_teacher(self.student, teacher_state_dict, layer_map)
        self.student.to(self.device)

    def train(self):
        set_seed(self.cfg.train.seed)
        data = load_mrpc(self.tokenizer, self.cfg.train.max_seq_len)
        train_data = data["train"].with_format(
            "torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
        )
        loader = DataLoader(
            train_data,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=data["collator"],
        )

        optimizer = AdamW(
            self.student.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        total_steps = len(loader) * self.cfg.train.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * self.cfg.train.warmup_ratio), total_steps
        )

        self.student.train()

        for epoch in range(self.cfg.train.epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"]

                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        output_hidden_states=True,
                    )

                student_outputs = self.student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                )

                total, ce, kl, hidden = self.loss_fn(
                    student_outputs["logits"],
                    teacher_outputs["logits"],
                    labels,
                    student_outputs["hidden_states"],
                    teacher_outputs["hidden_states"],
                    batch["attention_mask"],
                )

                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            metrics = self.evaluate()
            print(f"epoch {epoch}: {metrics} | ce={ce:.3f} kl={kl:.3f} hidden={hidden:.3f}")
            self.student.train()

    @torch.no_grad()
    def evaluate(self):
        data = load_mrpc(self.tokenizer, self.cfg.train.max_seq_len)
        validation_data = data["validation"].with_format(
            "torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
        )
        loader = DataLoader(
            validation_data, batch_size=self.cfg.train.batch_size, collate_fn=data["collator"]
        )

        self.student.eval()
        preds = []
        labels = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            preds.append(outputs["logits"].argmax(dim=-1).cpu())
            labels.append(batch["labels"].cpu())

        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        metrics = compute_metrics(preds, labels)

        return metrics
