"""Teacher: HF BertForSequenceClassification fine-tuned on MRPC (exact ops).

The teacher uses stock HF modules (exact softmax/GELU, full 12 layers). During
distillation it is frozen and run with ``output_hidden_states=True`` so the loss
can access its per-layer hidden states.
"""

from typing import Any

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from franken.config import Config
from franken.data.mrpc import compute_metrics, load_mrpc


def hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics(preds, labels)


def train_teacher(cfg: Config) -> str:
    tok = AutoTokenizer.from_pretrained(cfg.train.teacher_model)
    data = load_mrpc(tok, max_seq_len=cfg.train.max_seq_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.train.teacher_model, num_labels=2
    )

    args = TrainingArguments(
        output_dir=f"{cfg.train.output_dir}/teacher",
        learning_rate=cfg.train.lr,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        num_train_epochs=cfg.train.epochs,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=cfg.train.seed,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=data["collator"],
        compute_metrics=hf_metrics,
    )
    trainer.train()

    path = f"{cfg.train.output_dir}/teacher"
    trainer.save_model(path)
    tok.save_pretrained(path)

    return path


def load_teacher(cfg: Config) -> Any:
    """Load a frozen teacher for distillation.

    Loads from ``cfg.train.teacher_ckpt`` (or ``teacher_model``), sets eval mode,
    disables grads, and enables ``output_hidden_states=True``.
    """
    ckpt = cfg.train.teacher_ckpt or cfg.train.teacher_model
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, output_hidden_states=True)
    model.eval()
    model.requires_grad_(False)
    return model
