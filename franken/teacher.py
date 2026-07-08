"""Teacher: HF BertForSequenceClassification fine-tuned on MRPC (exact ops).

The teacher uses stock HF modules (exact softmax/GELU, full 12 layers). During
distillation it is frozen and run with ``output_hidden_states=True`` so the loss
can access its per-layer hidden states.
"""

from typing import Any

from franken.config import Config


def train_teacher(cfg: Config) -> str:
    """Fine-tune the HF teacher on MRPC and save a checkpoint.

    Returns the checkpoint path.
    """
    # TODO:
    #   AutoModelForSequenceClassification.from_pretrained(cfg.train.teacher_model,
    #       num_labels=2); fine-tune on MRPC; save to cfg.train.output_dir.
    raise NotImplementedError("train_teacher is a stub.")


def load_teacher(cfg: Config) -> Any:
    """Load a frozen teacher for distillation.

    Loads from ``cfg.train.teacher_ckpt`` (or ``teacher_model``), sets eval mode,
    disables grads, and enables ``output_hidden_states=True``.
    """
    # TODO:
    #   model = AutoModelForSequenceClassification.from_pretrained(ckpt,
    #       output_hidden_states=True)
    #   model.eval(); model.requires_grad_(False); return model
    raise NotImplementedError("load_teacher is a stub.")
