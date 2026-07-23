import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file as load_safetensors
from torch import nn
from transformers import BertConfig, BertForSequenceClassification


def load_bert_config(model_dir: Path) -> BertConfig:
    """Build an HF BertConfig from the model's config.json.

    The distilled model's config.json carries extra franken-specific fields
    (softmax/activation/lowrank), so only the standard BERT fields are read.
    """
    raw = json.loads((model_dir / "config.json").read_text())
    return BertConfig(
        num_hidden_layers=raw["num_hidden_layers"],
        hidden_size=raw["hidden_size"],
        num_attention_heads=raw["num_attention_heads"],
        intermediate_size=raw["intermediate_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        vocab_size=raw["vocab_size"],
        type_vocab_size=raw["type_vocab_size"],
        num_labels=raw.get("num_labels", 2),
        pad_token_id=raw.get("pad_token_id", 0),
        layer_norm_eps=raw.get("layer_norm_eps", 1e-12),
        output_hidden_states=True,
    )


class _QuadGELU(nn.Module):
    """MPCFormer quadratic GELU replacement: 0.125 x^2 + 0.25 x + 0.5. An nn.Module
    because HF's BertIntermediate stores intermediate_act_fn as a child module."""

    def forward(self, x):
        return 0.125 * x * x + 0.25 * x + 0.5


def load_model(data_type: str, model_path: str, type: str = "default"):
    """Load the distilled BERT as an HF BertForSequenceClassification.

    ``model_path`` is the path to the model's safetensors file; the matching
    config.json is read from the same directory. The distilled state dict is
    HF-name-matched and complete, so it loads with no missing/unexpected keys.

    If config.json declares ``"activation": "quad"``, each layer's FFN activation
    is swapped for the quadratic GELU so this plaintext reference matches what the
    HE forward computes (he.stage_13_gelu). Any other value keeps HF's GELU.
    """
    model_path = Path(model_path)
    config = load_bert_config(model_path.parent)
    model = BertForSequenceClassification(config)
    model.load_state_dict(load_safetensors(str(model_path)))
    model.eval()

    activation = json.loads((model_path.parent / "config.json").read_text()).get("activation", "exact")
    if activation == "quad":
        for layer in model.bert.encoder.layer:
            layer.intermediate.intermediate_act_fn = _QuadGELU()
    print(f"Model loaded for {data_type} ({config.num_hidden_layers} layers, activation={activation})")
    return model


def ld_entry(matrix: np.ndarray, l: int, i: int):  # noqa: E741
    """
    Get the i th entry of the l th lower diagonal of the matrix
    """
    b, c = matrix.shape
    return matrix[(l + i) % b, i % c]


def matrix_ld(matrix: np.ndarray, l: int):  # noqa: E741
    """
    Get the l th lower diagonal of the matrix
    """
    a, b = matrix.shape
    dim = max(a, b)
    return np.array([ld_entry(matrix, l, i) for i in range(dim)])


def to_blocks(
    matrix: np.ndarray, block_shape: tuple[int, int], diag: bool = True
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Convert the matrix to a list of block matrices.
    Return the blocks in diagonal form if diag is True
    """
    rows, cols = matrix.shape
    block_rows, block_cols = block_shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError("Matrix shape should be divisible by block shape")

    vertical = rows // block_rows
    horizontal = cols // block_cols
    blocks = matrix.reshape(vertical, block_rows, horizontal, block_cols).transpose(0, 2, 1, 3)

    if not diag:
        return blocks, (vertical, horizontal)

    diag_rows = min(vertical, horizontal)
    diag_cols = max(vertical, horizontal)
    diag_blocks = np.empty((diag_rows, diag_cols, block_rows, block_cols), dtype=matrix.dtype)
    diag_row_indices = np.arange(diag_rows)
    for diag_col in range(diag_cols):
        diag_blocks[:, diag_col] = blocks[(diag_row_indices + diag_col) % vertical, diag_col % horizontal]
    return diag_blocks, (diag_rows, diag_cols)
