"""Turning text into pooled embeddings, one implementation, so no two scorers can drift on
tokenization, truncation or batching."""

from __future__ import annotations

from contextlib import nullcontext

import torch


@torch.no_grad()
def embed_batches(backend, task, batches, device, *models, ctx=nullcontext):
    """Pooled fp32 CPU embeddings per model, from ONE pass: a second pass would have to reproduce
    the batching to keep student and teacher rows aligned. ``ctx`` is entered per batch."""
    outs = [[] for _ in models]
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = task.model_inputs(batch)
        with ctx():
            for acc, model in zip(outs, models, strict=True):
                acc.append(backend.forward(model, inputs)["output"].float().cpu())
    return tuple(torch.cat(a) for a in outs)


@torch.no_grad()
def embed_texts(backend, model, tokenizer, cfg, texts, device, batch_size: int = 32):
    """Embed a plain list of strings (tokenize + pool), truncated at ``cfg.train.max_seq_len``.

    Batches are length-sorted and the original order restored afterwards. Unsorted, each batch pads
    to its longest member: at ``max_seq_len`` 1024 that is ~1024 tokens per text against a median
    near 130. Longest-first, so an over-large batch fails on step 1 rather than 90% of the way in.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    out: list[torch.Tensor] = [None] * len(texts)
    for i in range(0, len(order), batch_size):
        chunk = order[i : i + batch_size]
        enc = tokenizer(
            [texts[j] for j in chunk],
            padding=True,
            truncation=True,
            max_length=cfg.train.max_seq_len,
            return_tensors="pt",
        ).to(device)
        inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        emb = backend.forward(model, inputs)["output"].float().cpu()
        for k, j in enumerate(chunk):
            out[j] = emb[k]
    return torch.stack(out)
