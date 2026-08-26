"""Document boundaries inside a packed block, recovered from `input_ids`.

`corpus.build` stores no boundary column, and one derivation has to feed both the student's mask
and the position_ids the HF teacher re-derives its own from.
"""

import torch


# Positions restart at index 0 or after an eos. A mid-document chop restarts too: the fragment
# has no prefix in the block, so presenting it as a fresh sequence is the honest reading.
def doc_positions(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    B, S = input_ids.shape
    idx = torch.arange(S, device=input_ids.device).expand(B, S)
    starts = torch.zeros_like(input_ids)
    starts[:, 1:] = (input_ids[:, :-1] == eos_id).to(input_ids.dtype)
    return idx - torch.cummax(idx * starts, dim=-1).values


# Segment index per token, deliberately the same rule as HF's find_packed_sequence_indices.
def doc_ids(position_ids: torch.Tensor) -> torch.Tensor:
    first = position_ids[:, :1] - 1  # so the diff at index 0 reads 1, never a boundary
    return (torch.diff(position_ids, prepend=first, dim=-1) != 1).cumsum(-1)
