"""Document boundaries inside a packed block, derived rather than stored.

`corpus.build` writes packed blocks with an all-ones mask and no boundary column, so the signal has
to come back out of `input_ids`. One derivation feeds both sides: the student's mask and the
`position_ids` handed to the HF teacher, which recovers the same segments itself.
"""

import torch


def doc_positions(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Positions restarting at every document start: index 0, or any index following an eos.

    A mid-document chop therefore restarts too -- the fragment has no prefix in the block, so
    presenting it as a fresh sequence is the honest reading.
    """
    B, S = input_ids.shape
    idx = torch.arange(S, device=input_ids.device).expand(B, S)
    starts = torch.zeros_like(input_ids)
    starts[:, 1:] = (input_ids[:, :-1] == eos_id).to(input_ids.dtype)
    return idx - torch.cummax(idx * starts, dim=-1).values


def doc_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Segment index per token. Deliberately the same rule as HF's
    `masking_utils.find_packed_sequence_indices`, so student and teacher cannot drift apart."""
    first = position_ids[:, :1] - 1  # so the diff at index 0 reads 1, never a boundary
    return (torch.diff(position_ids, prepend=first, dim=-1) != 1).cumsum(-1)
