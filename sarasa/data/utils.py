from typing import Any

import torch

from sarasa.models.attention import VarlenMetaData


def prepare_varlen_metadata(
    input_dict: dict[str, Any],
    bos_token_id: int,
) -> dict[str, torch.Tensor | VarlenMetaData]:
    # add metadata for varlen attention

    assert bos_token_id is not None

    input = input_dict["input"]  # T

    (bos_positions,) = (input == bos_token_id).nonzero(as_tuple=True)
    bos_positions = bos_positions.to(torch.int32)
    if len(bos_positions) == 0 or bos_positions[0] != 0:
        bos_positions = torch.cat((bos_positions.new_tensor([0]), bos_positions))
    cu_seq = torch.cat((
        bos_positions,
        bos_positions.new_tensor([input.size(0)]),
    ))
    max_seqlen = torch.diff(cu_seq).max().item()
    metadata = VarlenMetaData(
        cu_seq_q=cu_seq,
        cu_seq_k=cu_seq,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )
    input_dict["metadata"] = metadata
    return input_dict
