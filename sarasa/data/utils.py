import torch

from sarasa.models.attention import VarlenMetaData


def prepare_varlen_metadata(
    input_dict: dict[str, torch.Tensor],
    bos_token_id: int,
) -> dict[str, torch.Tensor]:
    # add metadata for varlen attention

    input = input_dict["input"]  # T

    (bos_positions,) = (input == bos_token_id).nonzero(as_tuple=True)
    bos_positions = bos_positions.to(torch.int32)
    cu_seq = torch.cat((
        bos_positions,
        bos_positions.new_tensor([input.size(0)]),
    ))
    max_seqlen = torch.diff(cu_seq).max()
    metadata = VarlenMetaData(
        cu_seq_q=cu_seq,
        cu_seq_k=cu_seq,
        max_q=max_seqlen.item(),
        max_k=max_seqlen.item(),
    )
    input_dict["metadata"] = metadata
    return input_dict
