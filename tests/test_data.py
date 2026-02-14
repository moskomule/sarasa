import torch

from sarasa.data.utils import prepare_varlen_metadata


def test_create_varlen_metadata():
    input = torch.tensor([0, 1, 1, 1, 0, 1, 1], dtype=torch.long)
    input_dict = {"input": input}
    bos_token_id = 0
    input_dict = prepare_varlen_metadata(input_dict, bos_token_id)
    metadata = input_dict["metadata"]
    assert metadata.cu_seq_q.tolist() == [0, 4, 7]
    assert metadata.cu_seq_k.tolist() == [0, 4, 7]
    assert metadata.max_q == 4
    assert metadata.max_k == 4
