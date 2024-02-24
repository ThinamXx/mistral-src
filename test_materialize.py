# This is a test file to test the materialize function from xformers
# https://github.com/facebookresearch/xformers/blob/158f36c02fcc52b1d88c045437e7aa3c47f3a7e3/xformers/ops/fmha/attn_bias.py

from typing import Tuple, Union
import torch


window_left = 0
window_right = 0


def materialize(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    mask = torch.full(
        shape,
        dtype=create_as,
        fill_value=1,
        device=device,
    )
    print(mask)

    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries

    mask = torch.triu(mask, diagonal=shift - window_left)
    print(mask)
    mask = torch.tril(mask, diagonal=shift + window_right)
    print(mask)
    mask = torch.log(mask)
    print(mask)
    return mask.to(dtype)


if __name__ == "__main__":
    mask = materialize((2, 2, 2, 2))
    print(mask)
