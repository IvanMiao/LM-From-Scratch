import torch
import einops


x = torch.tensor([
        [0., 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])

assert x.stride(0) == 4
assert x.stride(1) == 1
