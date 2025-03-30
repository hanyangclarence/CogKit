import torch.nn as nn


class AdditionFuser(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, f"Shapes of x1 {x1.shape} and x2 {x2.shape} do not match"
        return x1 + x2