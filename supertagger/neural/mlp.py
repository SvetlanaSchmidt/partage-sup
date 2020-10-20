import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):

    def __init__(
            self,
            in_size: int,
            hid_size: int,
            out_size: int,
            dropout: float,
            out_dropout: float,
    ):
        super(MLP, self).__init__()

        # [Linear -> Activation -> Dropout -> Linear -> Dropout]
        self.net = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hid_size, out_size),
            nn.Dropout(p=out_dropout, inplace=True),
        )

    def forward(self, vec: Tensor) -> Tensor:
        return self.net(vec)
