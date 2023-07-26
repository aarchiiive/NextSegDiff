import torch.nn as nn

from .nn import conv_bn, conv_dw


class MobBlock(nn.Module):
    def __init__(self,ind):
        super().__init__()


        if ind == 0:
            self.stage = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 1),
            conv_dw(128, 128, 1)
        )
        elif ind == 1:
            self.stage  = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1)
        )
        elif ind == 2:
            self.stage = nn.Sequential(
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 1)
            )
        else:
            self.stage = nn.Sequential(
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1)
            )

    def forward(self,x):
        return self.stage(x)