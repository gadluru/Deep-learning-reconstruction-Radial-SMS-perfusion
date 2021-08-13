import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary
from unet import *

class booster_runet(torch.nn.Module):
    def __init__(self,in_features, out_features, resolution, layers, N=16, dp=False, rl=False):
        super(booster_runet,self).__init__()

        self.unet1 = Unet(in_features=in_features, out_features=out_features, resolution=resolution, layers=layers, dp=dp, rl=rl)

        self.unet2 = Unet(in_features=2*in_features, out_features=out_features, resolution=resolution, layers=layers, dp=dp, rl=rl)

    def initialize_weights(self,m):
        if isinstance(m,nn.Conv3d):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data,0)
        elif isinstance(m,nn.BatchNorm3d):
            nn.init.constant_(m.weight.data,1)
            nn.init.constant_(m.bias.data,0)
            
    def forward(self,x):

        out1 = self.unet1(x)

        out2 = torch.cat((out1,x),dim=1)

        out2 = self.unet2(out2)

        out = out1 + out2 + x

        return out

if __name__ == '__main__':
    net = booster_runet(in_features=2, out_features=2, resolution=64, layers=3, dp=True)

    summary(net.cuda(),(2,144,144,32))
