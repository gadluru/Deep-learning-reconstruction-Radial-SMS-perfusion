import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

class input_layer(torch.nn.Module):
    def __init__(self,in_features,out_features):
        super(input_layer,self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        return out

class encode_layer(torch.nn.Module):
    def __init__(self,in_features,out_features,dp = False):
        super(encode_layer,self).__init__()

        self.dp = dp

        self.bnorm1 = nn.BatchNorm3d(in_features)

        self.maxpool = nn.MaxPool3d(kernel_size = 2)

        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        if self.dp:
            self.drop = nn.Dropout3d(p=0.25)

            self.conv3 = nn.Conv3d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv3d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1)
            self.relu4 = nn.ReLU()
            
    def forward(self,x):
        out = self.bnorm1(x)
        out = self.maxpool(out)
        
        out = self.conv1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        if self.dp:
            out = self.drop(out)

            out = self.conv3(out)
            out = self.relu3(out)

            out = self.conv4(out)
            out = self.relu4(out)
        
        return out        

class decode_layer(torch.nn.Module):
    def __init__(self,in_features,out_features):
        super(decode_layer,self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self,x1,x2):

        out = self.conv(x1)
        out = self.relu(out)
        out = self.up(out)

        out = torch.cat((out,x2),1)

        out = self.conv1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        return out

class output_layer(torch.nn.Module):
    def __init__(self,in_features,out_features,rl = False):
        super(output_layer,self).__init__()

        self.rl = rl

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=1)
    
    def forward(self,x,input):

        out = self.conv(x)

        if self.rl:
            out = out + input

        return out

class Unet(torch.nn.Module):
    def __init__(self,in_features,out_features,resolution,layers,dp = False,rl = False):
        super(Unet,self).__init__()

        self.inc = input_layer(in_features=in_features,out_features = resolution)

        self.encode = nn.ModuleList()
        for i in range(layers):
            filters = resolution * 2**(i)
            if i == (layers-1):
                self.encode.append(encode_layer(in_features=filters, out_features=2*filters, dp=dp))
            else:
                self.encode.append(encode_layer(in_features=filters, out_features=2*filters, dp=False))

        self.decode = nn.ModuleList()
        for i in reversed(range(layers)):
            filters = resolution * 2**(i)

            self.decode.append(decode_layer(in_features=2*filters, out_features=filters))

        self.out = output_layer(in_features = resolution, out_features= out_features, rl = rl)
                
    def forward(self,x):
        blocks = []

        out = self.inc(x)
        blocks.append(out)
        for i,layer in enumerate(self.encode):
            out = layer(out)
            if i != len(self.encode)-1:
                blocks.append(out)

        for i,layer in enumerate(self.decode):
            out = layer(out,blocks[len(blocks)-i-1])

        out = self.out(out,x)

        return out

if __name__ == '__main__':
    net = Unet(in_features=2, out_features=2, resolution=64, layers=3, dp=True)

    summary(net.cuda(),(2,144,144,32))
