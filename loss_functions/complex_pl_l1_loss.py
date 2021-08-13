import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary

class complex_pl_l1_loss(nn.Module):
    def __init__(self,alpha,beta):
        super(complex_pl_l1_loss,self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.tracker = {'pl':[],'l1':[]}

    def perceptual_loss(self,predicted,target):

        nb,ch,sy,sx,nt = predicted.shape

        predicted = predicted.view(nb,ch,sy,sx,nt)

        model = models.vgg16(pretrained=True).features[:16].cuda()

        for param in model.parameters():
            param.requires_grad = False

        loss = 0
        for i in range(ch):
            predicted_rgb = predicted[:,i,None,:,:,:].repeat(1,3,1,1,1)
            target_rgb = target[:,i,None,:,:,:].repeat(1,3,1,1,1)

            predicted_rgb = predicted_rgb.permute((0,4,1,2,3)).contiguous()
            target_rgb = target_rgb.permute((0,4,1,2,3)).contiguous()

            nb,nt,ch,sy,sx = predicted_rgb.shape
            
            predicted_rgb = predicted_rgb.view((nb*nt,ch,sy,sx))
            target_rgb = target_rgb.view((nb*nt,ch,sy,sx))

            loss = loss + torch.mean((model(target_rgb) - model(predicted_rgb))**2)
            
        return loss/ch
    
    def l1_loss(self,predicted,target):
        
        nb,ch,sy,sx,nt = predicted.shape

        loss = 0
        for i in range(ch):
            predicted_rgb = predicted[:,i,None,:,:,:].repeat(1,3,1,1,1)
            target_rgb = target[:,i,None,:,:,:].repeat(1,3,1,1,1)

            predicted_rgb = predicted_rgb.permute((0,4,1,2,3)).contiguous()
            target_rgb = target_rgb.permute((0,4,1,2,3)).contiguous()

            nb,nt,ch,sy,sx = predicted_rgb.shape
            
            predicted_rgb = predicted_rgb.view((nb*nt,ch,sy,sx))
            target_rgb = target_rgb.view((nb*nt,ch,sy,sx))
        
            loss = loss + torch.mean(torch.abs(target_rgb - predicted_rgb))
        
        return loss/ch

    def return_loss(self):
        return self.tracker.copy()

    def clear_loss(self):
        for key,value in self.tracker.items():
            self.tracker[key] = []

    def forward(self,outputs,targets):
        pl = self.alpha*self.perceptual_loss(outputs,targets)
        l1 = self.beta*self.l1_loss(outputs,targets)
        
        self.tracker['pl'].append(pl.detach().cpu().numpy())
        self.tracker['l1'].append(l1.detach().cpu().numpy())

        loss = pl + l1
        
        return loss
