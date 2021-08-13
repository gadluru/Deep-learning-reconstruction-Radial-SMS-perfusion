import os
import sys
import copy
import time
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import ImageGrid

def r2c(inputs,dim='last'):
    if dim == 'last':
        outputs = inputs[...,0] + 1j*inputs[...,1]
    elif dim == 'first':
        outputs = inputs[:,0,...] + 1j*inputs[:,1,...]

    return outputs.astype(np.complex64)

def c2r(inputs, dim='last'):
    if dim == 'last':
        outputs = np.zeros(inputs.shape + (2,))
        outputs[...,0] = inputs.real
        outputs[...,1] = inputs.imag
    elif dim == 'first':
        outputs = np.zeros((inputs.shape[0],) + (2,) + inputs.shape[1:])
        outputs[:,0,...] = inputs.real
        outputs[:,1,...] = inputs.imag

    return outputs.astype(np.float32)

def brighten(x,beta):

    tol = np.sqrt(sys.float_info.epsilon)
    if beta > 0:
        gamma = 1 - min(1-tol,beta)
    else:
        gamma = 1/(1 + max(-1+tol,beta))

    x = np.power(x,gamma)

    return x

def toPIL(x,truth):

    xPIL = []
    for i in range(x.shape[-1]):
        for j in range(x.shape[0]):
            xPIL.append(TF.to_pil_image(x[j,:,:,i]))

    truthPIL = []
    for i in range(truth.shape[-1]):
        for j in range(truth.shape[0]):
            truthPIL.append(TF.to_pil_image(truth[j,:,:,i]))

    return xPIL,truthPIL

def toTensor(x,truth):
    xTensor = []

    for img in x:
        xTensor.append(np.array(img))

    truthTensor = []
    for img in truth:
        truthTensor.append(np.array(img))

    xTensor = np.array(xTensor)
    truthTensor = np.array(truthTensor)

    nt,sx,sy = xTensor.shape

    xTensor = np.transpose(xTensor,(1,2,0))
    truthTensor = np.transpose(truthTensor,(1,2,0))

    xTensor = np.reshape(xTensor,(sx,sy,nt//2,2))
    truthTensor = np.reshape(truthTensor,(sx,sy,nt//2,2))

    xTensor = np.transpose(xTensor,(3,0,1,2))
    truthTensor = np.transpose(truthTensor,(3,0,1,2))

    xTensor = torch.tensor(xTensor)
    truthTensor = torch.tensor(truthTensor)
    
    return xTensor,truthTensor

def rotate(x,truth):
    angle = random.randint(-10,10)
    
    xRotated = []
    for img in x:
        xRotated.append(TF.rotate(img,angle=angle))
                         
    truthRotated = []        
    for img in truth:
        truthRotated.append(TF.rotate(img,angle=angle))

    return xRotated,truthRotated

def horizontal_flip(x,truth):
    xFlip = []
    for img in x:
        xFlip.append(TF.hflip(img))
                         
    truthFlip = []        
    for img in truth:
        truthFlip.append(TF.hflip(img))

    return xFlip,truthFlip

def vertical_flip(x,truth):
    xFlip = []
    for img in x:
        xFlip.append(TF.vflip(img))
                         
    truthFlip = []        
    for img in truth:
        truthFlip.append(TF.vflip(img))

    return xFlip,truthFlip

def translate(x,truth):

    hTrans = random.randint(-7,7) 
    vTrans = random.randint(-7,7)

    xTranslate = []
    for img in x:
        xTranslate.append(TF.affine(img,angle=0,translate=(hTrans,vTrans),scale=1,shear=0))
                         
    truthTranslate = []        
    for img in truth:
        truthTranslate.append(TF.affine(img,angle=0,translate=(hTrans,vTrans),scale=1,shear=0))

    return xTranslate,truthTranslate

def shear(x,truth):

    shearVal = random.randint(-10,10) 

    xShear = []
    for img in x:
        xShear.append(TF.affine(img,angle=0,translate=(0,0),scale=1,shear=shearVal))
                         
    truthShear = []        
    for img in truth:
        truthShear.append(TF.affine(img,angle=0,translate=(0,0),scale=1,shear=shearVal))

    return xShear,truthShear

def normalize(x):

    if type(x) is torch.Tensor:
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))

def check_orientation(x):
    x = np.transpose(x[2,:,:,:],(1,2,0))

    x = np.sum(x,axis=-1)
    
    o1 = x
    o2 = np.rot90(x,1)
    o3 = np.rot90(x,2)
    o4 = np.rot90(x,3)
    o5 = np.flipud(o1)
    o6 = np.flipud(o2)
    o7 = np.flipud(o3)
    o8 = np.flipud(o4)

    fig,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2,ncols=4)

    ax1.imshow(o1,cmap='gray')
    ax1.set_title(1)

    ax2.imshow(o2,cmap='gray')
    ax2.set_title(2)

    ax3.imshow(o3,cmap='gray')
    ax3.set_title(3)

    ax4.imshow(o4,cmap='gray')
    ax4.set_title(4)

    ax5.imshow(o5,cmap='gray')
    ax5.set_title(5)

    ax6.imshow(o6,cmap='gray')
    ax6.set_title(6)

    ax7.imshow(o7,cmap='gray')
    ax7.set_title(7)

    ax8.imshow(o8,cmap='gray')
    ax8.set_title(8)
    
    plt.draw()
    plt.pause(1)

    correct = input('select correct orientation ')
    
    plt.close(fig)

    return correct

def orientate(x,orientation):

    x = np.transpose(x,(2,3,0,1))

    if orientation == 2:
        x = np.rot90(x,1)
    if orientation == 3:
        x = np.rot90(x,2)
    if orientation == 4:
        x = np.rot90(x,3)
    if orientation == 5:
        x = np.flipud(x)
    if orientation == 6:
        x = np.flipud(np.rot90(x,1))
    if orientation == 7:
        x = np.flipud(np.rot90(x,2))
    if orientation == 8:
        x = np.flipud(np.rot90(x,3))

    x = np.transpose(x,(2,3,0,1))

    return x

def validate_network(model,data_loader,loss_function,tracker,device):
    model.eval()

    with torch.no_grad():
        for inputs,targets in data_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_function(outputs,targets)

            tracker.calculate_metrics(outputs.detach().cpu().numpy(),targets.detach().cpu().numpy())

def test_network(model,data_loader,device):
    model.eval()

    start = time.time() 
    with torch.no_grad():
        inputs = []
        outputs = []
        targets = []
        for ins,tars in data_loader:

            ins = ins.to(device)
            tars = tars.to(device)

            outs = model(ins)

            ins = ins.detach().cpu().numpy()
            outs = outs.detach().cpu().numpy()
            tars = tars.detach().cpu().numpy()

            inputs.append(ins)
            outputs.append(outs)
            targets.append(tars)

    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    end = time.time()

    return inputs,outputs,targets,(end-start)


class metric_tracker():
    def __init__(self,mean,std):
        self.tracker = {'ssim':[],'psnr':[],'nrmse':[]}

        self.mean = mean
        self.std = std

    def calculate_metrics(self,outputs,targets):
        outs = outputs*self.std + self.mean
        tars = targets*self.std + self.mean

        nb,ch,sx,sy,nt = tars.shape

        outs = np.transpose(outs,[0,4,1,2,3])
        tars = np.transpose(tars,[0,4,1,2,3])

        outs = np.reshape(outs,[nb*nt,ch,sx,sy])
        tars = np.reshape(tars,[nb*nt,ch,sx,sy])

        outs = np.abs(r2c(outs,dim='first'))
        tars = np.abs(r2c(tars,dim='first'))
        
        nt,sx,sy = tars.shape

        for i in range(nt):
            self.tracker['ssim'].append(ssim(tars[i,...],outs[i,...],data_range=np.max(tars)))
            self.tracker['psnr'].append(psnr(tars[i,...],outs[i,...],data_range=np.max(tars)))
            self.tracker['nrmse'].append(nrmse(tars[i,...],outs[i,...]))

    def return_metrics(self):
        return self.tracker.copy()

    def clear_metrics(self):
        for key,value in self.tracker.items():
            self.tracker[key] = []

def tb_add_scalar(writer,step,loss_dict):
    for key,value in loss_dict.items():
        writer.add_scalar(key,np.mean(value),step)

def mMaker(inputs,outputs,targets,savelocation,savename='results.gif'):

    input_diff = np.abs(targets - inputs)
    output_diff = np.abs(targets - outputs)
    target_diff = np.abs(targets - targets)

    fig = plt.figure()

    grid = ImageGrid(fig,121,nrows_ncols=(3,3),axes_pad=0,direction='column')
    grid_diff = ImageGrid(fig,122,nrows_ncols=(3,3),axes_pad=0,direction='column')

    def update(i):

        labels = ['slice 1','slice 2','slice 3']
        input_recon = inputs[:,i,:,:]
        output_recon = outputs[:,i,:,:]
        target_recon = targets[:,i,:,:]

        nsl,nfr,sx,sy = targets.shape
        
        count = 0
        
        for j in range(nsl):
            grid[count].imshow(input_recon[j,:,:],cmap='gray')
            grid[count].set_ylabel(labels[j],fontsize=12)
            grid[count].set_xticks([])
            grid[count].set_yticks([])
            if j == 0:
                grid[count].set_title('input',fontsize=12)
            count = count + 1

        for j in range(nsl):
            grid[count].imshow(output_recon[j,:,:],cmap='gray')
            grid[count].set_xticks([])
            grid[count].set_yticks([])
            if j == 0:
                grid[count].set_title('BU3',fontsize=12)
            count = count + 1

        for j in range(nsl):
            grid[count].imshow(target_recon[j,:,:],cmap='gray')
            grid[count].set_xticks([])
            grid[count].set_yticks([])
            if j == 0:
                grid[count].set_title('target',fontsize=12)
            count = count + 1

        count = 0

        labels = ['slice 1','slice 2','slice 3']
        input_diff_recon = input_diff[:,i,:,:]
        output_diff_recon = output_diff[:,i,:,:]
        target_diff_recon = target_diff[:,i,:,:]
        
        count = 0
        
        for j in range(nsl):
            grid_diff[count].imshow(input_diff_recon[j,:,:])
            grid_diff[count].set_ylabel(labels[j],fontsize=12)
            grid_diff[count].set_xticks([])
            grid_diff[count].set_yticks([])
            if j == 0:
                grid_diff[count].set_title('input',fontsize=12)
            count = count + 1
    
        for j in range(nsl):
            grid_diff[count].imshow(output_diff_recon[j,:,:])
            grid_diff[count].set_xticks([])
            grid_diff[count].set_yticks([])
            if j == 0:
                grid_diff[count].set_title('BU3',fontsize=12)
            count = count + 1

        for j in range(nsl):
            grid_diff[count].imshow(target_diff_recon[j,:,:])
            grid_diff[count].set_xticks([])
            grid_diff[count].set_yticks([])
            if j == 0:
                grid_diff[count].set_title('target',fontsize=12)
            count = count + 1

        count = 0

    anim = FuncAnimation(fig, update, frames = np.arange(0,60), interval = 100)
    anim.save(savelocation+savename, dpi = 100, writer = 'imagemagick')
