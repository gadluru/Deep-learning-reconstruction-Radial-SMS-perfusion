import sys
sys.path.append('../loss_functions')
sys.path.append('../template_models/')
sys.path.append('../utils')

import os.path
import torch.nn
import numpy as np
import torch
import time
import pickle
from datetime import datetime
from torchsummary import summary
from loadData import *
from complex_pl_l1_loss import *
from booster_runet import *
from torch.utils.tensorboard import SummaryWriter
from supportingFunctions import *

def residual_booster_network(N,batch,in_features,out_features,resolution,layers,dp,alpha,beta,lr,nepoch,save_interval,directory,save_directory,pretrain,device):

# -------------------------------------------------------------------------------------------
#
#     residual_booster_network(N,batch,in_features,out_features,resolution,layers,dp,alpha,beta,lr,nepoch,save_interval,directory,save_directory,device)
#
# -------------------------------------------------------------------------------------------
#    
#     inputs (2D radial SMS myocardial perfusion datasets)
#
#        -N (integer: 8|16|32): number of training time frames for each batch [default: 32]
#        -batch (integer): mini-batch size for training [default: 3]
#        -in_features (integer: 2): number of input channels (real and imaginary components)
#        -out_features (integer: 2): number of output channels (real and imaginary components)
#        -resolution (integer): number of filters in the first layer of each Unet in the residual booster network, doubled for each layer [default: 64]
#        -layers (integer): number of layers in each Unet in the residual booster network [default: 3]
#        -dp (boolean): adds 25% dropout and additional convolutional layers to each Unet [default: True]
#        -alpha (float): weight for perceptual loss component of the loss function [default: 0.04]
#        -beta (float): weight for the L1 component of the loss function [default: 1.0] 
#        -lr (float): learning rate for the adam optimizer [default: 0.0003]
#        -nepoch (integer): number of epochs to train the network [default: 100]
#        -save_interval (integer): epoch interval to save intermediate networks [default: 20]
#        -directory (string): directory to save the network and other parameters
#        -save_directory (string): name to save network
#        -device (cuda|cpu): determines whether to train network on GPU or CPU if not available
#        -pretrain (string): location of gated network for transfer learning

    #load SMS data for training
    train_set = loadData(N=N,split='train',do_transform=True,directory=directory)
    validation_set = loadData(N=N,split='val',do_transform=False,directory=directory)
    
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=batch,shuffle=False)

    #residual booster network initialization
    model = booster_runet(in_features=in_features, out_features=out_features, resolution=resolution, layers=layers, dp=dp).to(device)
    
    if pretrain is not None:
        state = torch.load(pretrain)
        model.load_state_dict(state['state_dict'])
    else:
        #orthogonal weight initialization
        model.apply(model.initialize_weights)

    #complex perceptual loss and L1 loss function
    pl_l1 = complex_pl_l1_loss(alpha=alpha,beta=beta)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    with open(directory+'scale_factors.pkl','rb') as f:
        mean,std = pickle.load(f)

    #tensorboard training visualization
    train_writer = SummaryWriter('runs/' + directory[15:] + 'training')
    validation_writer = SummaryWriter('runs/' + directory[15:] + 'validation')

    #tracker for calculating SSIM, PSNR, and NRMSE during training
    train_metric_tracker = metric_tracker(mean,std)
    validation_metric_tracker = metric_tracker(mean,std)
    
    print('beginning training...')
    print('network saved to ',directory,' as ', save_directory)

    step = 0
    currentLoss = 0
    for epoch in range(nepoch):
        start = time.time()
        for inputs,targets in train_loader:
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            loss = pl_l1(outputs,targets)

            train_metric_tracker.calculate_metrics(outputs.detach().cpu().numpy(),targets.detach().cpu().numpy())

            loss.backward()
            
            optimizer.step()
            
        train_loss_dict = pl_l1.return_loss()
        train_metric_dict = train_metric_tracker.return_metrics()
        
        #tensorboard writer for training intermediate reconstructions, loss, and image metrics
        tb_add_scalar(train_writer,step,train_loss_dict)
        tb_add_scalar(train_writer,step,train_metric_dict)

        pl_l1.clear_loss()
        train_metric_tracker.clear_metrics()

        validate_network(model,validation_loader,pl_l1,validation_metric_tracker,device)
        
        validation_loss_dict = pl_l1.return_loss()
        validation_metric_dict = validation_metric_tracker.return_metrics()

        #tensorboard writer for validation intermediate reconstructions, loss, and image metrics
        tb_add_scalar(validation_writer,step,validation_loss_dict)
        tb_add_scalar(validation_writer,step,validation_metric_dict)

        pl_l1.clear_loss()
        validation_metric_tracker.clear_metrics()

        train_writer.flush()
        validation_writer.flush()

        state = {'epoch': epoch,
                 'state_dict':model.state_dict(),
                 'optimizer':optimizer.state_dict()}

        step = step + 1

        end = time.time()        
        
        train_loss = np.sum([np.mean(value) for key,value in train_loss_dict.items()])
        validation_loss = np.sum([np.mean(value) for key,value in validation_loss_dict.items()])

        print('Epoch: %.d' %epoch, end='   ')  
        print('Time: %.4f' %(end-start))
        print('Train Loss: %.4f' %train_loss)
        print('Validation Loss: %.4f' %validation_loss)
        print('')

        #saves best network according to minimal validation loss
        if currentLoss == 0 or validation_loss < currentLoss:
            currentLoss = validation_loss

            try:
                os.system('rm ' + currentSave)
            except:
                pass
            
            currentSave = (directory + 'residual_booster_network_epoch_%.d_val_%.4f.pth' %(epoch,currentLoss))
            torch.save(state,currentSave)

        # saves intermediate networks according to save interval
        if epoch % save_interval == 0:
            torch.save(state,directory+'checkpoint_epoch_%.d_val_%.4f.pth' %(epoch,validation_loss))

    torch.save(state,directory+save_directory+'.pth')
    torch.onnx.export(model,torch.zeros((3,2,144,144,32),device='cuda'),directory+save_directory+'.onnx',opset_version=11)
