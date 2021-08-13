import sys
sys.path.append('../train_network')
sys.path.append('../loss_functions')
sys.path.append('../template_models')
sys.path.append('../utils')

import os.path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from supportingFunctions import *
import argparse
import scipy.io
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
from booster_runet import *
from loadData import *
import time
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------------------------------
#
# visualize results of trained network
# run in command-line 
# command: python visualize test_mode nset gif
#                 -test_mode (integer): selects test dataset to visualize [default: 0]
#                 -nset (integer 0|1|2): selects slice group to visualize [default: 1]
#                 -gif (boolean True|False): choose whether to generate movie or not, saves to results directory [default: 1 (True)]
#
# ------------------------------------------------------------------------------------------------------



# Modify this line with location of trained network
directory = 'trainedNetwork/13Jul_0937am/'

with open(directory+'parameters.pkl','rb') as f:
    parameters = pickle.load(f)

with open(directory+'scale_factors.pkl','rb') as f:
    mean,std = pickle.load(f)

state = torch.load(directory+parameters['save_directory']+'.pth')

parser = argparse.ArgumentParser(description = 'visualize results of network')
parser.add_argument('--test_mode', type = int, default=1)
parser.add_argument('--nset',type = int, default=1)
parser.add_argument('--gif',type = int, default=1)
args = parser.parse_args()

savelocation = 'results/testset' + str(args.test_mode) + '/set' + str(args.nset) + '/'

if not os.path.exists(savelocation):
    os.makedirs(savelocation)

testset = loadData(N=parameters['N'], split='test', test_mode=args.test_mode, 
                   directory=directory)

test_loader = torch.utils.data.DataLoader(testset,batch_size=parameters['batch'])

model = booster_runet(in_features=parameters['in_features'], out_features=parameters['out_features'], resolution=parameters['resolution'], layers=parameters['layers'], dp=parameters['dp']).to(device)

model.load_state_dict(state['state_dict'])

inputs,outputs,targets,total_time = test_network(model,test_loader,device)

test_metric_tracker = metric_tracker(mean,std)

test_metric_tracker.calculate_metrics(outputs,targets)

test_metric_dict = test_metric_tracker.return_metrics()

for key,value in test_metric_dict.items():
    test_metric_dict[key] = {'mean': np.mean(value), 'std': np.std(value)}

print('')
print('TIME: %.4f'  %total_time)
print('SSIM: %.4f +/- %.4f' %(test_metric_dict['ssim']['mean'],test_metric_dict['ssim']['std']))
print('PSNR: %.4f +/- %.4f' %(test_metric_dict['psnr']['mean'],test_metric_dict['psnr']['std']))
print('NRMSE: %.4f +/- %.4f' %(test_metric_dict['nrmse']['mean'],test_metric_dict['nrmse']['std']))

nb,ch,sy,sx,nt = targets.shape

inputs = inputs*std + mean
outputs = outputs*std + mean
targets = targets*std + mean

inputs = np.abs(r2c(inputs,dim='first'))
outputs = np.abs(r2c(outputs,dim='first'))
targets = np.abs(r2c(targets,dim='first'))

inputs_sys = inputs[:nb//2,...]
outputs_sys = outputs[:nb//2,...]
targets_sys = targets[:nb//2,...]

inputs_dia = inputs[nb//2:,...]
outputs_dia = outputs[nb//2:,...]
targets_dia = targets[nb//2:,...]

inputs_sys = np.transpose(inputs_sys,(0,3,1,2))
outputs_sys = np.transpose(outputs_sys,(0,3,1,2))
targets_sys = np.transpose(targets_sys,(0,3,1,2))

inputs_dia = np.transpose(inputs_dia,(0,3,1,2))
outputs_dia = np.transpose(outputs_dia,(0,3,1,2))
targets_dia = np.transpose(targets_dia,(0,3,1,2))

nb,nt,sx,sy = targets_sys.shape

inputs_sys = np.reshape(inputs_sys,(nb*nt,sy,sx))
outputs_sys = np.reshape(outputs_sys,(nb*nt,sy,sx))
targets_sys = np.reshape(targets_sys,(nb*nt,sy,sx))

inputs_dia = np.reshape(inputs_dia,(nb*nt,sy,sx))
outputs_dia = np.reshape(outputs_dia,(nb*nt,sy,sx))
targets_dia = np.reshape(targets_dia,(nb*nt,sy,sx))

nt,sy,sx = targets_sys.shape
nsl = 3

inputs_sys = np.reshape(inputs_sys,(nsl,nt//nsl,sy,sx))
outputs_sys = np.reshape(outputs_sys,(nsl,nt//nsl,sy,sx))
targets_sys = np.reshape(targets_sys,(nsl,nt//nsl,sy,sx))

inputs_dia = np.reshape(inputs_dia,(nsl,nt//nsl,sy,sx))
outputs_dia = np.reshape(outputs_dia,(nsl,nt//nsl,sy,sx))
targets_dia = np.reshape(targets_dia,(nsl,nt//nsl,sy,sx))

nsl,nfr,sy,sx = targets_sys.shape
clip = (nfr//3)*args.nset

inputs_sys = inputs_sys[:,clip:clip+60,:,:]
outputs_sys = outputs_sys[:,clip:clip+60,:,:]
targets_sys = targets_sys[:,clip:clip+60,:,:]

inputs_dia = inputs_dia[:,clip:clip+60,:,:]
outputs_dia = outputs_dia[:,clip:clip+60,:,:]
targets_dia = targets_dia[:,clip:clip+60,:,:]

orientation = int(check_orientation(targets_sys))

inputs_sys = orientate(inputs_sys,orientation)
outputs_sys = orientate(outputs_sys,orientation)
targets_sys = orientate(targets_sys,orientation)

inputs_dia = orientate(inputs_dia,orientation)
outputs_dia = orientate(outputs_dia,orientation)
targets_dia = orientate(targets_dia,orientation)

inputs_sys = normalize(inputs_sys)
outputs_sys = normalize(outputs_sys)
targets_sys = normalize(targets_sys)

inputs_dia = normalize(inputs_dia)
outputs_dia = normalize(outputs_dia)
targets_dia = normalize(targets_dia)

inputs_sys = brighten(inputs_sys,0.4)
outputs_sys = brighten(outputs_sys,0.4)
targets_sys = brighten(targets_sys,0.4)

inputs_dia = brighten(inputs_dia,0.4)
outputs_dia = brighten(outputs_dia,0.4)
targets_dia = brighten(targets_dia,0.4)

diff_sys = np.abs(targets_sys - outputs_sys)
diff_dia = np.abs(targets_dia - outputs_dia)

nsl,nt,sy,sx = targets_sys.shape

print('generating movie...')
print('saving movie to' + savelocation)
if args.gif:
    mMaker(inputs_sys,outputs_sys,targets_sys,savelocation,'results_sys.gif')
    mMaker(inputs_dia,outputs_dia,targets_dia,savelocation,'results_dia.gif')

fig = plt.figure()
grid_sys = ImageGrid(fig,121,nrows_ncols=(3,4),axes_pad=0,direction='column')
grid_dia = ImageGrid(fig,122,nrows_ncols=(3,4),axes_pad=0,direction='column')
plt.ion()
for i in range(nt):
    labels = ['slice 1','slice 2','slice 3']
    input_recon = inputs_sys[:,i,:,:]
    output_recon = outputs_sys[:,i,:,:]
    truth_recon = targets_sys[:,i,:,:]
    diff_recon = diff_sys[:,i,:,:]
    count = 0

    for j in range(nsl):
        grid_sys[count].imshow(input_recon[j,:,:],cmap='gray')
        grid_sys[count].set_ylabel(labels[j],fontsize=12)
        grid_sys[count].set_xticks([])
        grid_sys[count].set_yticks([])
        if j == 0:
            grid_sys[count].set_title('input',fontsize=12)
        count = count + 1

    for j in range(nsl):
        grid_sys[count].imshow(output_recon[j,:,:],cmap='gray')
        grid_sys[count].set_xticks([])
        grid_sys[count].set_yticks([])
        if j == 0:
            grid_sys[count].set_title('BU3',fontsize=12)
        count = count + 1

    for j in range(nsl):
        grid_sys[count].imshow(truth_recon[j,:,:],cmap='gray')
        grid_sys[count].set_xticks([])
        grid_sys[count].set_yticks([])
        if j == 0:
            grid_sys[count].set_title('target',fontsize=12)
        count = count + 1

    for j in range(nsl):
        grid_sys[count].imshow(diff_recon[j,:,:],cmap='gray')
        grid_sys[count].set_xticks([])
        grid_sys[count].set_yticks([])
        if j == 0:
            grid_sys[count].set_title('diff',fontsize=12)
        count = count + 1
        
    count = 0

    labels = ['slice 1','slice 2','slice 3']
    input_recon = inputs_dia[:,i,:,:]
    output_recon = outputs_dia[:,i,:,:]
    truth_recon = targets_dia[:,i,:,:]
    diff_recon = diff_dia[:,i,:,:]

    for j in range(nsl):
        grid_dia[count].imshow(input_recon[j,:,:],cmap='gray')
        grid_dia[count].set_ylabel(labels[j],fontsize=12)
        grid_dia[count].set_xticks([])
        grid_dia[count].set_yticks([])
        if j == 0:
            grid_dia[count].set_title('input',fontsize=12)
        count = count + 1

    for j in range(nsl):
        grid_dia[count].imshow(output_recon[j,:,:],cmap='gray')
        grid_dia[count].set_xticks([])
        grid_dia[count].set_yticks([])
        if j == 0:
            grid_dia[count].set_title('BU3',fontsize=12)
        count = count + 1

    for j in range(nsl):
        grid_dia[count].imshow(truth_recon[j,:,:],cmap='gray')
        grid_dia[count].set_xticks([])
        grid_dia[count].set_yticks([])
        if j == 0:
            grid_dia[count].set_title('target',fontsize=12)
        count = count + 1

    for j in range(nsl):
        grid_dia[count].imshow(diff_recon[j,:,:],cmap='gray')
        grid_dia[count].set_xticks([])
        grid_dia[count].set_yticks([])
        if j == 0:
            grid_dia[count].set_title('diff',fontsize=12)
        count = count + 1
        
    plt.waitforbuttonpress()

    for j in range(count):
        grid_sys[j].clear()
        grid_dia[j].clear()

