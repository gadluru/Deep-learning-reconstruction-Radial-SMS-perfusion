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
#                 -gif (boolean True|False): choose whether to generate movie or not, 
#                                            saves to results directory [default: 1 (True)]
#
# ------------------------------------------------------------------------------------------------------



# Modify this line with location of trained network
directory = 'trainedNetwork/12Jul_0749pm/'

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

nb,ch,sx,sy,nt = targets.shape

inputs = inputs*std + mean
outputs = outputs*std + mean
targets = targets*std + mean

inputs = np.abs(r2c(inputs,dim='first'))
outputs = np.abs(r2c(outputs,dim='first'))
targets = np.abs(r2c(targets,dim='first'))

inputs = np.transpose(inputs,(0,3,1,2))
outputs = np.transpose(outputs,(0,3,1,2))
targets = np.transpose(targets,(0,3,1,2))

inputs = np.reshape(inputs,(nb*nt,sx,sy))
outputs = np.reshape(outputs,(nb*nt,sx,sy))
targets = np.reshape(targets,(nb*nt,sx,sy))

nt,sx,sy = targets.shape
nsl = 3

inputs = np.reshape(inputs,(nsl,nt//nsl,sx,sy))
outputs = np.reshape(outputs,(nsl,nt//nsl,sx,sy))
targets = np.reshape(targets,(nsl,nt//nsl,sx,sy))

nsl,nfr,sx,sy = targets.shape
clip = (nfr//3)*args.nset

inputs = inputs[:,clip:clip+60,:,:]
outputs = outputs[:,clip:clip+60,:,:]
targets = targets[:,clip:clip+60,:,:]

orientation = int(check_orientation(targets))

inputs = orientate(inputs,orientation)
outputs = orientate(outputs,orientation)
targets = orientate(targets,orientation)

inputs = normalize(inputs)
outputs = normalize(outputs)
targets = normalize(targets)

inputs = brighten(inputs,0.4)
outputs = brighten(outputs,0.4)
targets = brighten(targets,0.4)

diff = np.abs(targets - outputs)

nsl,nt,sy,sx = targets.shape

print('generating movie...')
print('saving movie to' + savelocation)
if args.gif:
    mMaker(inputs,outputs,targets,savelocation,'results.gif')

fig = plt.figure()
grid = ImageGrid(fig,111,nrows_ncols=(3,4),axes_pad=0,direction='column')
plt.ion()
for i in range(nt):
    labels = ['slice 1','slice 2','slice 3']
    input_recon = inputs[:,i,:,:]
    output_recon = outputs[:,i,:,:]
    truth_recon = targets[:,i,:,:]
    diff_recon = diff[:,i,:,:]
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
        grid[count].imshow(truth_recon[j,:,:],cmap='gray')
        grid[count].set_xticks([])
        grid[count].set_yticks([])
        if j == 0:
            grid[count].set_title('target',fontsize=12)
        count = count + 1

    for j in range(nsl):
        grid[count].imshow(diff_recon[j,:,:],cmap='gray')
        grid[count].set_xticks([])
        grid[count].set_yticks([])
        if j == 0:
            grid[count].set_title('diff',fontsize=12)
        count = count + 1
        
    plt.waitforbuttonpress()

    for j in range(count):
        grid[j].clear()

    count = 0

