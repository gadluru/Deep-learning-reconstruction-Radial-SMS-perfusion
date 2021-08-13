import sys
sys.path.append('../utils')

import os.path
import numpy as np
import glob
import random
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io
from  supportingFunctions import *
from torch.utils.data import Dataset

def load_ungated_sms(N=32, split='train', test_mode=None, val_mode=None):

# -------------------------------------------------------------------------------------------
#
#     inputs,targets = load_sms_data(N=32, split='train', test_mode=None, val_mode=None)
#                      - loads and prepares SMS datasets for training and testing the 
#                        residual booster network
#                      - helper function for loadData()
#                      - SMS datasets have dimensions [sx,sy,nt,nsets,nsl]
#                            - sx: spatial dimension
#                            - sy: spaitla dimension
#                            - nt: number of time frames
#                            - nsets: number of slice groups
#                            - nsl: number of simultaneously exicted slices
# -------------------------------------------------------------------------------------------
#    
#     inputs (Ungated 2D radial SMS myocardial perfusion datasets)
#
#        -N (integer: 8|16|32): number of training time frames for each batch [default: 32]
#        -split (string: 'train|val|test'): load selected datasets [default:'train']
#        -test_mode (integer: 0|1|2|3|4|5): selects test set to load,
#                                           loads all test datasets if not specified [default: None]
#        -val_mode (integer: 0|1): selects validation set to load, 
#                                  loads all validation datasets if not specified [default: None]
#
# -------------------------------------------------------------------------------------------
#     outputs
#        - inputs [nb,ch,sx,sy,nt]: inputs to feed into the residual booster network 
#        - targets [nb,ch,sx,sy,nt]: PT-STCR references for training the residual booster network
#              -nb: batch dimension
#              -ch: real and imaginary components (2)
#              -sx: spatial dimension
#              -sy: spatial dimension 
#              -nt: number of time frames chosen by N 
#
#

    if split == 'test':
        datasets = glob.glob('data/testsets/*.mat')

        if test_mode is not None:
            datasets = [datasets[test_mode]]

    if split == 'val':
        datasets = glob.glob('data/valsets/*.mat')

        if val_mode is not None:
            datasets = [datasets[test_mode]]
    
    if split == 'train':
        datasets = glob.glob('data/trainsets/*.mat')

    inputs = []
    targets = []
    for i in range(len(datasets)):
        print(i)

        f = scipy.io.loadmat(datasets[i],variable_names=['Image_init_sys','Image_init_dia','Image_sys','Image_dia'])

        sx,sy,nfr,nsets,nsl = f['Image_sys'].shape

        sys_init = f['Image_init_sys']
        sys = f['Image_sys']
        dia_init = f['Image_init_dia']
        dia = f['Image_dia']

        ll = N*(nfr//N)

        sys_init = sys_init[:,:,:ll,:,:]
        sys = sys[:,:,:ll,:,:]
        dia_init = dia_init[:,:,:ll,:,:]
        dia = dia[:,:,:ll,:,:]

        sys_init = np.transpose(sys_init,[0,1,4,3,2])
        sys = np.transpose(sys,[0,1,4,3,2])
        dia_init = np.transpose(dia_init,[0,1,4,3,2])
        dia = np.transpose(dia,[0,1,4,3,2])

        sx,sy,nsl,nsets,nfr = sys.shape
        
        sys_init = np.reshape(sys_init,(sx,sy,nsl*nsets*nfr))
        sys = np.reshape(sys,(sx,sy,nsl*nsets*nfr))
        dia_init = np.reshape(dia_init,(sx,sy,nsl*nsets*nfr))
        dia = np.reshape(dia,(sx,sy,nsl*nsets*nfr))

        sx,sy,nfr = sys.shape

        sys_init = np.reshape(sys_init,(sx,sy,nfr//N,N))
        sys = np.reshape(sys,(sx,sy,nfr//N,N))
        dia_init = np.reshape(dia_init,(sx,sy,nfr//N,N))
        dia = np.reshape(dia,(sx,sy,nfr//N,N))

        inputs.append(sys_init)
        inputs.append(dia_init)
        targets.append(sys)
        targets.append(dia)

    inputs = np.concatenate(inputs,2)
    targets = np.concatenate(targets,2)
    
    inputs = np.transpose(inputs,(2,0,1,3))
    targets = np.transpose(targets,(2,0,1,3))

    inputs = c2r(inputs,dim='first')
    targets = c2r(targets,dim='first')
        
    return inputs,targets

class loadData(Dataset):
    def __init__(self,N=32, split='train', test_mode=None, val_mode=None, do_transform=False, directory=None):
        
# -------------------------------------------------------------------------------------------
#
#     inputs,targets = loadData(N=32, split='train', test_mode=None, val_mode=None)
#                      - loads and prepares SMS datasets for training and testing the 
#                        residual booster network
#                      - SMS datasets have dimensions [sx,sy,nt,nsets,nsl]
#                            - sx: spatial dimension
#                            - sy: spaitla dimension
#                            - nt: number of time frames
#                            - nsets: number of slice groups
#                            - nsl: number of simultaneously exicted slices
# -------------------------------------------------------------------------------------------
#    
#     inputs (2D radial SMS myocardial perfusion datasets)
#
#        -N (integer: 8|16|32): number of training time frames for each batch [default: 32]
#        -split (string: 'train|val|test'): load selected datasets [default:'train']
#        -test_mode (integer: 0,1,2,3,4,5): selects test set to load,
#                                           loads all test datasets if not specified [default: None]
#        -val_mode (integer: 0,1): selects validation set to load, 
#                                  loads all validation datasets if not specified [default: None]
#        -do_transform (boolean: True|False): performs data augmentations [default: True]
#        -directory (string): save location for network and other parameters [default: None]
# -------------------------------------------------------------------------------------------
#     outputs
#        - inputs [nb,ch,sx,sy,nt]: inputs to feed into the residual booster network 
#        - targets [nb,ch,sx,sy,nt]: PT-STCR references for training the residual booster network
#              -nb: batch dimension
#              -ch: real and imaginary components (2)
#              -sx: spatial dimension
#              -sy: spatial dimension 
#              -nt: number of time frames chosen by N 
#
#

        inputs,targets = load_ungated_sms(split=split, N=N, test_mode=test_mode, val_mode=val_mode)

        self.do_transform = do_transform
        self.inputs = inputs
        self.targets = targets

        if split == 'train':
            if os.path.isfile(directory+'scale_factors.pkl'):

                with open(directory+'scale_factors.pkl','rb') as f:
                    self.mean,self.std = pickle.load(f)

            else:
                self.mean = np.mean(self.inputs)
                self.std = np.std(self.inputs)

                with open(directory+'scale_factors.pkl','wb') as f:
                    pickle.dump([self.mean,self.std],f)

            self.inputs = (self.inputs - self.mean)/self.std
            self.targets = (self.targets  - self.mean)/self.std

        else:
            if os.path.isfile(directory+'scale_factors.pkl'):
                
                with open(directory+'scale_factors.pkl','rb') as f:
                    self.mean,self.std = pickle.load(f)
            else:
                print('scale file does not exist. run script in training mode first.')
                sys.exit()

            self.inputs = (self.inputs - self.mean)/self.std
            self.targets = (self.targets - self.mean)/self.std

    def __getitem__(self,index):
        if self.do_transform:
            if random.uniform(0,1) < 1:
                inputs,targets = toPIL(self.inputs[index],self.targets[index])
                if random.uniform(0,1) < 0.5:
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = horizontal_flip(inputs,targets)
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = vertical_flip(inputs,targets)
                elif random.uniform(0,1) < 0.5:
                    inputs,targets = shear(inputs,targets)
                else:
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = translate(inputs,targets)
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = rotate(inputs,targets)
                inputs,targets = toTensor(inputs,targets)            
            else:
                inputs = self.inputs[index]
                targets = self.targets[index]
        else:
            inputs = self.inputs[index]
            targets = self.targets[index]

        return inputs,targets

    def __len__(self):
        return len(self.targets)
        
if __name__ == '__main__':
    loadData(split='test')
