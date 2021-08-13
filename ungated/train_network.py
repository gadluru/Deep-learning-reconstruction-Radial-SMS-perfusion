import time
import pickle
import os.path
import scipy.io
from residual_booster_network import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#number of frames to train on
N = 32
#batch size
batch = 3
#number of input features
in_features = 2
#number of output features
out_features = 2
#filter width
resolution = 64
#number of layers
layers = 3
#drop out
dp = True
#perceptual loss weight
alpha = 0.04
#mean absolute error weight
beta = 1
#learning rate
lr = 0.0003
#number of training epochs
nepoch = 100
#interval at which to save checkpoint networks
save_interval = 20
#save directory
directory = 'trainedNetwork/'+datetime.now().strftime("%d%b_%I%M%P") + '/'
#name of final network to save
save_directory = 'residual_booster_network'
#pre-train network with gated variant
pretrain = '../gated/trainedNetwork/12Jul_0749pm/residual_booster_network.h5'

if not os.path.exists(directory):
    os.makedirs(directory)

parameters = {'N':N, 'batch':batch, 'in_features':in_features,'out_features':out_features,'resolution':resolution, 'layers':layers,'dp':dp,'save_directory':save_directory}

with open(directory+'parameters.pkl','wb') as f:
    pickle.dump(parameters,f)

residual_booster_network(N,batch,in_features,out_features,resolution,layers,dp,alpha,beta,lr,nepoch,save_interval,directory,save_directory,pretrain,device)
