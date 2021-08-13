# Deep-learning-reconstruction-Radial-SMS-perfusion 

This repository contains code and example test datasets for the paper 'Deep Learning for Radial SMS Myocardial Perfusion Reconstruction using the 3D Residual Booster U-Net'

Instructions:
Running this code requires the installation of Conda and Pip for creating virtual environments and installing packages. Instructions for installing conda and pip can be found in the following link:

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

https://pip.pypa.io/en/stable/installation/

This repository only contains test datasets for visualizing the results of residual booster 3D U-Net paper. Training and validation datasets can be provided upon request.

1. Download the trained networks and test datasets for the gated and ungated residual booster 3D U-Net using the following links.

Gated Network - https://drive.google.com/drive/folders/1muef-xB9ccEA00En_qpQEVFsd76ZlXWa?usp=sharing
Ungated Network - https://drive.google.com/drive/folders/1lYOVBRg0xOsRAfV8P003GpOjEk8Xn3eP?usp=sharing
Gated Testsets - https://drive.google.com/drive/folders/1Vo9FF1IcpgL0krB27qb4aeaeM35M1_5J?usp=sharing
Ungated Testsets - https://drive.google.com/drive/folders/1ZBHQz8mAcuoFD3clYGFvCaqZ76GnQThu?usp=sharing

2. Insert the folders containing the trained networks in the 'trainedNetwork' sub-directory within the gated and ungated directories respectively
3. Insert the folders containing the testing datasets in the 'data' sub-directory within the gated and ungated directories respectively
4. Create a virtual environment using the command 'conda create -n myenv python=3.6.7.'
5. Activate the virtual environment using the command 'source activate myenv'
6. Run 'pip install -r requirements.txt' to install the required packages.
7. Run 'train_network.py in the gated or ungated directories to train the respective networks, traind networks save to the trainedNetwork directory.
8. Run 'visualize.py' to visualize a test dataset, movies of the reconstructions saves to the results file.

The residual booster 3D U-Net was trianed on a Quadro RTX 6000 GPU (~24 GB) on a Linux Fedora 26 operatoring system. Trainiing the network for 100 epochs requires ~12 hours.

https://user-images.githubusercontent.com/35586452/129275208-073007a2-6466-48d0-b807-616cae605c15.mp4


|           |        BU2          |        MoDL         |     CRNN-MRI        |        BU3          |    BU3 (ungated)    |
|:---------:|:-----------:|:-----------:|:------------:|:-----------:|:-----------:|
|   SSIM    |   0.807 ± 0.034     |   0.720 ± 0.036     |   0.935 ± 0.029     |   0.963 ± 0.012     |   0.915 ± 0.028     |
|   PSNR    |   32.084 ± 1.960    |   30.145 ± 1.408    |   38.707 ± 2.850    |   40.238 ± 2.424    |   35.239 ± 2.670    |
|   NRMSE   |   0.375 ± 0.069     |   0.468 ± 0.075     |   0.149 ± 0.052     |   0.147 ± 0.033     |   0.181 ± 0.037     |
|   TIME (s)|         7           |         98          |        110          |         8           |         12          |

Performance comparisons of the residual booster 3D U-Net (BU3), the residual booster 2D U-Net (BU2), MoDL, and CRNN-MRI for the structural similarity index (SSIM), peak signal-to-noise ratio (PSNR), and normalized root mean squared error (NRMSE) averaged over 6 gated (6 ungated if ungated variant) radial simultaneous multi-slice (SMS) test datasets, Mean ± SD. Quality metrics were measured for each time frame and averaged over all test datasets.

CRNN-MRI: Qin C, Schlemper J, Caballero J, Price A, Hajnal J, Rueckert D. Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction. IEEE Transaction on Medical Imaging 2017;38(1):280-90.

MoDL: Aggarwal H, Mani M, Jacob M. MoDL: Model Based Deep Learning Architecture for Inverse Problems. IEEE Transaction on Medical Imaging 2019;38(2):394-405.


Contact: 

Johnathan Le

le.johnv@outlook.com

Ganesh Adluru

gadluru@gmail.com
