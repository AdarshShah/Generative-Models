import random
from random import shuffle
import torch
from torch.nn import init, Module, ReLU, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d, BatchNorm2d, Tanh
from torchvision import datasets, transforms, utils
from torch.utils.data import Subset, DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from LS_GAN import sample, Discriminator, Generator

device = 'cuda:0'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath = '/home/adarsh/ADRL/assignment_1/GAN/ls_gan/'
fake_path = '/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/bitmoji_fake_ls_gan/'
real_path = '/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/bitmoji_real/'

bitmoji_transform = transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.CenterCrop((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)
dataloader = DataLoader(dataset, batch_size=130, shuffle=True)

model = Sequential(Discriminator(), Generator()).to(device)

fid = {}

for f in os.listdir(modelpath):

    # load model
    model.load_state_dict(torch.load(os.path.join(modelpath, f)))
    
    # clear folder of fake images
    for _f in os.listdir(fake_path):
        os.remove(os.path.join(fake_path, _f))
    
    ## generate fake images using current generator
    noise = sample(1000)
    with torch.no_grad():
        fake_img = model[1](noise).detach_().cpu()
    for j in range(1000):
        utils.save_image(fake_img[j]*0.5 + 0.5, '%simg%d.png' % (fake_path, j))

    #calculating FID using the following command (third party)
    command = 'python -m pytorch_fid "' + real_path + '" "' + fake_path + '" --device cuda:0'
    res = subprocess.getstatusoutput(command)

    #append the FID to the FID list
    fid_score = float(res[1][res[1].rfind(' '):])
    fid[int(f.split('gan')[1].split('.pt')[0])] = fid_score

    print("%s: %f"%(f, fid_score))

list1 = []
for x in sorted(list(fid.keys())):
    list1.append(fid[x])

plt.figure(figsize=(8,5))
plt.title("FID Score Plot LS-GAN")

plt.plot(list1)
plt.xlabel("Iterations % 500")
plt.ylabel("FIDs")
plt.savefig('/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/ls_fid.png')
