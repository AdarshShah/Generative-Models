'''
The following code is taken from this blog -> 'https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html'

We have modified the model and tuned the training parameters according to our dataset.
'''

import os
import json
import math
import numpy as np 
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

bitmoji_transform = transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ])

datapath = '/data2/home/sasankasahu/VAE/data/bitmojis1'
dataset = ImageFolder(root=datapath, transform=bitmoji_transform)

train_dataset = data.Subset(dataset, torch.arange(60000))
test_dataset = data.Subset(dataset, torch.arange(60000,70000))

train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=2)
test_loader  = data.DataLoader(test_dataset,  batch_size=256, shuffle=False, drop_last=False, num_workers=2)

device = 'cuda'

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

class CNNModel(nn.Module):

    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2
        c_hid4 = hidden_features*4

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid4, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid4, c_hid4, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid4*4, c_hid3),
            Swish(),
            nn.Linear(c_hid3, out_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x

class Sampler:

    def __init__(self, model, img_shape, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=100, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=100, step_size=10, return_img_per_step=False):
        """
        Function for sampling images for a given model. 
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input. 
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True
        
        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        
        # List for storing generations at each step (for later analysis)
        imgs_per_step = []
        
        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)
            
            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)
            
            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())
        
        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(True)
        
        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(True)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs

class DeepEnergyModel(nn.Module):
    
    def __init__(self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0):
        super().__init__()
        
        self.img_shape = img_shape
        self.cnn = CNNModel()
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)
        self.alpha = alpha
 
    def forward(self, x):
        z = self.cnn(x)
        return z

model = DeepEnergyModel(img_shape=(3,32,32), 
                    batch_size=train_loader.batch_size,
                    lr=1e-4,
                    beta1=0.0)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        
def training_step(batch):

    # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
    real_imgs, _ = batch
    small_noise = torch.randn_like(real_imgs) * 0.005
    real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)
    
    # Obtain samples
    fake_imgs = model.sampler.sample_new_exmps(steps=120, step_size=10)
    
    real_imgs = real_imgs.to(device)
    fake_imgs = fake_imgs.to(device)
    
    # Predict energy score for all images
    inp_imgs = torch.cat((real_imgs, fake_imgs), dim=0)
    inp_imgs.to(device)
    real_out, fake_out = model.cnn(inp_imgs).chunk(2, dim=0)
    
    # Calculate losses
    reg_loss = model.alpha * (real_out ** 2 + fake_out ** 2).mean()
    cdiv_loss = fake_out.mean() - real_out.mean()
    loss = reg_loss + cdiv_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss

def generate_imgs(model):
    model.eval()
    start_imgs = torch.rand((10,) + model.img_shape).to(device)
    start_imgs = start_imgs * 2 - 1
    torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
    imgs_per_step = Sampler.generate_samples(model.cnn, start_imgs, steps=288, step_size=10, return_img_per_step=True)
    torch.set_grad_enabled(False)
    model.train(True)
    return imgs_per_step

def validate(epoch):
  imgs_per_step = generate_imgs(model)
  imgs_per_step = imgs_per_step.cpu()

  torch.transpose(imgs_per_step, 0, 1)
  idx = [0, 31, 63, 95, 127, 159, 191, 223, 255, 287]
  x = imgs_per_step[idx]
  torch.transpose(x, 0, 1)
  x = x.reshape(-1, 3, 32, 32)

  save_image(x*0.5+0.5, f'/data2/home/sasankasahu/ADRL/ebm' +str(epoch) + 'i' +'.png', nrow=10)

epochs = 50
for ep in range(epochs):
    for batch in train_loader:
        training_step(batch)

validate(epoch)

