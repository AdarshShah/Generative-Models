from torch import optim
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils.data import DataLoader
from resnet_utils import resnet_train, ResNet, evaluate
import numpy as np
import math
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, channels=3, filter_size=64):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, filter_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_size, filter_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_size * 2, filter_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_size * 4, 1, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        alpha = 0.2
        x = self.layers(x)
        x = x.reshape([x.shape[0], -1]).mean(1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filter_size=64):
        super(Generator, self).__init__()
        conv_dim=64
        self.layers = nn.Sequential(
            nn.Conv2d(1, filter_size, 5, 1, 2, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size, filter_size * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size * 2, filter_size * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size * 4, filter_size * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(filter_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size * 4, filter_size * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(filter_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(filter_size * 4, filter_size * 2, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(filter_size * 2, filter_size , 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size, 1, 5, 1, 2, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.tanh(x)
        return x

out_path = '/home/adarsh/ADRL/assignment_2/Domain_Adaptation/cycle_gan_result'
root = '/home/adarsh/ADRL/datasets/'
device = 'cuda'

m2u_gen = Generator(in_channels=1, out_channels=1)
u2m_gen = Generator(in_channels=1, out_channels=1)
m_disc = Discriminator(channels=1)
u_disc = Discriminator(channels=1)

m2u_gen.to(device)
u2m_gen.to(device)
m_disc.to(device)
u_disc.to(device)

gen_optim = optim.Adam(list(m2u_gen.parameters()) + list(u2m_gen.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=2e-5)
disc_optim = optim.Adam(list(m_disc.parameters()) + list(u_disc.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=2e-5)

common_transforms = transforms.Compose([transforms.Grayscale(), transforms.Resize((32, 32)), transforms.ToTensor()])

dataset = 'mnist'
# dataset = 'clipart'

if dataset == 'mnist':
    usps_train = datasets.USPS(root=root+'usps', train=True, download=True, transform=common_transforms)
    usps_test = datasets.USPS(root=root+'usps', train=False, download=True, transform=common_transforms)
    mnist_train = datasets.MNIST(root=root+'mnist', train=True, download=True, transform=common_transforms)
    mnist_test = datasets.MNIST(root=root+'mnist', train=False, download=True, transform=common_transforms)
    num_classes = 10

if dataset == 'clipart':
    usps_train = datasets.ImageFolder('/home/adarsh/ADRL/datasets/OfficeHomeDataset_10072016/Real_world', transform=common_transforms)
    usps_test = usps_train
    mnist_train = datasets.ImageFolder('/home/adarsh/ADRL/datasets/OfficeHomeDataset_10072016/Clipart', transform=common_transforms)
    mnist_test = mnist_train 
    num_classes = 65

mnist_train_dataloader = DataLoader(mnist_train, batch_size=128, shuffle=True)
usps_train_dataloader = DataLoader(usps_train, batch_size=128, shuffle=True)

mnist_val_dataloader = DataLoader(mnist_test, batch_size=128, shuffle=True)
usps_val_dataloader = DataLoader(usps_test, batch_size=128, shuffle=True)

epochs = 10
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    m2u_gen.train()
    u2m_gen.train()
    m_disc.train()
    u_disc.train()

    with tqdm(enumerate(zip(mnist_train_dataloader, usps_train_dataloader))) as tepoch:
        for i, (x, y) in tepoch:
            x, y = x[0].cuda(), y[0].cuda()
            y_fake, x_fake  = m2u_gen(x), u2m_gen(y)
            x_out, x_fake_out  = m_disc(x), m_disc(x_fake.detach())
            y_out, y_fake_out = u_disc(y), u_disc(y_fake.detach())
            x_ones, x_zeros = torch.ones(x_out.shape).cuda(), torch.zeros(x_fake_out.shape).cuda()
            y_ones, y_zeros = torch.ones(y_out.shape).cuda(), torch.zeros(y_fake_out.shape).cuda()
            loss_from_x = loss_fn(x_out, x_ones) + loss_fn(x_fake_out, x_zeros)
            loss_from_y = loss_fn(y_out, y_ones) + loss_fn(y_fake_out, y_zeros)
            disc_optim.zero_grad()
            d_loss = loss_from_x + loss_from_y
            d_loss.backward()
            disc_optim.step()
            x_fake_out, y_fake_out = m_disc(x_fake), u_disc(y_fake)
            x_ones, y_ones = torch.ones(x_fake_out.shape).cuda(), torch.ones(y_fake_out.shape).cuda()
            gen_loss = loss_fn(x_fake_out, x_ones) + loss_fn(y_fake_out, y_ones)
            consistency_loss = torch.mean((x - u2m_gen(y_fake)).abs()) + torch.mean((y - m2u_gen(x_fake)).abs())
            gen_optim.zero_grad()
            g_loss = gen_loss + consistency_loss
            g_loss.backward()
            gen_optim.step()
            tepoch.set_postfix({'g_loss':g_loss.item(), 'd_loss':d_loss.item()})

mnist_classifier = ResNet(num_classes=num_classes, grayscale=True, size=32).cuda(0)

resnet_train(mnist_classifier, mnist_train, mnist_val_dataloader, len(mnist_test), 5)
print(evaluate(mnist_classifier, usps_train))

mnist_classifier2 = ResNet(num_classes=num_classes, grayscale=True, size=32).cuda(0)

resnet_train(mnist_classifier2, mnist_train, usps_val_dataloader, len(usps_test), 5, cycle_gan=True, gen=m2u_gen)
print(evaluate(mnist_classifier2, usps_train))

