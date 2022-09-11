import torch
from torch.nn import AvgPool2d, Module, ReLU, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d, BatchNorm2d, Tanh
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torch.nn.utils import spectral_norm

class GeneratorBlocks(Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(GeneratorBlocks, self).__init__()
        self.conv1 = spectral_norm(ConvTranspose2d(in_channels, out_channels, 2, 2))
        self.conv2 = spectral_norm(Conv2d(out_channels, out_channels, 3, padding=1))
        self.conv3 = spectral_norm(Conv2d(out_channels, out_channels, 3, padding=1))
        
    
    def forward(self, input):
        h = self.conv1(input)
        h = torch.nn.functional.leaky_relu(self.conv2(h), 0.2, True)
        h = torch.nn.functional.leaky_relu(self.conv3(h), 0.2, True)
        return h

class Generator(Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv1 = ConvTranspose2d(512, 512, 4, 4)
        self.conv2 = Conv2d(512, 512, 3, padding=1)
        self.conv3 = Conv2d(512, 512, 3, padding=1)
        self.blocks = ModuleList()
        self.rgb = Conv2d(512, 3, 3, padding=1)
    
    def forward(self, input):
        h = self.conv1(input)
        h = torch.nn.functional.leaky_relu(self.conv2(h), 0.2, True)
        h = torch.nn.functional.leaky_relu(self.conv3(h), 0.2, True)
        h = torch.nn.functional.tanh(self.rgb(h))
        return h

class DiscriminatorBlock(Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.pooling = AvgPool2d(2, 2)
    
    def forward(self, input):
        h = torch.nn.functional.leaky_relu(self.conv1(input), 0.2, True)
        h = torch.nn.functional.leaky_relu(self.conv2(h), 0.2, True)
        return self.pooling(h)

class Discriminator(Module):

    def __init__(self) -> None:
        super(Discriminator, self).__init__()
    
    def forward(self, input):
        return input

