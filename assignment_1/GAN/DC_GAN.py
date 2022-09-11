import torch
from torch.nn import Module, ReLU, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d, BatchNorm2d, Tanh
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torch.nn.utils import spectral_norm
torch.autograd.set_detect_anomaly(True)

'''
This version of DC-GAN is implemented as W-GAN.
Using Spectral Normalization on Convolutions as regularizer to make discriminator 1-Lipchitz

From here : https://proceedings.neurips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf
Tried:
1. Feature Matching 

Trying this : https://arxiv.org/pdf/1710.10196.pdf
'''

# Initialization parameters
device = 'cuda:0'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath = '/home/adarsh/ADRL/assignment_1/GAN/model_01.pt'


class Generator(Module):
    '''
    Take Z:8x8 -> I:128x128 where Z comes from Normal Distribution
    '''

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.layers = ModuleList([
            spectral_norm(ConvTranspose2d(1, 2048, 1, bias=False)),
            spectral_norm(BatchNorm2d(2048)),
            ReLU(),
            spectral_norm(Conv2d(2048, 2048, 1, 1, bias=False)),
            spectral_norm(BatchNorm2d(2048)),
            ReLU(),
            spectral_norm(ConvTranspose2d(2048, 512, 4, 2, 1, bias=False)),
            spectral_norm(BatchNorm2d(512)),
            ReLU(),
            spectral_norm(ConvTranspose2d(512, 256, 4, 2, 1, bias=False)),
            spectral_norm(BatchNorm2d(256)),
            ReLU(),
            spectral_norm(ConvTranspose2d(256, 128, 4, 2, 1, bias=False)),
            spectral_norm(BatchNorm2d(128)),
            ReLU(),
            spectral_norm(Conv2d(128, 64, 1, 1, bias=False)),
            spectral_norm(BatchNorm2d(64)),
            ReLU(),
            spectral_norm(Conv2d(64, 32, 1, 1, bias=False)),
            spectral_norm(BatchNorm2d(32)),
            ReLU(),
            spectral_norm(ConvTranspose2d(32, 1, 4, 2, 1, bias=False)),
            Tanh()
        ])

    def forward(self, x):
        [x := layer(x) for layer in self.layers]
        return x

    def forward(self, x):
        [x := layer(x) for layer in self.layers]
        return x


class Discriminator(Module):
    '''
    Take I:128x128 -> [0,1]
    '''

    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.layers = ModuleList([
            spectral_norm(Conv2d(1, 16, 4, 2, 1, bias=False)),
            LeakyReLU(0.2),

            spectral_norm(Conv2d(16, 8, 4, 2, 1, bias=False)),
            spectral_norm(BatchNorm2d(8)),
            LeakyReLU(0.2),

            spectral_norm(Conv2d(8, 4, 4, 2, 1, bias=False)),
            spectral_norm(BatchNorm2d(4)),
            LeakyReLU(0.2),

            spectral_norm(Conv2d(4, 2, 4, 2, 1, bias=False)),
            spectral_norm(BatchNorm2d(2)),
            LeakyReLU(0.2),

            spectral_norm(Conv2d(2, 1, 4, 2, 1, bias=False)),
            spectral_norm(BatchNorm2d(1)),
            LeakyReLU(0.2),

            spectral_norm(Conv2d(1, 1, 6, 2, 1, bias=False)),
            Sigmoid()
        ])

    def forward(self, x):
        [x := layer(x) for layer in self.layers]
        return x


def sample(k):
    return torch.rand((k, 1, 8, 8)).to(device)


def discriminator_epoch(model, x, z, optim):
    '''
        Discriminator Loss:
        maximize f = E[log(1-D(G(z)))] + E[log(D(x))]
    '''
    loss = (model[0](model[1](z)) - model[0](x)).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    # with torch.no_grad():
    #     for param in model[0].parameters():
    #         param.clamp_(-0.1, 0.1)
    return loss


def generator_epoch(model, X, z, optim):
    '''
        Generator Loss:
        minimize f = E[log(1-D(G(z)))]
    '''
    loss = -model[0](model[1](z)).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


if __name__ == '__main__':

    bitmoji_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)

    train_dataset = Subset(dataset, torch.arange(300))

    model = Sequential(Discriminator(), Generator()).to(device)

    # Load Pretrained Model if avaliable in modelpath
    try:
        model.load_state_dict(torch.load(modelpath))
    except:
        print('incorrect modelpath')

    # Training Code
    i = 0
    batch_size = 32
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    optim_generator = torch.optim.Adam(model[1].parameters(),lr=0.0001,betas=(0.5,0.999))
    optim_discriminator = torch.optim.Adam(model[0].parameters(),lr=0.0001,betas=(0.5,0.999))
    for ep in range(1000):
        try:
            with tqdm(train_dataloader) as tepoch:
                tepoch.set_description(f'Epoch {ep+1} : ')
                for X, _ in tepoch:
                    loss = discriminator_epoch(model, X.to(device), sample(batch_size), optim_discriminator).item(
                    ), generator_epoch(model, X.to(device), sample(batch_size), optim_generator).item()
                    for _ in range(4):
                        generator_epoch(model, X.to(device), sample(batch_size), optim_generator)
                    tepoch.set_postfix({'loss': loss})
                    tepoch.refresh()
                    i += 1
                    if i % 10 == 0:
                        torch.save(model.state_dict(), modelpath)
        except:
            pass
        
