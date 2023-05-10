from random import shuffle
import torch
from torch.nn import init, Module, ReLU, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d, BatchNorm2d, Tanh
from torchvision import datasets, transforms, utils
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import os
import subprocess
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchinfo import summary
torch.autograd.set_detect_anomaly(True)

from loss_plot import LossPlot

# Initialization parameters
device = 'cuda:0'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath = '/home/adarsh/ADRL/assignment_1/GAN/dc_gan/'
fake_path = '/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/bitmoji_fake_dc_gan/'
real_path = '/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/bitmoji_real/'
writer = SummaryWriter('/home/adarsh/ADRL/assignment_1/GAN/logs/dcgan/')
global_step = 0
# Latent Length
Z = 100


loss_plot = LossPlot(epochs=8,
                        batch_size=128,
                        data_len= 130227,
                        plot_interval=50,
                        dir=modelpath)

class Generator(Module):
    '''
    Take Z:100x1x1 -> I:filter_sizexfilter_size where Z comes from Normal Distribution
    '''

    def __init__(self) -> None:
        super(Generator, self).__init__()
        filter_size = 64
        self.layers = ModuleList([
            ConvTranspose2d( Z, filter_size * 8, 4, 1, 0, bias=False),
            BatchNorm2d(filter_size * 8),
            ReLU(True),
            ConvTranspose2d(filter_size * 8, filter_size * 4, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 4),
            ReLU(True),
            ConvTranspose2d( filter_size * 4, filter_size * 2, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 2),
            ReLU(True),
            ConvTranspose2d( filter_size * 2, filter_size, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size),
            ReLU(True),
            ConvTranspose2d( filter_size, 3, 4, 2, 1, bias=False),
            Tanh()
        ])

    def forward(self, x):
        [x := layer(x) for layer in self.layers]
        return x

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0)


class Discriminator(Module):
    '''
    Take I:filter_sizexfilter_size -> [0,1]
    '''

    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        filter_size = 64
        self.layers = ModuleList([
            Conv2d(3, filter_size, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv2d(filter_size, filter_size * 2, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 2),
            LeakyReLU(0.2, inplace=True),
            Conv2d(filter_size * 2, filter_size * 4, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 4),
            LeakyReLU(0.2, inplace=True),
            Conv2d(filter_size * 4, filter_size * 8, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 8),
            LeakyReLU(0.2, inplace=True),
            Conv2d(filter_size * 8, 1, 4, 1, 0, bias=False)
        ])

    def forward(self, x):
        [x := layer(x) for layer in self.layers]
        return torch.sigmoid(x)
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0)


def sample(k):
    return torch.randn((k, Z, 1, 1)).to(device)


def discriminator_epoch_part1(model, x_real, optim):
    '''
        Discriminator Loss:
        maximize f = E[log(D(x_real))]
    '''
    optim.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy(model[0](x_real).squeeze(),torch.ones((len(x_real),)).to(device))
    loss.backward()
    optim.step()
    # with torch.no_grad():
    #     for param in model[0].parameters():
    #         param.clamp_(-0.1, 0.1)
    return loss

def discriminator_epoch_part2(model, z, optim):
    '''
        Discriminator Loss:
        maximize f = E[log(D(x_real))]
    '''
    optim.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy(model[0](model[1](z).detach()).squeeze(),torch.zeros((len(z),)).to(device))
    loss.backward()
    optim.step()
    # with torch.no_grad():
    #     for param in model[0].parameters():
    #         param.clamp_(-0.1, 0.1)
    return loss


def generator_epoch(model, z, optim):
    '''
        Generator Loss:
        minimize f = E[log(1-D(G(z)))]
    '''
    optim.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy(model[0](model[1](z)).squeeze(),torch.ones((len(z),)).to(device))
    loss.backward()
    optim.step()
    return loss


if __name__ == '__main__':

    bitmoji_transform = transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.4, 0.4, 0.4)),
                           ])
    dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)

    train_dataset = Subset(dataset, torch.arange(300))

    model = Sequential(Discriminator(), Generator()).to(device)

    # Load Pretrained Model if avaliable in modelpath
    try:
        model.load_state_dict(torch.load(modelpath))
    except:
        print('inorrect modelpath')

    fid = []
    # Training Code
    count = 0
    i = 0
    batch_size = 128
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim_generator = torch.optim.Adam(model[1].parameters(),lr=0.0002,betas=(0.5,0.999))
    optim_discriminator = torch.optim.Adam(model[0].parameters(),lr=0.0002,betas=(0.5,0.999))
    for ep in range(8):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            for X, _ in tepoch:
                count += X.shape[0]
                loss1 = discriminator_epoch_part1(model, X.to(device), optim_discriminator).item()
                loss2 = discriminator_epoch_part2(model, sample(batch_size), optim_discriminator).item()
                loss3 = generator_epoch(model, sample(batch_size), optim_generator).item()
                tepoch.set_postfix({'loss': (loss1, loss2, loss3)})
                tepoch.refresh()
                writer.add_scalars('DC-GAN Training',{
                    'discriminator loss' : loss1+loss2,
                    'generator loss' : loss3
                }, global_step=global_step)
                loss_plot.add_item(loss1+loss2)
                loss_plot.add_test_item(loss3)
                global_step += 1
                i += 1
                if i % 500 == 0:
                    torch.save(model.state_dict(), modelpath+'my_dc_gan'+str(i)+'.pt')
                    z = torch.randn(100, 100, 1, 1).to(device)
                    z = model[1](z).cpu()
                    save_image(z*0.5+0.5,
                            f'/home/adarsh/ADRL/assignment_1/GAN/dc_gan/dc_{str(i)}.png', nrow=10)