import torch
from torch.nn import Module, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# Initialization parameters
device = 'cuda:0'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath = '/home/adarsh/ADRL/assignment_1/GAN/model_01.pt' 

class Generator(Module):
    '''
    Take Z:4x4 -> I:128x128 where Z comes from Normal Distribution
    '''
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.layers = ModuleList([
            ConvTranspose2d(1, 32, 2, 2),
            LeakyReLU(),
            ConvTranspose2d(32, 64, 4),
            ConvTranspose2d(64, 128, 3),
            ConvTranspose2d(128, 64, 4),
            LeakyReLU(),
            ConvTranspose2d(64, 32, 2, 2),
            ConvTranspose2d(32, 16, 2, 2),
            LeakyReLU(),
            ConvTranspose2d(16, 1, 2, 2),
            Sigmoid()
        ])
    
    def forward(self, x):
        [ x:= layer(x) for layer in self.layers]
        return x


class Discriminator(Module):
    '''
    Take I:128x128 -> [0,1]
    '''
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.layers = ModuleList([
            Conv2d(1, 16, 2),
            LeakyReLU(),
            Conv2d(16, 64, 2),
            MaxPool2d(2),
            Conv2d(64, 64, 2),
            LeakyReLU(),
            Conv2d(64, 64, 2),
            MaxPool2d(2),
            Conv2d(64, 1, 2),
            LeakyReLU(),
            Conv2d(1, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(196, 1),
            Sigmoid(),
        ])
    
    def forward(self, x):
        [ x:= layer(x) for layer in self.layers]
        return x

def sample(k):
    return torch.randn((k,4,4)).to(device)

def discriminator_epoch(model, x, z, optim):
    '''
        Discriminator Loss:
        minimize f = E[log(D(G(z)))] - E[log(D(x))]
    '''
    loss = torch.log(model[0](model[1](z))) - torch.log(model[0](x))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def generator_epoch(model, z, optim):
    '''
        Discriminator Loss:
        maximize f = E[log(D(G(z)))]
    '''
    loss = -1*torch.log(model[0](model[1](z)))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


if __name__=='__main__':
    
    bitmoji_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)

    train_dataset = Subset(dataset, torch.arange(300))
    train_dataloader = DataLoader(dataset, batch_size=1)

    model = Sequential(Discriminator(), Generator()).to(device)

    # Load Pretrained Model if avaliable in modelpath
    # try:
    #     model.load_state_dict(torch.load(modelpath))
    # except:
    #     print('incorrect modelpath')

    # Training Code
    optim_discriminator = torch.optim.Adam(model[0].parameters(), lr=0.0001)
    optim_generator = torch.optim.Adam(model[1].parameters(), lr=0.01)
    for ep in range(1):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            for X, _ in tepoch:
                loss = discriminator_epoch(model, X.to(device), sample(1), optim_discriminator)
                for _ in range(15):
                    generator_epoch(model, sample(1), optim_generator)
                tepoch.set_postfix({'loss': loss.item()})
                tepoch.refresh()
    torch.save(model.state_dict(), modelpath)


