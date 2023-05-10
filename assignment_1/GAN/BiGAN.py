from random import shuffle
import torch
from torch.nn import init, Module, ReLU, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d, BatchNorm2d, Tanh, Dropout2d, Dropout
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

'''
Trying this : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
'''

# Initialization parameters
device = 'cuda:0'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath = '/home/adarsh/ADRL/assignment_1/GAN/bi_gan/'

# Latent Length
Z = 100


class Encoder(torch.nn.Module):
    '''
    Maps I: 216x176 -> Z:128
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        filter_size = 64

        self.convs = torch.nn.ModuleList([
            Conv2d(3, filter_size, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size), 
            ReLU(True),

            Conv2d(filter_size, filter_size * 2, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 2), 
            ReLU(True),

            Conv2d(filter_size * 2, filter_size * 4, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 4), 
            ReLU(True),

            Conv2d(filter_size * 4, filter_size * 8, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 8), 
            ReLU(True),

            Conv2d(filter_size * 8, filter_size * 8, 4, 1, 0, bias=False), 
            BatchNorm2d(filter_size * 8), 
            ReLU(True),

            Conv2d(filter_size * 8, 100, 1, 1)
        ])

    def forward(self, x):
        [ x := conv(x) for conv in self.convs ]
        return x

class Generator(torch.nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()

        filter_size = 64
        self.convs = torch.nn.ModuleList([
            ConvTranspose2d(100, filter_size * 8, 4, 1, 0, bias=False), 
            BatchNorm2d(filter_size* 8), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size * 8, filter_size * 4, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 4), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size * 4, filter_size * 2, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 2), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size * 2, filter_size, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size, 3, 4, 2, 1, bias=False), 
            Tanh()
        ])           
    
    def forward(self, x):
        [x:=conv(x) for conv in self.convs]
        return x

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        filter_size = 64
        # Inference over x
        self.convs = torch.nn.ModuleList([
            Conv2d(3, filter_size, 4, 2, 1), 
            LeakyReLU(0.2),

            Conv2d(filter_size, filter_size * 2, 4, 2, 1), 
            BatchNorm2d(filter_size * 2),
            LeakyReLU(0.2),

            Conv2d(filter_size * 2, filter_size * 4, 4, 2, 1), 
            BatchNorm2d(filter_size * 4),
            LeakyReLU(0.2),

            Conv2d(filter_size * 4, filter_size * 8, 4, 2, 1), 
            BatchNorm2d(filter_size * 8),
            LeakyReLU(0.2),

            Conv2d(filter_size* 8, filter_size * 8, 4, 1, 0), 
            LeakyReLU(0.2)
        ])

        self.convs2 = torch.nn.ModuleList([
            Conv2d(100, 512, 1, 1, 0), 
            LeakyReLU(0.2),

            Conv2d(512, 512, 1, 1, 0), 
            LeakyReLU(0.2)
        ])
        
        self.convs3 = torch.nn.ModuleList([
            Conv2d(1024, 2048, 1, 1, 0), 
            LeakyReLU(0.2),

            Conv2d(2048, 2048, 1, 1, 0), 
            LeakyReLU(0.2),

            Conv2d(2048, 1, 1, 1, 0),
            torch.nn.Sigmoid()
        ])

    def forward(self, x, z):

        [ x := conv(x) for conv in self.convs ]
        [ z := conv(z) for conv in self.convs2 ]
        xz = torch.cat((x, z), dim=1)
        [ xz := conv(xz) for conv in self.convs3 ]
        return xz

def sample(k):
    return torch.randn((k, Z, 1, 1)).to(device)

if __name__ == '__main__':

    bitmoji_transform = transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.CenterCrop((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)

    train_dataset = Subset(dataset, torch.arange(300))

    model = Sequential(Discriminator(), Generator(), Encoder()).to(device)

    print(model[2])
    summary(model[2], (128, 3, 64, 64))
    # i = 0
    # batch_size = 128
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optim_generator = torch.optim.Adam([{'params' : model[1].parameters()},
    #                      {'params' : model[2].parameters()}],lr=0.0002,betas=(0.5,0.999))
    # optim_discriminator = torch.optim.Adam(model[0].parameters(),lr=0.0002,betas=(0.5,0.999))

    # for ep in range(15):
    #     with tqdm(train_dataloader) as tepoch:
    #         tepoch.set_description(f'Epoch {ep+1} : ')
    #         for X, _ in tepoch:
    #             x = X.to(device)
    #             z = sample(batch_size)

    #             z_hat, x_hat = model[2](x), model[1](z)
    #             real_preds, fake_preds = model[0](x, z_hat), model[0](x_hat, z)

    #             d_loss = F.binary_cross_entropy(real_preds.squeeze(), torch.ones((len(real_preds),)).to(device)) + F.binary_cross_entropy(fake_preds.squeeze(), torch.zeros((len(fake_preds),)).to(device))
    #             g_loss = F.binary_cross_entropy(real_preds.squeeze(), torch.zeros((len(real_preds),)).to(device)) + F.binary_cross_entropy(fake_preds.squeeze(), torch.ones((len(fake_preds),)).to(device))
                
    #             if not i%6 == 0: # train discriminator 5 times & generator once
    #                 optim_discriminator.zero_grad()
    #                 d_loss.backward()
    #                 optim_discriminator.step()
                
    #             else:
    #                 optim_generator.zero_grad()
    #                 g_loss.backward()
    #                 optim_generator.step()

    #             tepoch.set_postfix({'loss': (d_loss.item(), g_loss.item())})
    #             tepoch.refresh()
    #             i += 1
    #             if i % 500 == 0: 
    #                 torch.save(model.state_dict(), modelpath+'bi_gan'+str(i)+'.pt')
                