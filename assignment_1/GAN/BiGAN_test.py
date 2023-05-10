from random import shuffle
import torch
from torch import nn
from torch.nn import init, Module, ReLU, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d, BatchNorm2d, Tanh, Dropout2d, Dropout
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce
torch.autograd.set_detect_anomaly(True)

'''
Trying this : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
'''

# Initialization parameters
device = 'cuda:0'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath = '/home/adarsh/ADRL/assignment_1/GAN/bi_gan/'

# Latent Length
Z = 128

class Encoder(torch.nn.Module):
    '''
    Maps I: 216x176 -> Z:128
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),
            #torch.nn.Conv2d(8, 1, 3),
        ])
        self.mean = torch.nn.Linear(2048, 128)
        self.std = torch.nn.Linear(2048, 128)
    
    def posteriors(self, mean, std):
        '''
        K-samples from P(Z|X) = Normal(mean(X), std(X))
        '''
        return torch.randn(mean.shape).to(device)*std + mean

    def forward(self, x):
        [ x := conv(x) for conv in self.convs ]
        x = torch.flatten(x, start_dim=1)
        mean = self.mean(x)
        std = torch.exp(self.std(x)*0.5)
        return self.posteriors(mean, std)
    

class Generator(torch.nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.in_decoder = torch.nn.Linear(128, 2048)

        self.convs = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 3, 3, padding= 1),
            torch.nn.Tanh(),          
        ])
    
    def forward(self, x):
        x = self.in_decoder(x)
        x = x.view(-1, 512, 2, 2)
        [x:=conv(x) for conv in self.convs]
        return x

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        filter_size = 64
        # Inference over x
        self.convs = ModuleList([
            Conv2d(3, filter_size, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            Conv2d(filter_size, filter_size * 2, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 2),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            Conv2d(filter_size * 2, filter_size * 4, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 4),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            Conv2d(filter_size * 4, filter_size * 8, 4, 2, 1, bias=False),
            BatchNorm2d(filter_size * 8),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            Conv2d(filter_size * 8, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        ])

        self.convs2 = torch.nn.ModuleList([
            Conv2d(128, 512, 1, stride=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            Conv2d(512, 512, 1, stride=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        ])

        self.convs3 = torch.nn.ModuleList([
            # Conv2d(1024, 1024, 1, stride=1, bias=False),
            # LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.2),
            Conv2d(1024, 1024, 1, stride=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            Conv2d(1024, 1, 1, stride=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        ])

    def forward(self, x, z):
        [ x := conv(x) for conv in self.convs ]
        [ z := conv(z) for conv in self.convs2 ]
        xz = torch.cat((x, z), dim=1)
        [ xz := conv(xz) for conv in self.convs3 ]
        return torch.sigmoid(xz)

# class Discriminator(nn.Module):

#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.dropout = 0.2

#         self.infer_x = nn.Sequential(
#             nn.Conv2d(3, 32, 5, stride=1, bias=True),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(0.2),

#             nn.Conv2d(32, 64, 4, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout),

#             nn.Conv2d(64, 128, 4, stride=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout),

#             nn.Conv2d(128, 256, 4, stride=2, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout),

#             nn.Conv2d(256, 512, 4, stride=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout)
#         )

#         self.infer_z = nn.Sequential(
#             nn.Conv2d(100, 512, 1, stride=1, bias=False),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout),

#             nn.Conv2d(512, 512, 1, stride=1, bias=False),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout)
#         )

#         self.infer_joint = nn.Sequential(
#             nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout),

#             nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=self.dropout)
#         )

#         self.final = nn.Conv2d(1024, 1, 1, stride=1, bias=True)

#     def forward(self, x, z):
#         output_x = self.infer_x(x)
#         output_z = self.infer_z(z)
#         output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
#         output = self.final(output_features)
#         output = F.sigmoid(output)
#         return output.squeeze()#, output_features.view(x.size()[0], -1)

def sample(k):
    return torch.randn((k, Z, 1, 1)).to(device)


def discriminator_epoch_part1(model, x_real, optim):
    '''
        Discriminator Loss:
        maximize f = E[log(D(x_real))]
    '''
    optim.zero_grad()
    y_fake = model[2](x_real)
    loss = torch.nn.functional.binary_cross_entropy(model[0](x_real, y_fake).squeeze(),torch.ones((len(x_real),)).to(device))
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
    loss = torch.nn.functional.binary_cross_entropy(model[0](model[1](z).detach(), z).squeeze(),torch.zeros((len(z),)).to(device))
    loss.backward()
    optim.step()
    # with torch.no_grad():
    #     for param in model[0].parameters():
    #         param.clamp_(-0.1, 0.1)
    return loss


def generator_epoch(model, x_real, z, optim):
    '''
        Generator Loss:
        minimize f = E[log(1-D(G(z)))]
    '''
    optim.zero_grad()
    y_fake = model[2](x_real)
    loss = torch.nn.functional.binary_cross_entropy(model[0](model[1](z), z).squeeze(),torch.ones((len(z),)).to(device)) + torch.nn.functional.binary_cross_entropy(model[0](x_real, y_fake).squeeze(),torch.zeros((len(x_real),)).to(device))
    loss.backward()
    optim.step()

    return loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def softplus(_x):
    return torch.log(1.0 + torch.exp(_x))

if __name__ == '__main__':

    bitmoji_transform = transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.CenterCrop((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    

    model = Sequential(Discriminator(), Generator(), Encoder()).to(device)
    model.apply(weights_init)
    # Load Pretrained Model if avaliable in modelpath
    # try:
    #     model.load_state_dict(torch.load(modelpath))
    # except:
    #     print('inorrect modelpath')

    # Training Code
    i = 0
    batch_size = 128
    
    optim_generator = torch.optim.Adam([{'params' : model[1].parameters()},
                         {'params' : model[2].parameters()}],lr=0.0002,betas=(0.5,0.9))
    optim_discriminator = torch.optim.Adam(model[0].parameters(),lr=0.0002,betas=(0.5,0.999))
    optim_encoder = torch.optim.Adam(model[2].parameters())
    for ep in range(5):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            for X, _ in tepoch:
                # loss1 = discriminator_epoch_part1(model, X.to(device), optim_discriminator).item()
                # loss2 = discriminator_epoch_part2(model, sample(batch_size), optim_discriminator).item()
                # loss3 = generator_epoch(model, X.to(device), sample(batch_size), optim_generator).item()
                
                x = X.to(device)

                z = torch.randn(x.size(0), 100, 1, 1).to(device)

                z_hat, x_tilde = model[2](x), model[1](z)
                data_preds, sample_preds = model[1](x, z_hat), model[1](x_tilde, z)

                real_target = torch.ones((len(data_preds),)).to(device)
                fake_target = torch.zeros((len(sample_preds),)).to(device)

                d_loss = F.binary_cross_entropy(data_preds.squeeze(), real_target) + F.binary_cross_entropy(sample_preds.squeeze(), fake_target)
                g_loss = F.binary_cross_entropy(data_preds.squeeze(), fake_target) + F.binary_cross_entropy(sample_preds.squeeze(), real_target)
                
                if not i%5 == 0: # train discriminator 5 times & generator once
                    optim_discriminator.zero_grad()
                    d_loss.backward()
                    optim_discriminator.step()
                
                else:
                    optim_generator.zero_grad()
                    g_loss.backward()
                    optim_generator.step()

                tepoch.set_postfix({'loss': (d_loss.item(), g_loss.item())})
                tepoch.refresh()
                i += 1
                if i % 500 == 0: 
                    torch.save(model.state_dict(), modelpath+'bi_gan'+str(i)+'.pt')
                