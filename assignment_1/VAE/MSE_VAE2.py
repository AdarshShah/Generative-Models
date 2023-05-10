import numpy as np
import torch
import matplotlib as plt
import sklearn
from torch.utils.data import DataLoader, Subset
from torchvision.io import read_image
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

#Initialization parameters
device = 'cuda:0'
path = '/home/adarsh/ADRL/datasets/img_align_celeba'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath2 = '/home/adarsh/ADRL/assignment_1/VAE/model_14.pt'
modelpath = '/home/adarsh/ADRL/assignment_1/VAE/model_bitmoji.pt'
# latentpath = '/home/adarsh/ADRL/assignment_1/VAE/Z_03.pkl'

#Utility Scripts
class Encoder(torch.nn.Module):
    '''
    Maps I: 216x176 -> Z:1x23x18
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 32, 3, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, 3, 2, 1),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 512, 3, 2, 1),
            torch.nn.LeakyReLU(),
            #torch.nn.Conv2d(8, 1, 3),
        ])
        self.mean = torch.nn.Linear(8192, 128)
        self.std = torch.nn.Linear(8192, 128)
    
    def forward(self, x):
        [ x := conv(x) for conv in self.convs ]
        x = torch.flatten(x, start_dim=1)
        mean = self.mean(x)
        std = torch.exp(self.std(x)*0.5)
        return mean, std
        

def posteriors(mean, std, k):
    '''
    K-samples from P(Z|X) = Normal(mean(X), std(X))
    '''
    return torch.randn(mean.shape).to(device)*std + mean
    # return torch.stack([mean + std*torch.randn(mean.shape).to(device) for _ in range(k)]).squeeze(dim=1)

class Decoder(torch.nn.Module):

    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.in_decoder = torch.nn.Linear(128, 2048)

        self.convs = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 16, 3, 2, 1, output_padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 3, 3, padding= 1),
            torch.nn.Sigmoid(),          
        ])
    
    def forward(self, x):
        x = self.in_decoder(x)
        x = x.view(-1, 512, 2, 2)
        [x:=conv(x) for conv in self.convs]
        return x


def epoch(X, k, loss_func, model, optim):
    X = X.to(device)
    mean, std = model[0](X)
    z = posteriors(mean, std, k)
    Y = model[1](z)
    kl_term = torch.mean(0.5*torch.sum(1+2*torch.log(std)-torch.square(mean)-torch.square(std), dim=1), dim=0)
    loss = loss_func(X.squeeze(), Y.squeeze()) - 0.00025*kl_term
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, z


if __name__ == '__main__':
    # celebA_transform = transforms.Compose(
    #     [transforms.Resize(128), transforms.CenterCrop(128) , transforms.ToTensor()])
    # celebA_dataset = datasets.ImageFolder(
    #     root=path, transform=celebA_transform)
    # train_set = Subset(celebA_dataset, torch.arange(0, 500))
    # dataloader = DataLoader(celebA_dataset, batch_size=256)

    bitmoji_transform = transforms.Compose([
                               transforms.Resize((128, 128)),
                               transforms.CenterCrop((128, 128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = torch.nn.Sequential(
        Encoder(),
        Decoder()
    )

    #model.load_state_dict(torch.load(modelpath))
    model.to(device)

    optim = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()

    Z = []
    for ep in range(3):
        with tqdm(dataloader) as tepoch:
            for X,_ in tepoch:
                loss, z = epoch(X, 2, loss_func, model, optim)
                #Z.append(z.detach().cpu().numpy())
                tepoch.set_postfix({'loss':loss.item()})
                tepoch.refresh()
    
    #np.save(latentpath, Z, allow_pickle=True)
    torch.save(model.state_dict(), modelpath)

    #reconstruction

    # for X,_ in dataloader:
    #     X = X.to(device)
    #     mean, std = model[0](X)
    #     print(mean.shape, std.shape)
    #     z = posteriors(mean, std, 1)
    #     print(z.shape)
    #     Y = model[1](z)
    #     for i in range(10):
    #         plt.imshow(Y[i].detach().cpu().numpy().T)
    #         plt.show()
    #     break

    # #sample 
    # z = torch.randn(10, 128)
    # z = z.to(device)
    # Y = model[1](z)
    # print(Y.shape)
    # for x in Y:
    #     plt.imshow(x.detach().cpu().numpy().T)
    #     plt.show()


