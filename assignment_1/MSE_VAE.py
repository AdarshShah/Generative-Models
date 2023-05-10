import numpy as np
import torch
import matplotlib as plt
import sklearn
from torch.utils.data import DataLoader, Subset
from torchvision.io import read_image
from torchvision import datasets, transforms
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

#Initialization parameters
device = 'cuda:0'
path = '/home/adarsh/ADRL/datasets/img_align_celeba'
modelpath = '/home/adarsh/ADRL/assignment_1/VAE/model_03.pt'
latentpath = '/home/adarsh/ADRL/assignment_1/VAE/Z_03.pkl'

#Utility Scripts
class Encoder(torch.nn.Module):
    '''
    Maps I: 216x176 -> Z:1x23x18
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 128, 3),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 32, 3),
            torch.nn.Conv2d(32, 16, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 8, 3),
            torch.nn.Conv2d(8, 1, 3),
        ])
        self.mean = torch.nn.Conv2d(1, 1, 2, 2)
        self.std = torch.nn.Conv2d(1, 1, 2, 2)
    
    def forward(self, x):
        [ x := conv(x) for conv in self.convs ]
        return torch.tanh(self.mean(x)), torch.exp(self.std(x)*0.5)

def posteriors(mean, std, k):
    '''
    K-samples from P(Z|X) = Normal(mean(X), std(X))
    '''
    return torch.stack([mean + std*torch.randn(mean.shape).to(device) for _ in range(k)]).squeeze(dim=1)

class Decoder(torch.nn.Module):

    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(1, 1, 2, 2, output_padding=1),
            torch.nn.ConvTranspose2d(1, 8, 3),
            torch.nn.ConvTranspose2d(8, 16, 3),
            torch.nn.ConvTranspose2d(16, 16, 3, 2),
            torch.nn.ConvTranspose2d(16, 32, 3),
            torch.nn.ConvTranspose2d(32, 128, 3),
            torch.nn.ConvTranspose2d(128, 128, 2, 2),
            torch.nn.ConvTranspose2d(128, 128, 3),
            torch.nn.ConvTranspose2d(128, 1, 3),
            torch.nn.Sigmoid()          
        ])
    
    def forward(self, x):
        [ x:=conv(x) for conv in self.convs]
        return x


def epoch(X, k, loss_func, model, optim):
    X = X.to(device)
    mean, std = model[0](X)
    z = posteriors(mean, std, k)
    Y = model[1](z).mean(dim=0)
    loss = loss_func(X.squeeze(), Y.squeeze()) - 0.5*(1+2*torch.log(std.flatten())-torch.square(mean.flatten())-torch.square(std.flatten())).sum()
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, z


if __name__ == '__main__':
    celebA_transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
    celebA_dataset = datasets.ImageFolder(root=path, transform=celebA_transform)
    train_set = Subset(celebA_dataset, torch.arange(0, 5000))
    dataloader = DataLoader(celebA_dataset, batch_size=1)

    model = torch.nn.Sequential(
        Encoder(),
        Decoder()
    )

    #model.load_state_dict(torch.load(modelpath))
    model.to(device)

    optim = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()

    Z = []
    with tqdm(dataloader) as tepoch:
        for X,_ in tepoch:
            loss, z = epoch(X, 2, loss_func, model, optim)
            #Z.append(z.detach().cpu().numpy())
            tepoch.set_postfix({'loss':loss.item()})
            tepoch.refresh()
    
    #np.save(latentpath, Z, allow_pickle=True)
    torch.save(model.state_dict(), modelpath)


