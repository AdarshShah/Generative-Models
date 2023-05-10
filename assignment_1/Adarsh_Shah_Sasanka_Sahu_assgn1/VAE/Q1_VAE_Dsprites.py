import torch
from torch import nn
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
import torch
import matplotlib as plt
import sklearn
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from pickle import dump, load
from torchinfo import summary
from torchvision.utils import save_image
from sklearn.decomposition import PCA
from torch.distributions.multivariate_normal import MultivariateNormal as MN
# from torch import tensor as Tensor

device = 'cuda:1'
modelpath = '/data2/home/sasankasahu/h/model_dsprites2718.pt'

class Encoder(torch.nn.Module):
    '''
    Maps I: 216x176 -> Z:1x23x18
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 32, 3, 2, 1),
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
        self.mean = torch.nn.Linear(2048, 128)
        self.std = torch.nn.Linear(2048, 128)
    
    def forward(self, x):
        [ x := conv(x) for conv in self.convs ]
        x = torch.flatten(x, start_dim=1)
        mean = self.mean(x)
        std = torch.exp(self.std(x)*0.5)
        return mean, std
        

def posteriors(mean, std):
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
            torch.nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1),
            torch.nn.Conv2d(32, 1, 3, padding= 1),
            torch.nn.Sigmoid(),          
        ])
    
    def forward(self, x):
        x = self.in_decoder(x)
        x = x.view(-1, 512, 2, 2)
        [x:=conv(x) for conv in self.convs]
        return x


def epoch(X, loss_func, model, optim):
    X = X.to(device)
    mean, std = model[0](X)
    z = posteriors(mean, std)
    # print(z.shape)
    Y = model[1](z)
    # print(Y.shape)
    kl_term = torch.mean(0.5*torch.sum(1+2*torch.log(std)-torch.square(mean)-torch.square(std), dim=1), dim=0)
    loss = loss_func(X.squeeze(), Y.squeeze()) - 0.00025*kl_term
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, z

def q(z, z_PCA, gmm):
    z = z.reshape(1,-1).cpu().detach().numpy()
    z = z_PCA.transform(z)
    out = gmm.score(z)
    return np.exp(out)

def p_theta(z, z_PCA):
    z = z.cpu().detach().numpy()
    z = z_PCA.transform(z)
    p_theta = MN(torch.zeros(z.shape[1]), torch.eye(z.shape[1]))
    out = p_theta.log_prob(torch.tensor(z))

    return np.exp(out)

def p_theta_given_z(x, z, x_PCA):
    z = z.to(device)
    mean = model[1](z.float()).cpu().detach().numpy()

    mean = mean.flatten().reshape(1, -1)
    x = x.cpu().detach().numpy().flatten().reshape(1, -1)
    mean = x_PCA.transform(mean)
    x = x_PCA.transform(x)

    p_theta = MN(torch.tensor(mean), torch.eye(mean.shape[1]))
    log_prob = p_theta.log_prob(torch.tensor(x))

    return np.exp(log_prob)

def get_marginal_likelihood(x_i):
    L = 1000
    sum = 0
    mean, std = model[0](x_i.to(device))
    for i in range(L):
        z = posteriors(mean, std)
        a = q(z)
        b = p_theta(z)
        c = p_theta_given_z(x_i, z)
        sum += a/(b*c)

    return L/sum

class myDataset(Dataset):

    def __init__(self, path, transform=None):

        self.path = path
        dict_ = np.load(self.path, allow_pickle=True, encoding='bytes')
        self.imgs = dict_['imgs']
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = self.transform(img)
        return img

model = torch.nn.Sequential(
        Encoder(),
        Decoder()
    )

model.load_state_dict(torch.load(modelpath))
model.to(device)

path = '/data2/home/sasankasahu/VAE/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

dataset = myDataset(path, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
loss_func = torch.nn.MSELoss()

optim = torch.optim.Adam(model.parameters())

for ep in range(80):
    r_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
            data = data.to(device)
            loss, z = epoch(data, loss_func, model, optim)
            r_loss += loss.item()
    print(r_loss)

    with torch.no_grad():

        for i in range(10):
            z = torch.randn(100, 128).to(device)
            z_img = model[1](z).cpu()
            save_image(z_img,
                    f'/data2/home/sasankasahu/h/my_sample_100_{str(i)}.png', nrow=10)

        torch.save(model.state_dict(), modelpath+str(ep)+str(int(r_loss))+".pt")



# Marginals calculation
'''Z = torch.zeros((0, 128))
x = torch.zeros((0, 4096))

for X in train_loader:
    X = X.to(device)
    mean, std = model[0](X)
    z = posteriors(mean, std)
    bs = X.shape[0]
    Z = torch.cat((Z, z.detach().cpu().view(bs, -1)))
    x = torch.cat((x, X.detach().cpu().view(bs,-1)))

z_PCA = PCA(2)
x_PCA = PCA(4)

z_PCA = z_PCA.fit(Z.numpy())
x_PCA = x_PCA.fit(x.numpy())

all_z_PCA = z_PCA.transform(Z.numpy())

index = np.random.choice(Z.shape[0], Z.shape[0]//10, replace=False) 

z = all_z_PCA[index]

gmm = GaussianMixture(n_components=10)
gmm.fit(z)
labels = gmm.predict(z)

for X in train_loader:
    for i in range(10):
        input = X[i].reshape(1, 1, 64, 64)
        print(get_marginal_likelihood(input))
    break'''