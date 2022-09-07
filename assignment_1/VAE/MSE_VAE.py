import numpy as np
import torch
import matplotlib as plt
import sklearn
from torch.utils.data import DataLoader, Subset
from torchvision.io import read_image
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from pickle import dump, load

torch.autograd.set_detect_anomaly(True)

# Initialization parameters
device = 'cuda:0'
path = '/home/adarsh/ADRL/datasets/img_align_celeba'
modelpath = '/home/adarsh/ADRL/assignment_1/VAE/model_lg.pt'
latentpath = '/home/adarsh/ADRL/assignment_1/VAE/Z_04.npy'
gmmpath = '/home/adarsh/ADRL/assignment_1/VAE/gmm_04.pkl'


# Utility Scripts
class Encoder(torch.nn.Module):
    '''
    Maps I: 216x176 -> Z:1x23x18
    '''

    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 128, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 32, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 16, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 8, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, 1, 3),
        ])
        self.mean = torch.nn.Conv2d(1, 1, 2, 2)
        self.std = torch.nn.Conv2d(1, 1, 2, 2)

    def forward(self, x):
        [x := conv(x) for conv in self.convs]
        return 10*torch.tanh(self.mean(x)), torch.exp(torch.tanh(self.std(x)*0.5))


def posteriors(mean, std, k):
    '''
    K-samples from Normal(mean(X), std(X))
    '''
    return torch.stack([mean + std*torch.randn(mean.shape).to(device) for _ in range(k)]).squeeze(dim=1)


class Decoder(torch.nn.Module):

    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(1, 8, 2, 2, output_padding=1),
            torch.nn.ConvTranspose2d(8, 16, 3),
            torch.nn.ConvTranspose2d(16, 32, 3),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 64, 3, 2),
            torch.nn.ConvTranspose2d(64, 128, 3),
            torch.nn.ConvTranspose2d(128, 256, 3),
            torch.nn.ConvTranspose2d(256, 256, 2, 2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(256, 1, 5),
            torch.nn.Sigmoid()
        ])

    def forward(self, x):
        [x := conv(x) for conv in self.convs]
        return x


def epoch(X, beta, gamma, k, loss_func, model, optim):
    X = X.to(device)
    mean, std = model[0](X)
    z = posteriors(mean, std, k).squeeze(dim=0)
    Y = model[1](z)
    Y = posteriors(Y, gamma, 1).squeeze()
    loss = loss_func(X.squeeze(), Y) - beta*0.5*(1+2*torch.log(
        std.flatten())-torch.square(mean.flatten())-torch.square(std.flatten())).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, z

#Uncomment things when necessary
if __name__ == '__main__':
    celebA_transform = transforms.Compose(
        [transforms.Grayscale(1), transforms.ToTensor()])
    celebA_dataset = datasets.ImageFolder(
        root=path, transform=celebA_transform)
    train_set = Subset(celebA_dataset, torch.arange(0, 200000))
    dataloader = DataLoader(celebA_dataset, batch_size=1)
    train_dataloader = DataLoader(celebA_dataset, batch_size=128)

    model = torch.nn.Sequential(
        Encoder(),
        Decoder()
    )

    # Load Pretrained Model if avaliable in modelpath
    try:
        model.load_state_dict(torch.load(modelpath))
    except:
        print('incorrect modelpath')
    model.to(device)

    # Training Code
    '''optim = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()
    for ep in range(5):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            for X, _ in tepoch:
                loss, z = epoch(X, 0.00005, 0.1, 1, loss_func, model, optim)
                mean, std = model[0](X.to(device))
                tepoch.set_postfix({'loss': loss.item()})
                tepoch.refresh()
    torch.save(model.state_dict(), modelpath)'''

    # Generating Latents for learning Prior
    Z = []
    with tqdm(dataloader) as tepoch:
        for X, _ in tepoch:
            mean, std = model[0](X.to(device))
            z = posteriors(mean, std, 1).squeeze()
            Z.append(z.detach().cpu().numpy())
    np.save(latentpath, Z, allow_pickle=True)

    # Learning Priors on Latents using GMM
    Z = np.load(latentpath, allow_pickle=True)
    Z = np.reshape(Z, (len(Z), 23*18))
    gmm = GaussianMixture(n_components=10, verbose=1, verbose_interval=10)
    gmm.fit(Z)
    dump(gmm, open(gmmpath, 'wb'))
    gmm = load(open(gmmpath, 'rb'))

    # Sampling from gmm
    '''z = gmm.sample()[0]
    z = np.reshape(z, (1, 23, 18))
    Y = model[1](torch.tensor(z).to(device).float())
    plt.imshow(Y.squeeze().detach().cpu().numpy(), 'gray')
    plt.show()'''
