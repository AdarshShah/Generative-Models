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
from sklearn.decomposition import PCA
from torch.distributions.multivariate_normal import MultivariateNormal as MN
import math

torch.autograd.set_detect_anomaly(True)

# Initialization parameters
device = 'cuda:0'
path = '/home/adarsh/ADRL/datasets/img_align_celeba'
modelpath = '/home/adarsh/ADRL/assignment_1/VAE/model_lg.pt'
latentpath = '/home/adarsh/ADRL/assignment_1/VAE/Z_04.npy'
gmmpath = '/home/adarsh/ADRL/assignment_1/VAE/gmm_04.pkl'
figpath = '/home/adarsh/ADRL/assignment_1/VAE/demo.jpg'

class Encoder(torch.nn.Module):
    '''
    Maps I: 128 * 128 -> 128 * 1 (mean & std)
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
    K-samples from P(Z|X) = Normal(mean(X), std(X)) reparameterize
    '''
    return torch.randn(mean.shape).to(device)*std + mean
    # return torch.stack([mean + std*torch.randn(mean.shape).to(device) for _ in range(k)]).squeeze(dim=1)

class Decoder(torch.nn.Module):

    '''
    latent: 128 * 1 -> 128 * 128
    '''
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
        z = posteriors(mean, std, 1)
        a = q(z)
        b = p_theta(z)
        c = p_theta_given_z(x_i, z)
        sum += a/(b*c)

    return L/sum

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
    optim = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()
    for ep in range(5):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            for X, _ in tepoch:
                loss, z = epoch(X, 0.00005, 0.1, 1, loss_func, model, optim)
                mean, std = model[0](X.to(device))
                tepoch.set_postfix({'loss': loss.item()})
                tepoch.refresh()
    torch.save(model.state_dict(), modelpath)

    # Generating Latents for learning Prior
    '''Z = []
    with tqdm(dataloader) as tepoch:
        for X, _ in tepoch:
            mean, std = model[0](X.to(device))
            z = posteriors(mean, std, 1).squeeze()
            Z.append(z.detach().cpu().numpy())
    np.save(latentpath, Z, allow_pickle=True)'''

    # Learning Priors on Latents using GMM
    '''Z = np.load(latentpath, allow_pickle=True)
    Z = np.reshape(Z, (len(Z), 23*18))
    gmm = GaussianMixture(n_components=10, verbose=1, verbose_interval=10)
    gmm.fit(Z)
    dump(gmm, open(gmmpath, 'wb'))
    gmm = load(open(gmmpath, 'rb'))'''

    # Sampling from gmm
    '''z = gmm.sample()[0]
    z = np.reshape(z, (1, 23, 18))
    Y = model[1](torch.tensor(z).to(device).float())
    plt.imshow(Y.squeeze().detach().cpu().numpy(), 'gray')
    plt.show()'''

    # Generate 10x10 random samples
    '''for i in range(100):
        z = gmm.sample()[0]
        z = np.reshape(z, (1, 23, 18))
        Y = model[1](torch.tensor(z).to(device).float())
        plt.subplot(10,10,i+1)
        plt.imshow(Y.squeeze().detach().cpu().numpy(), 'gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(figpath, dpi=300, bbox_inches='tight')'''

    # Marginals calculation
    '''Z = torch.zeros((0, 128))
    x = torch.zeros((0, 49152))

    for X,_ in dataloader:
        X = X.to(device)
        mean, std = model[0](X)
        z = posteriors(mean, std, 1)
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

    for X,_ in dataloader:
        for i in range(10):
            input = X[i].reshape(1, 3, 128, 128)
            print(get_marginal_likelihood(input))
        break'''