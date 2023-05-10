import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
import pickle
from matplotlib import pyplot as plt


device = 'cuda:0'

writer = SummaryWriter('/home/adarsh/ADRL/assignment_1/VAE/logs/vqvae_run')
zmodelpath = '/home/adarsh/ADRL/assignment_1/VAE/vq_z_vae_model.pt'
datapath = '/home/adarsh/ADRL/datasets/tiny-imagenet-200/tiny-imagenet-200/train'
modelpaths = {
    'enc' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_enc_2.pt',
    'dec' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_dec_2.pt',
    'quan' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_quan_2.pt',
}

class Encoder(nn.Module):
    '''
    I:64x64 -> Z:1x16x16
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 1024, 3, 1, 1),
            nn.ReLU(True),
            
            nn.Conv2d(1024, 512, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(512, 256, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(256, 1, 1, 1)
        ])
    
    def forward(self, h):
        [ h:=layer(h) for layer in self.layers]
        return h


class Decoder(nn.Module):
    '''
    Z:1x32x32 -> I:3x64x64
    '''
    filter = 64
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        filter_size = 128 

        self.layers = nn.ModuleList([
            nn.ConvTranspose2d( 1, filter_size * 8, 1, 1, bias=False), #0
            nn.ReLU(True), #2
            
            nn.ConvTranspose2d( filter_size * 8, filter_size * 4, 4, 2, 1, bias=False), #0
            nn.ReLU(True), #2

            nn.ConvTranspose2d( filter_size * 4, filter_size * 2, 4, 2, 1, bias=False), #0
            nn.ReLU(True), #2
            
            nn.ConvTranspose2d( filter_size * 2, filter_size, 1 , bias=False), #0
            nn.ReLU(True), #2

            nn.ConvTranspose2d( filter_size, 3, 1, bias=False),
            nn.Sigmoid() #20

        ])
    
    def forward(self, h):
        [ h:=layer(h) for layer in self.layers]

        return h

class Quantizer(nn.Module):

    def __init__(self, codebook=128) -> None:
        super(Quantizer, self).__init__()
        self._embedding = nn.Embedding(codebook, 1)
        self.codebook = codebook
        self._embedding_dim = 1
        self._num_embeddings = codebook
    
    def forward(self, inputs:torch.Tensor):
        
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        return quantized.permute(0, 3, 1, 2).contiguous()

def encoder_epoch(encoder:Encoder, decoder:Decoder, quantizer:Quantizer, X, beta, optim:torch.optim.Adam):
    optim.zero_grad()
    loss = nn.functional.binary_cross_entropy(decoder(encoder(X) + (quantizer(encoder(X))-encoder(X)).detach()), X) + beta*torch.nn.functional.mse_loss(quantizer(encoder(X)).detach(), encoder(X))
    #loss = nn.functional.mse_loss(X, decoder(encoder(X))) 
    loss.backward()
    optim.step()
    return loss.item()

def decoder_epoch(encoder:Encoder, decoder:Decoder, quantizer:Quantizer, X, optim:torch.optim.Adam):
    optim.zero_grad()
    loss = nn.functional.binary_cross_entropy(decoder(quantizer(encoder(X)).detach()), X)
    #loss = nn.functional.mse_loss(X, decoder(encoder(X)))
    loss.backward()
    optim.step()
    return loss.item()

def quantizer_epoch(encoder:Encoder, decoder:Decoder, quantizer:Quantizer, X, optim:torch.optim.Adam):
    optim.zero_grad()
    loss =  torch.nn.functional.mse_loss(quantizer(encoder(X).detach()), encoder(X).detach())
    loss.backward()
    optim.step()
    return loss.item()

'''
VAE for Latent Space
'''
class ZEncoder(nn.Module):
    '''
    Z : 256 -> 64
    '''
    def __init__(self) -> None:
        super(ZEncoder, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mean = nn.Linear(64, 64)
        self.std = nn.Linear(64, 64)
    
    def forward(self, z):
        z = self.seq(z)
        return torch.tanh(self.mean(z)), torch.exp(self.std(z))

def sample(mean, std):
    return mean + std*torch.randn(mean.shape).to(device)

class ZDecoder(nn.Module):
    '''
    Z : 64 -> 256
    '''
    def __init__(self) -> None:
        super(ZDecoder, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
        )
    
    def forward(self, z):
        return self.seq(z)

def zepoch(X, beta, gamma, loss_func, model, optim):
    X = X.to(device)
    mean, std = model[0](X)
    z = sample(mean, std)
    Y = model[1](z)
    Y = sample(Y, gamma).squeeze()
    loss = loss_func(X.squeeze(), Y) - beta*0.5*(1+2*torch.log(std.flatten())-torch.square(mean.flatten())-torch.square(std.flatten())).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss, z

zmodelpath  = '/home/adarsh/ADRL/assignment_1/VAE/vq_z_vae_model.pt'
zmodel = nn.Sequential(ZEncoder(), ZDecoder()).to(device)
zoptim = torch.optim.Adam(zmodel.parameters())

if __name__=='__main__':

    imagenet_transform = transforms.Compose([transforms.ToTensor()])

    imagenet = datasets.ImageFolder(datapath, transform=imagenet_transform)
    train_dataloader = DataLoader(imagenet, 16, shuffle=True)

    models = {
        'enc' : Encoder().to(device),
        'dec' : Decoder().to(device),
        'quan' : Quantizer(512//2).to(device)
    }
    try:
        [ models[key].load_state_dict(torch.load(modelpaths[key])) for key in modelpaths.keys()]
    except:
        print('incorrect modelpaths')

    # Training Code
    '''optimizers = {
        'enc' : torch.optim.Adam(models['enc'].parameters()),
        'dec' : torch.optim.Adam(models['dec'].parameters()),
        'quan' : torch.optim.Adam(models['quan'].parameters()),
    }

    for ep in range(2):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            i=0
            for X, _ in tepoch:
                enc_loss = encoder_epoch(models['enc'], models['dec'], models['quan'], X.to(device), 0.25, optimizers['enc'])
                dec_loss = decoder_epoch(models['enc'], models['dec'], models['quan'], X.to(device), optimizers['dec'])
                quan_loss = quantizer_epoch(models['enc'], models['dec'], models['quan'], X.to(device), optimizers['quan'])
                #tepoch.set_postfix({'loss': (enc_loss, dec_loss)})
                tepoch.set_postfix({'loss': (enc_loss, dec_loss, quan_loss)})
                tepoch.refresh()
                writer.add_scalars(f'Errors_{ep}',{
                    'encoder loss':enc_loss,
                    'decoder_loss':dec_loss,
                    'quantizer_loss':quan_loss,
                },
                    global_step=i
                )
                i+=1
                [ torch.save(models[key].state_dict(), modelpaths[key]) for key in modelpaths.keys() ]'''

    # Code to Generate 10x10 images from models with codebook size 128, 256 and 512
    '''device = 'cuda:0'
    datapath = '/home/adarsh/ADRL/datasets/tiny-imagenet-200/tiny-imagenet-200/train'
    modelpaths = {
        'enc' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_enc_3.pt',
        'dec' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_dec_3.pt',
        'quan' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_quan_3.pt',
    }

    models = {
            'enc' : Encoder().to(device),
            'dec' : Decoder().to(device),
            'quan' : Quantizer(512).to(device)
        }
    [ models[key].load_state_dict(torch.load(modelpaths[key])) for key in modelpaths.keys()]

    modelpaths1 = {
        'enc' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_enc_1.pt',
        'dec' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_dec_1.pt',
        'quan' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_quan_1.pt',
    }

    models1 = {
            'enc' : Encoder().to(device),
            'dec' : Decoder().to(device),
            'quan' : Quantizer(512//4).to(device)
        }
    [ models1[key].load_state_dict(torch.load(modelpaths1[key])) for key in modelpaths1.keys()]

    modelpaths2 = {
        'enc' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_enc_2.pt',
        'dec' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_dec_2.pt',
        'quan' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_quan_2.pt',
    }

    models2 = {
            'enc' : Encoder().to(device),
            'dec' : Decoder().to(device),
            'quan' : Quantizer(512//2).to(device)
        }
    [ models2[key].load_state_dict(torch.load(modelpaths2[key])) for key in modelpaths2.keys()]

    imagenet_transform = transforms.Compose([transforms.ToTensor()])
    imagenet = datasets.ImageFolder(datapath, transform=imagenet_transform)
    train_dataloader = DataLoader(imagenet, 1, shuffle=True)
    rgb = transforms.ToPILImage()
    images = []
    i = 0
    for img, _ in train_dataloader:
        images.append(img)
        i+=1
        if i==100:
            break

    i=1
    for X in images:
        img = rgb(models['dec'](models['quan'](models['enc'](X.to(device)))).squeeze())
        plt.subplot(10, 10, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(img)
        i+=1
        if i>100:
            break
    plt.savefig('/home/adarsh/ADRL/assignment_1/VAE/vqvae_512.png', dpi=300)

    i=1
    for X in images:
        img = rgb(models1['dec'](models1['quan'](models1['enc'](X.to(device)))).squeeze())
        plt.subplot(10, 10, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(img)
        i+=1
        if i>100:
            break
    plt.savefig('/home/adarsh/ADRL/assignment_1/VAE/vqvae_128.png', dpi=300)

    i=1
    for X in images:
        img = rgb(models2['dec'](models2['quan'](models2['enc'](X.to(device)))).squeeze())
        plt.subplot(10, 10, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(img)
        i+=1
        if i>100:
            break
    plt.savefig('/home/adarsh/ADRL/assignment_1/VAE/vqvae_256.png', dpi=300)'''

    # Training Code to fit GMM on Z space
    '''Z = []
    rgb = transforms.ToPILImage()
    with torch.no_grad():
        for X,_ in tqdm(train_dataloader):
            Z.append(models['quan'](models['enc'](X.to(device))).cpu().detach().flatten().numpy())
    gmm = GaussianMixture(100)
    gmm.fit(Z[:5000])
    with open('/home/adarsh/ADRL/assignment_1/VAE/vq_vae_gmm.pkl','wb') as pkl:
        pickle.dump(gmm, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/adarsh/ADRL/assignment_1/VAE/vq_vae_gmm.pkl','rb') as pkl:
        gmm = pickle.load(pkl)
    samples = gmm.sample(100)
    i=1
    for sample in samples[0]:
        z = torch.Tensor(sample).reshape((1, 1, 16, 16)).to(device)
        img = rgb(models['dec'](models['quan'](z)).squeeze())
        plt.subplot(10, 10, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(img)
        i+=1
    plt.savefig('/home/adarsh/ADRL/assignment_1/VAE/vqvae_gmm.png', dpi=300)'''


    # Training Code for Simple VAE on Z space
    '''Z = []
    with torch.no_grad():
        for X,_ in tqdm(train_dataloader):
            Z.append(models['quan'](models['enc'](X.to(device))).cpu().detach().flatten().numpy())

    loss_func = torch.nn.MSELoss()
    ZZ = []

    with tqdm(Z) as tepoch:
        for X in tepoch:
            loss, z = zepoch(torch.tensor(X), 0.1, 0.1, loss_func, zmodel, zoptim)
            ZZ.append(z.detach().cpu().flatten().numpy())
            tepoch.set_postfix({'loss':loss.item()})
            tepoch.refresh()

    torch.save(zmodel.state_dict(), '/home/adarsh/ADRL/assignment_1/VAE/vq_z_vae_model.pt')
    zmodel.load_state_dict(torch.load(zmodelpath))
    samples = [ sample(mean=torch.zeros((64,)).to(device), std=1) for _ in range(100) ]
    i=1
    for z in samples:
        z = zmodel[1](z)
        z = z.reshape((1, 1, 16, 16))
        img = rgb(models['dec'](models['quan'](z)).squeeze())
        plt.subplot(10, 10, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(img)
        i+=1
    plt.savefig('/home/adarsh/ADRL/assignment_1/VAE/vqvae_vae.png', dpi=300)'''
