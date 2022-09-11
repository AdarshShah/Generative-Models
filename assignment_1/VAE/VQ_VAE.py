import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda:1'
datapath = '/home/adarsh/ADRL/datasets/tiny-imagenet-200/tiny-imagenet-200/train'
modelpaths = {
    'enc' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_enc_2.pt',
    'dec' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_dec_2.pt',
    'quan' : '/home/adarsh/ADRL/assignment_1/VAE/vq_vae_quan_2.pt',
}

class Encoder(nn.Module):
    '''
    I:64x64 -> Z:4x4
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 1, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        ])
    
    def forward(self, h):
        [ h:=layer(h) for layer in self.layers]
        return h

class Decoder(nn.Module):
    '''
    Z:4x4 -> I:64x64
    '''
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(1, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(128, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
    
    def forward(self, h):
        [ h:=layer(h) for layer in self.layers]
        return h

class Quantizer(nn.Module):

    def __init__(self, codebook=128, shape=(4,4)) -> None:
        super(Quantizer, self).__init__()
        self.embeddings = nn.Embedding(codebook, 1)
        self.shape = shape
        self.codebook = codebook
    
    def quantize(self, i:torch.Tensor):
        h = self.embeddings(torch.Tensor([0]).long().to(device))
        curr =  torch.abs(h - i)
        for e in self.embeddings(torch.arange(self.codebook).to(device)):
            if torch.abs(e - i) <= curr:
                curr = torch.abs(e - i)
                h = e
        return h.squeeze()
    
    def forward(self, h:torch.Tensor):
        shape = h.shape
        h = h.flatten()
        h = torch.stack([self.quantize(i) for i in h])
        h = torch.reshape(h, shape)
        return h

def encoder_epoch(encoder:Encoder, decoder:Decoder, quantizer:Quantizer, X, beta, optim:torch.optim.Adam):
    loss = nn.functional.mse_loss(X, decoder(encoder(X))) + beta*torch.norm(quantizer(encoder(X)) - encoder(X))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()

def decoder_epoch(encoder:Encoder, decoder:Decoder, quantizer:Quantizer, X, optim:torch.optim.Adam):
    loss = nn.functional.mse_loss(X, decoder(quantizer(encoder(X))))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()

def quantizer_epoch(encoder:Encoder, decoder:Decoder, quantizer:Quantizer, X, optim:torch.optim.Adam):
    loss = torch.norm(quantizer(encoder(X)) - encoder(X))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()

if __name__=='__main__':

    imagenet = datasets.ImageFolder(datapath, transform=transforms.ToTensor())
    train_dataloader = DataLoader(imagenet, 32, shuffle=True)

    codebook = 128

    models = {
        'enc' : Encoder().to(device),
        'dec' : Decoder().to(device),
        'quan' : Quantizer(codebook).to(device)
    }
    [ models[key].load_state_dict(torch.load(modelpaths[key])) for key in modelpaths.keys()]
    optimizers = {
        'enc' : torch.optim.Adam(models['enc'].parameters()),
        'dec' : torch.optim.Adam(models['dec'].parameters()),
        'quan' : torch.optim.Adam(models['quan'].parameters()),
    }

    # Training Code
    for ep in range(1):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            for X, _ in tepoch:
                quan_loss = quantizer_epoch(models['enc'], models['dec'], models['quan'], X.to(device), optimizers['quan'])
                enc_loss = encoder_epoch(models['enc'], models['dec'], models['quan'], X.to(device), 0.25, optimizers['enc'])
                dec_loss = decoder_epoch(models['enc'], models['dec'], models['quan'], X.to(device), optimizers['dec'])
                tepoch.set_postfix({'loss': (enc_loss, dec_loss, quan_loss)})
                tepoch.refresh()
                [ torch.save(models[key].state_dict(), modelpaths[key]) for key in modelpaths.keys() ]