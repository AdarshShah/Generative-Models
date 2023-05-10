import torch
from VQ_VAE import zmodel, zoptim, zepoch, Encoder, Decoder, Quantizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


device = 'cuda:0'
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

imagenet_transform = transforms.Compose([transforms.ToTensor()])
imagenet = datasets.ImageFolder(datapath, transform=imagenet_transform)
train_dataloader = DataLoader(imagenet, 1, shuffle=True)
rgb = transforms.ToPILImage()

Z = []
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

