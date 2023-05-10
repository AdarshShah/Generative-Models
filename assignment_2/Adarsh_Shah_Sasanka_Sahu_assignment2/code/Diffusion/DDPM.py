from SimpleUnet import UNet, device
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

'''
Using Linear Beta schedule B0 = 1e-4  -> BT = 0.02 where T = 500/1000
'''
B0 = 1e-4
Bt = 0.02
timesteps = 500 # 500 for bitmoji, 1000 for celebA
beta = torch.linspace(B0, Bt, timesteps).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


path = '/home/adarsh/ADRL/datasets/celebA/classes'
# path = '/home/adarsh/ADRL/datasets/bitmoji_faces'

img_transform = transforms.Compose(
        [ transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# img_dataset = datasets.ImageFolder(root=path, transform=img_transform)
img_dataset = datasets.CelebA('/home/adarsh/ADRL/datasets/celebA', 'train', target_type='attr', transform=img_transform, download=True)
train_dataloader = DataLoader(img_dataset, batch_size=256, shuffle=True, num_workers=8, persistent_workers=True)

#CelebA classwise dataset generator
def generate_classes():
    attr = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'.split()
    under_consideration = 'Bald  Bangs  Big_Lips  Black_Hair  Brown_Hair  Eyeglasses  Gray_Hair  Mustache  Receding_Hairline  Wearing_Hat'.split()
    index = { a:attr.index(a) for a in under_consideration }
    train_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=True)
    no = 0
    for img, y in tqdm(train_dataloader):
        no+=1
        for key, idx in index.items():
            if y[0][idx] == 1:
                save_image(img, f'/home/adarsh/ADRL/datasets/celebA/classes/{key}/{no}.png', normalize=True)


# model = UNet(image_channels=3, n_channels=64, ch_mults=(1,2,2,4), is_attn=(False, False, False, False), n_blocks=2).to(device) #Bitmoji
model = UNet(image_channels=3, n_channels=64, ch_mults=(1,2,4,4), is_attn=(False, False, False, False), n_blocks=2).to(device) #celebA
# model = UNet(image_channels=3, n_channels=64, ch_mults=(1,2,4,4), is_attn=(False, False, False, False), n_blocks=2, num_classes=10).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

@torch.no_grad()
def mu(x0, t):
    return (torch.sqrt(alpha_bar[t])*x0 + torch.sqrt(1-alpha_bar[t])*torch.randn(x0.shape).to(device)).to(device)

a = torch.sqrt(alpha_bar)
b = torch.sqrt(1-alpha_bar)
@torch.no_grad()
def q_xt_x0(x0, t):
    noise = torch.rand_like(x0)
    if t<0:
        return x0, noise
    return (a[t]*x0 + b[t]*noise), noise

@torch.no_grad()
def q_xt_xt_1(xt_1, t):
    noise = torch.randn(xt_1.shape).to(device)
    return (torch.sqrt(alpha[t])*xt_1 + torch.sqrt(1-alpha[t])*noise).to(device), noise


def batch_train(X, model, optim):
    loss = 0
    pos = np.random.randint(timesteps)
    Xt, noise = q_xt_x0(X, pos)
    loss = F.mse_loss(model(Xt, pos), noise)
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()  # type: ignore

def guided_batch_train(X, Y, model, optim, w):
    loss = 0
    pos = np.random.randint(timesteps)
    Xt, noise = q_xt_x0(X, pos)
    loss = F.mse_loss((1+w)*model(Xt, pos, Y+1) - w*model(Xt, pos, torch.zeros(Y.shape).to(device)), noise)
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()  # type: ignore

def epoch(model, dataloader):
    losses = []
    with tqdm(dataloader) as tepoch:
        for X,_ in tepoch:
            losses.append(batch_train(X.to(device), model, optim))
            tepoch.set_postfix({'loss':losses[-1]})
            tepoch.refresh()
    return np.array(losses)

def guided_epoch(model, dataloader):
    losses = []
    with tqdm(dataloader) as tepoch:
        for X,Y in tepoch:
            # k = np.random.randint(1,5)
            # Y = torch.topk(Y, k, dim=1, sorted=False).indices.to(device).long()
            # Y = Y[:,torch.randperm(Y.size()[1])][:,0]
            losses.append(guided_batch_train(X.to(device), Y.to(device), model, optim, w=1))
            if len(losses)>500:
                return np.array(losses)
            tepoch.set_postfix({'loss':losses[-1]})
            tepoch.refresh()
    return np.array(losses)

@torch.no_grad()
def sample(model, batch, size=64, skip=1):
    with torch.no_grad():
        Xt = torch.randn((batch, 3, size, size)).to(device)
        for pos in tqdm(np.flip(np.arange( timesteps))):
            if pos%skip==0:
                Xt = (Xt - model(Xt, pos)*(1-alpha[pos])/torch.sqrt(1-alpha_bar[pos]))/torch.sqrt(alpha[pos]) + ( (((1-alpha_bar[pos-1])/(1-alpha_bar[pos]))*beta[pos]*torch.randn((batch, 3, size, size)).to(device)) if pos > 0 else 0 )
        return Xt

@torch.no_grad()
def guided_sample(model, batch, Y, w, size=64):
    with torch.no_grad():
        Xt = torch.randn((batch, 3, size, size)).to(device)
        for pos in np.flip(np.arange( timesteps)):
            Xt = (Xt - ((1+w)*model(Xt, pos, Y+1) - w*model(Xt, pos, torch.zeros(Y.shape).to(device)))*(1-alpha[pos])/torch.sqrt(1-alpha_bar[pos]))/torch.sqrt(alpha[pos]) + ( (((1-alpha_bar[pos-1])/(1-alpha_bar[pos]))*beta[pos]*torch.randn((batch, 3, size, size)).to(device)) if pos > 0 else 0 )
        return Xt

@torch.no_grad()
def sample_with_history(model, path, size=64, skip=1):
    batch=10
    with torch.no_grad():
        Xt = torch.randn((batch, 3, size, size)).to(device)
        imgs = []
        for pos in tqdm(np.flip(np.arange( timesteps))):
            if pos%skip==0:
                Xt = (Xt - model(Xt, pos)*(1-alpha[pos])/torch.sqrt(1-alpha_bar[pos]))/torch.sqrt(alpha[pos]) + ( (((1-alpha_bar[pos-1])/(1-alpha_bar[pos]))*beta[pos]*torch.randn((batch, 3, size, size)).to(device)) if pos > 0 else 0 )
            if pos%50==0:
                imgs.append(Xt)
        images = torch.concat(imgs, dim=0)
        save_image(0.5*images + 0.5, path, nrow=10)

@torch.no_grad()
def sample100(model, path, size=64, skip=1):
    images = sample(model, 100, size, skip=skip)
    # images = images.permute(0,2,3,1).cpu()
    # images = (images - images.min())/(images.max() - images.min())  # 100 * H * W * C
    save_image(0.5*images+0.5, path, nrow=10, normalize=False)


if __name__=='__main__':
    '''
    The main program contains commented code used for training and sampling purposes. The utility functions are above and in SimpleUnet.py file.
    '''
    # generate_classes()    
    # for ep in range(100):
    #     epoch(model, train_dataloader)  # Unguided Diffusion training
    #     # guided_epoch(model, train_dataloader)   # Guided Diffusion training
    #     torch.save(model.state_dict(), f'/home/adarsh/ADRL/assignment_2/Diffusion/models/celebA/diff_{ep}.pth')
   
    # model = UNet(image_channels=3, n_channels=64, ch_mults=(1,2,2,4), is_attn=(False, False, False, False), n_blocks=2).to(device) #Bitmoji
    # model.load_state_dict(torch.load(f'/home/adarsh/ADRL/assignment_2/Diffusion/models/bitmoji_final.pth',map_location=device))
    # sample100(model, '/home/adarsh/ADRL/assignment_2/Diffusion/results/celebA/celebA_final.png', 64, skip=1)
    # for i in range(1000):
    #     img = sample(model, 1, size=64, skip=4)
    #     save_image(img,f'/home/adarsh/ADRL/assignment_2/Diffusion/results/bitmoji_fake_125/{i}.png', normalize=True)

    # for i in range(10):
    #     img = sample(model, 100, size=64, skip=1)
    #     img = (img.clamp(-1, 1) + 1) / 2
    #     for k in range(100):
    #         save_image(img[k],f'/home/adarsh/ADRL/assignment_2/Diffusion/results/bitmoji_fake/{100*i+k}.png')
    
    # for i in range(10):
    #     img = sample(model, 100, size=64, skip=2)
    #     img = (img.clamp(-1, 1) + 1) / 2
    #     for k in range(100):
    #         save_image(img[k],f'/home/adarsh/ADRL/assignment_2/Diffusion/results/bitmoji_fake_250/{100*i+k}.png')
    
    # for i in range(10):
    #     img = sample(model, 100, size=64, skip=4)
    #     img = (img.clamp(-1, 1) + 1) / 2
    #     for k in range(100):
    #         save_image(img[k],f'/home/adarsh/ADRL/assignment_2/Diffusion/results/bitmoji_fake_125/{100*i+k}.png')


