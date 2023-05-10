from random import shuffle
import torch
from torch.nn import init, Module, ReLU, Sequential, ModuleList, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Sigmoid, ConvTranspose2d, BatchNorm2d, Tanh, Dropout2d, Dropout
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.nn.functional as F
from torchvision.utils import save_image
torch.autograd.set_detect_anomaly(True)

# Initialization parameters
device = 'cuda:0'
datapath = '/home/adarsh/ADRL/datasets/bitmoji_faces'
modelpath = '/home/adarsh/ADRL/assignment_1/GAN/bi_gan/'
fake_path = '/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/bitmoji_fake_bi_gan/'
real_path = '/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/bitmoji_real/'

# Latent Length
Z = 100


class Encoder(torch.nn.Module):
    '''
    Maps I: 216x176 -> Z:128
    '''
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        filter_size = 64

        self.convs = torch.nn.ModuleList([
            Conv2d(3, filter_size, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size), 
            ReLU(True),

            Conv2d(filter_size, filter_size * 2, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 2), 
            ReLU(True),

            Conv2d(filter_size * 2, filter_size * 4, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 4), 
            ReLU(True),

            Conv2d(filter_size * 4, filter_size * 8, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 8), 
            ReLU(True),

            Conv2d(filter_size * 8, filter_size * 8, 4, 1, 0, bias=False), 
            BatchNorm2d(filter_size * 8), 
            ReLU(True),

            Conv2d(filter_size * 8, 100, 1, 1)
        ])

    def forward(self, x):
        [ x := conv(x) for conv in self.convs ]
        return x

class Generator(torch.nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()

        filter_size = 64
        self.convs = torch.nn.ModuleList([
            ConvTranspose2d(100, filter_size * 8, 4, 1, 0, bias=False), 
            BatchNorm2d(filter_size* 8), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size * 8, filter_size * 4, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 4), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size * 4, filter_size * 2, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size * 2), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size * 2, filter_size, 4, 2, 1, bias=False), 
            BatchNorm2d(filter_size), 
            ReLU(inplace=True),

            ConvTranspose2d(filter_size, 3, 4, 2, 1, bias=False), 
            Tanh()
        ])           
    
    def forward(self, x):
        [x:=conv(x) for conv in self.convs]
        return x

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        filter_size = 64
        # Inference over x
        self.convs = torch.nn.ModuleList([
            Conv2d(3, filter_size, 4, 2, 1), 
            LeakyReLU(0.2),

            Conv2d(filter_size, filter_size * 2, 4, 2, 1), 
            BatchNorm2d(filter_size * 2),
            LeakyReLU(0.2),

            Conv2d(filter_size * 2, filter_size * 4, 4, 2, 1), 
            BatchNorm2d(filter_size * 4),
            LeakyReLU(0.2),

            Conv2d(filter_size * 4, filter_size * 8, 4, 2, 1), 
            BatchNorm2d(filter_size * 8),
            LeakyReLU(0.2),

            Conv2d(filter_size* 8, filter_size * 8, 4, 1, 0), 
            LeakyReLU(0.2)
        ])

        self.convs2 = torch.nn.ModuleList([
            Conv2d(100, 512, 1, 1, 0), 
            LeakyReLU(0.2),

            Conv2d(512, 512, 1, 1, 0), 
            LeakyReLU(0.2)
        ])
        
        self.convs3 = torch.nn.ModuleList([
            Conv2d(1024, 2048, 1, 1, 0), 
            LeakyReLU(0.2),

            Conv2d(2048, 2048, 1, 1, 0), 
            LeakyReLU(0.2),

            Conv2d(2048, 1, 1, 1, 0),
            torch.nn.Sigmoid()
        ])

    def forward(self, x, z):

        [ x := conv(x) for conv in self.convs ]
        [ z := conv(z) for conv in self.convs2 ]
        xz = torch.cat((x, z), dim=1)
        [ xz := conv(xz) for conv in self.convs3 ]
        return xz

def sample(k):
    return torch.randn((k, Z, 1, 1)).to(device)

if __name__ == '__main__':

    bitmoji_transform = transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.CenterCrop((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    dataset = datasets.ImageFolder(root=datapath, transform=bitmoji_transform)

    train_dataset = Subset(dataset, torch.arange(300))

    model = Sequential(Discriminator(), Generator(), Encoder()).to(device)

    i = 0
    batch_size = 128
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optim_generator = torch.optim.Adam([{'params' : model[1].parameters()},
                         {'params' : model[2].parameters()}],lr=0.0002,betas=(0.5,0.999))
    optim_discriminator = torch.optim.Adam(model[0].parameters(),lr=0.0002,betas=(0.5,0.999))

    for ep in range(15):
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep+1} : ')
            for X, _ in tepoch:
                x_real = X.to(device)
                z_real = sample(batch_size)

                z_fake, x_fake = model[2](x_real), model[1](z_real)
                real_preds, fake_preds = model[0](x_real, z_fake), model[0](x_fake, z_real)

                disc_loss = F.binary_cross_entropy(real_preds.squeeze(), torch.ones((len(real_preds),)).to(device)) + F.binary_cross_entropy(fake_preds.squeeze(), torch.zeros((len(fake_preds),)).to(device))
                gen_loss = F.binary_cross_entropy(real_preds.squeeze(), torch.zeros((len(real_preds),)).to(device)) + F.binary_cross_entropy(fake_preds.squeeze(), torch.ones((len(fake_preds),)).to(device))
                
                if not i%6 == 0: # train discriminator 5 times & generator once
                    optim_discriminator.zero_grad()
                    disc_loss.backward()
                    optim_discriminator.step()
                
                else:
                    optim_generator.zero_grad()
                    gen_loss.backward()
                    optim_generator.step()

                tepoch.set_postfix({'loss': (disc_loss.item(), gen_loss.item())})
                tepoch.refresh()
                i += 1
                if i % 500 == 0: 
                    torch.save(model.state_dict(), modelpath+'bi_gan'+str(i)+'.pt')
                    z = torch.randn(100, 100, 1, 1).to(device)
                    z = model[1](z).cpu()
                    save_image(z*0.5+0.5,
                            f'/home/adarsh/ADRL/assignment_1/GAN/bi_gan/bi_{str(i)}.png', nrow=10)

    '''fid = {}

    for f in os.listdir(modelpath):

        # load model
        model.load_state_dict(torch.load(os.path.join(modelpath, f)))
        
        # clear folder of fake images
        for _f in os.listdir(fake_path):
            os.remove(os.path.join(fake_path, _f))
        
        ## generate fake images using current generator
        noise = sample(1000)
        with torch.no_grad():
            fake_img = model[1](noise).detach_().cpu()
        for j in range(1000):
            utils.save_image(fake_img[j]*0.5 + 0.5, '%simg%d.png' % (fake_path, j))

        #calculating FID using the following command (third party)
        command = 'python -m pytorch_fid "' + real_path + '" "' + fake_path + '" --device cuda:0'
        res = subprocess.getstatusoutput(command)

        #append the FID to the FID list
        fid_score = float(res[1][res[1].rfind(' '):])
        fid[int(f.split('gan')[1].split('.pt')[0])] = fid_score

        print("%s: %f"%(f, fid_score))

    list1 = []
    for x in sorted(list(fid.keys())):
        list1.append(fid[x])

    plt.figure(figsize=(8,5))
    plt.title("FID Score Plot Bi-GAN")

    plt.plot(list1)
    plt.xlabel("Iterations % 500")
    plt.ylabel("FIDs")
    plt.savefig('/home/adarsh/ADRL/assignment_1/GAN/bitmoji_fid/bigan_fid.png')'''

    # Latent Traversal code
    '''z1 = sample(1).to(device)
    z1_img = model[1](z1).cpu()

    Z = torch.zeros((0, 100, 1, 1))
    Z_img = torch.zeros((0, 3, 64, 64))
    val = z1[:,25,:,:].item()
    print(val)
    for i in range(100):
        new_z = z1.clone().detach()
        to_add = i - 49
        new_z[:,25,:,:] = val + (to_add)*0.5
        print(new_z[:,25,:,:])
        new_z_img = model[1](new_z).cpu()
        Z = torch.cat((Z, new_z.cpu()))
        Z_img = torch.cat((Z_img, new_z_img))

    print(Z_img.shape)
    save_image(Z_img*0.5+0.5,
            f'/home/adarsh/ADRL/assignment_1/GAN/bi_gan/bi_gan_latent_traversal.png')'''