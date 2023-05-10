import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/home/adarsh/ADRL/assignment_2/Diffusion/logs/EBM')
global_step = 0
# Reference : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html

# Model is Copied. Made few modifications


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNModel(nn.Module):

    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2
        c_hid4 = hidden_features*4

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid4, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid4, c_hid4, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid4*4, c_hid3),
            Swish(),
            nn.Linear(c_hid3, out_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x

# Not Copied


def sample(model: torch.nn.Module, input_images: torch.Tensor, steps, step_size):
    for parameter in model.parameters():
        parameter.requires_grad = False
    torch.set_grad_enabled(True)
    input_images.requires_grad = True
    noise = torch.randn(input_images.shape, device=input_images.device)
    for k in range(steps):
        noise = torch.randn(input_images.shape, device=input_images.device)
        input_images.data.add_(noise.data)
        input_images.data.clamp_(min=-1.0, max=1.0)

        # Part 2: calculate gradients for the current input.
        out_imgs = -model(input_images)
        out_imgs.sum().backward()
        # For stabilizing and preventing too high gradients
        input_images.grad.data.clamp_(-0.03, 0.03)

        # Apply gradients to our current samples
        input_images.data.add_(-step_size * input_images.grad.data)
        input_images.grad.detach_()
        input_images.grad.zero_()
        input_images.data.clamp_(min=-1.0, max=1.0)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return input_images

# Not Copied


def batch_train(model: torch.nn.Module, X: torch.Tensor, optim: torch.optim.Adam, global_step):
    # Randomly select Langevin steps between 2 and 6
    T = np.random.randint(1, 60)
    fake_X = sample(model, input_images=torch.randn(
        X.shape, device=X.device), steps=T, step_size=10)  # step size to be tuned

    optim.zero_grad()
    # Constrastive Loss Divergence
    fake_X, X = model(fake_X), model(X)
    loss = fake_X.mean() - X.mean() + 0.1*(X ** 2 + fake_X ** 2).mean()
    loss.backward()
    optim.step()
    return loss.item(), global_step

# Not Copied

def epoch(model: torch.nn.Module, dataset: Dataset):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    global_step = 0
    with tqdm(dataloader) as tepoch:
        for X, _ in tepoch:
            loss, global_step = batch_train(model, X.to('cuda:0'), optim, global_step+1)
            writer.add_scalar('train_loss', loss, global_step)
            tepoch.set_postfix({'loss': loss})


if __name__ == '__main__':
    path = '/home/adarsh/ADRL/datasets/bitmoji_faces'
    img_transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_dataset = ImageFolder(root=path, transform=img_transform)
    model = CNNModel().cuda(0)
    img_dataset, _ = random_split(img_dataset, [3*len(img_dataset)//10, len(img_dataset)-3*len(img_dataset)//10])
    for ep in range(100):
        epoch(model, img_dataset)
        torch.save(model.state_dict(), f'/home/adarsh/ADRL/assignment_2/Diffusion/models/EBM/model_{ep}.pth')
        images = sample(model, torch.randn(torch.Size([32,3,32,32])).cuda(0), 60, 10)
        save_image(images, f'/home/adarsh/ADRL/assignment_2/Diffusion/results/ebm_{ep}.png', nrow=8, normalize=True)
    # model.load_state_dict(torch.load('/home/adarsh/ADRL/assignment_2/Diffusion/models/EBM/model.pth'))
    # images = sample(model, torch.randn(torch.Size([32,3,32,32])).cuda(0), 60, 0.5)
    # save_image(images, '/home/adarsh/ADRL/assignment_2/Diffusion/results/ebm.png', nrow=8, normalize=True)
