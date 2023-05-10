from torchsummary import summary
import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda:0'


class UnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, timesteps) -> None:
        super(UnetBlock, self).__init__()
        self.time_dim = in_channels
        self.conv1 = nn.Conv2d(self.time_dim + in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.timedown = nn.Linear(self.time_dim, self.time_dim)


        self.timeConvDown = nn.Linear(timesteps, in_channels)
        self.timeConvUp = nn.Linear(timesteps, 2*out_channels)

        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv3 = nn.ConvTranspose2d(
            self.time_dim+2*out_channels, out_channels, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.ConvTranspose2d(
            out_channels, in_channels, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(out_channels)
        self.timeup = nn.Linear(self.time_dim, self.time_dim)

    def positional_embedding(self, pos):
        dims = self.time_dim
        inv = 1 / \
            (10000 ** (torch.arange(0, dims, 2).to(device)/dims))
        a = torch.sin(pos*inv)
        b = torch.cos(pos*inv)
        pos = torch.concat((a, b), dim=0)
        return pos[:dims]


    def down(self, x, pos):
        pos = F.relu(self.timedown(self.positional_embedding(pos))).reshape(
            (1, -1, 1, 1)).repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        down = self.pool(self.batchnorm2(F.relu(self.conv2(self.batchnorm1(F.relu(self.conv1(torch.concat((x, pos), dim=1))))))))
        return down

    def up(self, x, h, pos):
        h = torch.concat((x,h), dim=1)
        pos = F.relu(self.timeup(self.positional_embedding(pos))).reshape(
            (1, -1, 1, 1)).repeat(x.shape[0], 1, h.shape[2], h.shape[3])
        h = torch.concat((h,pos),dim=1)
        up = F.relu(self.conv4(self.batchnorm3(F.relu(self.conv3((self.unpool(h)))))))
        return up


class Unet(nn.Module):

    def __init__(self, timesteps) -> None:
        super(Unet, self).__init__()
        self.timesteps = timesteps
        self.u1 = UnetBlock(3, 2 ** 5, timesteps)
        self.u2 = UnetBlock(2 ** 5, 2 ** 6, timesteps)
        self.u3 = UnetBlock(2 ** 6, 2 ** 7, timesteps)
        self.u4 = UnetBlock(2 ** 7, 2 ** 8, timesteps)
        self.u5 = UnetBlock(2 ** 8, 2 ** 9, timesteps)
        self.u6 = UnetBlock(2 ** 9, 2 ** 10, timesteps)

    def positional_embedding(self, pos):
        inv = 1 / \
            (10000 ** (torch.arange(0, self.timesteps, 2).to(device)/self.timesteps))
        a = torch.sin(pos*inv)
        b = torch.cos(pos*inv)
        pos = torch.concat((a, b), dim=0)
        return pos

    def forward(self, x, pos):
        # pos = self.positional_embedding(pos)
        d1 = self.u1.down(x, pos)
        d2 = self.u2.down(d1, pos)
        d3 = self.u3.down(d2, pos)
        d4 = self.u4.down(d3, pos)
        d5 = self.u5.down(d4, pos)
        d6 = self.u6.down(d5, pos)
        u6 = d6
        u5 = self.u6.up(d6, u6, pos)
        u4 = self.u5.up(d5, u5, pos)
        u3 = self.u4.up(d4, u4, pos)
        u2 = self.u3.up(d3, u3, pos)
        u1 = self.u2.up(d2, u2, pos)
        y = self.u1.up(d1, u1, pos)
        return y


# model = Unet(1000).to(device)
# summary(model, (3, 32, 32))
