import os
import torch
import torch.optim as optim
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck 
import random
from torchvision import datasets, transforms
from tqdm import tqdm

root = '/home/adarsh/ADRL/datasets/'

class MyResNet50(ResNet):

  def __init__(self, *args, **kwargs):
    super().__init__(block=Bottleneck, layers=[3, 4, 6, 3],*args, **kwargs)
    self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    self.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=512
        )
    )

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, feat):
        return self.fc(feat)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=0)
        )

    def forward(self, input):
        out = self.layer(input)
        return out

def resnet_train(encoder, classifier, data_loader, usps_dl, mnist_val):
    encoder.train()
    classifier.train()
    resnet_optim = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4, betas=(0.5, 0.9))
    loss_fn = nn.CrossEntropyLoss()
    epochs = 20
    for epoch in range(epochs):
        with tqdm(enumerate(data_loader)) as tepoch:
            for idx, (x, y) in tepoch:
                resnet_optim.zero_grad()
                preds = classifier(encoder(x.cuda()))
                loss = loss_fn(preds, y.cuda())
                loss.backward()
                resnet_optim.step()
                tepoch.set_postfix({'loss':loss.item()})

    return encoder, classifier


def evaluate(encoder, classifier, data_loader):
    encoder.eval()
    classifier.eval()
    acc = 0
    loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for x, y in data_loader:
        x = encoder(x.cuda())
        preds = classifier(x.cuda())
        loss += loss_fn(preds, y.cuda()).item()
        pred_cls = preds.max(1)[1]
        acc += pred_cls.eq(y.cuda()).cpu().sum()

    encoder.train()
    classifier.train()
    acc = acc/len(data_loader.dataset)
    return acc.item()
    
def adapt(src_encoder, target_encoder, discriminator, source_train_dl, target_train_dl, source_classifier, target_test_dl):

    target_encoder.train()
    discriminator.train()
    loss_fn = nn.CrossEntropyLoss()
    optim_target = optim.Adam(target_encoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_disc = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    epochs = 1
    for epoch in range(epochs):
        loader_zipped = enumerate(zip(source_train_dl, target_train_dl))
        with tqdm(loader_zipped) as tepoch:
            for idx, (x, y) in loader_zipped:
                src_img = x[0].to(device)
                tgt_img = y[0].to(device)
                if idx%15 == 0:    
                    optim_disc.zero_grad()
                    source_feature = src_encoder(src_img)
                    target_feature = target_encoder(tgt_img)
                    source_disc_out = discriminator(source_feature)
                    target_disc_out = discriminator(target_feature)
                    pred_concat = torch.cat((source_disc_out, target_disc_out), 0)

                    # wasserstein loss
                    loss_d = source_disc_out.mean() - target_disc_out.mean()
                    loss_d.backward()
                    optim_disc.step()
                    torch.nn.utils.clip_grad_value_(discriminator.parameters(), 0.01)

                optim_disc.zero_grad()
                optim_target.zero_grad()
                target_feature = target_encoder(src_img)
                target_disc_out = discriminator(target_feature)
                loss_g = -target_disc_out.mean()
                loss_g.backward()
                optim_target.step()
                tepoch.set_postfix({'g_loss':loss_g.item(), 'd_loss':loss_d.item()})
                        
        evaluate(target_encoder, source_classifier, target_test_dl)

    return target_encoder

if __name__ == '__main__':

    dataset = 'mnist'
    # dataset = 'clipart'

    if dataset == 'mnist':
        common_transforms = transforms.Compose([transforms.Grayscale(),transforms.Resize((32,32)),transforms.ToTensor()])
        usps_train = datasets.USPS(root=root+'usps', train=True, download=True, transform=common_transforms)
        usps_test = datasets.USPS(root=root+'usps', train=False, download=True, transform=common_transforms)
        mnist_train= datasets.MNIST(root=root+'mnist', train=True, download=True, transform=common_transforms)
        mnist_test= datasets.MNIST(root=root+'mnist', train=False, download=True, transform=common_transforms)
        source_train_dl = torch.utils.data.DataLoader(mnist_train, batch_size=512, shuffle=True)
        target_train_dl = torch.utils.data.DataLoader(usps_train, batch_size=512, shuffle=True)
        source_test_dl = torch.utils.data.DataLoader(mnist_test, batch_size=512, shuffle=True)
        target_test_dl = torch.utils.data.DataLoader(usps_test, batch_size=512, shuffle=True)
        num_classes = 10

    if dataset == 'clipart':
        common_transforms = transforms.Compose([transforms.Grayscale(),transforms.Resize((32,32)),transforms.ToTensor()])
        clipart_train = datasets.ImageFolder(root+'OfficeHomeDataset_10072016/Clipart', transform=common_transforms)
        real_train = datasets.ImageFolder(root+'OfficeHomeDataset_10072016/Real_world', transform=common_transforms)
        source_train_dl = torch.utils.data.DataLoader(clipart_train, batch_size=512, shuffle=True)
        target_train_dl = torch.utils.data.DataLoader(real_train, batch_size=512, shuffle=True)
        source_test_dl = torch.utils.data.DataLoader(clipart_train, batch_size=512, shuffle=True)
        target_test_dl = torch.utils.data.DataLoader(real_train, batch_size=512, shuffle=True)
        num_classes = 65

    source_encoder = MyResNet50()
    source_classifier = Classifier(num_classes=num_classes)
    target_encoder = MyResNet50()
    discriminator = Discriminator()

    device = 'cuda'
    source_encoder.cuda()
    source_classifier.cuda()
    target_encoder.cuda()
    discriminator.cuda()

    source_encoder, source_classifier = resnet_train(source_encoder, source_classifier, source_train_dl, target_test_dl, source_test_dl)
    print(evaluate(source_encoder, source_classifier, target_test_dl))
    target_encoder.load_state_dict(source_encoder.state_dict())
    target_encoder = adapt(source_encoder, target_encoder, discriminator, source_train_dl, target_train_dl, source_classifier, target_test_dl)
    print(evaluate(target_encoder, source_classifier, target_test_dl))
