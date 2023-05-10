from resnet_utils import ResNetEncoder, Discriminator, evaluate, ResNet, resnet_train
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np

device='cuda:0'
results = {}

'''
Results:

1. Layer 1:
{'mnist-usps': (0.711425044575504, 0.812097106021122), 'usps-mnist': (0.5251166666666667, 0.6537166666666666),
 'real-clip': (0.17512049575395916, 0.1037411062657792), 'clip-real': (0.1720504009163803, 0.0882016036655212)}
2. Layer 2:
{'mnist-usps': (0.7219860101494994, 0.9600877794541215), 'usps-mnist': (0.6770333333333334, 0.9409), 
'real-clip': (0.18384209318338307, 0.21138397980261647), 'clip-real': (0.2316151202749141, 0.21786941580756014)}
3. Layer 3:
{'mnist-usps': (0.8570840762584008, 0.9138664106432588), 'usps-mnist': (0.7607166666666667, 0.9038333333333334), 
'real-clip': (0.18682579756713336, 0.16341519394078494), 'clip-real': (0.20595647193585337, 0.15853379152348224)}
'''

class DANN_Dataset(Dataset):

    def __init__(self, dataset1, dataset2) -> None:
        super(DANN_Dataset, self).__init__()
        self.data = ConcatDataset((dataset1, dataset2))
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        p = np.random.rand()
        if p < 0.5:
            index = np.random.randint(low=0, high=self.len1)
            return self.data[index][0], self.data[index][1], 1
        index = self.len1 + np.random.randint(low=0, high=self.len2)
        return self.data[index][0], self.data[index][1], 0 #if index >= self.len1 else 1

def DANN_simulation(dataset1, dataset2, size=32, num_classes=10, grayscale=False, batch_size=256, epochs=10, use_resnet=False):
    feature_extractor = ResNetEncoder(grayscale=grayscale, size=size, use_resnet=use_resnet, gradient_reversal=True).to(device)
    fe_optim = torch.optim.Adam(feature_extractor.parameters())
    class_discriminator = Discriminator(num_classes=num_classes, size=size, use_resnet=use_resnet).to(device)
    cd_optim = torch.optim.Adam(class_discriminator.parameters(), lr=1e-4)
    domain_discriminator = Discriminator(num_classes=2, size=size, gradient_reversal=True, use_resnet=use_resnet).to(device)
    dm_optim = torch.optim.Adam(domain_discriminator.parameters(), lr=1e-4)

    dataset1_classifier = ResNet(num_classes=num_classes, grayscale=grayscale, size=size, use_resnet=use_resnet, gradient_reversal=False).to(device)
    print('> training without DANN')
    resnet_train(dataset1_classifier, dataset1, epochs=epochs, batch_size=batch_size)
    acc0 = evaluate(dataset1_classifier, dataset1)
    print(f'Accuracy on dataset1 : {acc0}')
    acc1 = evaluate(dataset1_classifier, dataset2)
    print(f'Accuracy on dataset2 without DAN : {acc1}')

    acc2 = 0
    dann_dataset = DANN_Dataset(dataset1, dataset2)
    train_data_loader = DataLoader(dann_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    print('> training with DANN')
    for ep in range(epochs):
        with tqdm(train_data_loader) as tepoch:
            tepoch.set_description(f'Epoch {ep}')
            for X, cls, domain in tepoch:
                    domain = domain.to(device)
                    fe_optim.zero_grad()
                    dm_optim.zero_grad()
                    cd_optim.zero_grad()
                    features = feature_extractor(X.to(device))
                    loss = F.cross_entropy(domain_discriminator(features),domain) + (F.cross_entropy(class_discriminator(features), cls.to(device), reduction='none')*domain).sum()/domain.sum()
                    loss.backward()
                    fe_optim.step()
                    dm_optim.step()
                    cd_optim.step()
                    
                    tepoch.set_postfix({'loss':loss.item()})
        model = nn.Sequential(feature_extractor, class_discriminator)
        acc2_ = evaluate(model, dataset2)
        acc2 = max(acc2, acc2_)
        print(f'Accuracy on dataset2 with DAN : {acc2_}, Best :{acc2}')
    
    print(f'Accuracy on dataset2 without DAN : {acc1}')
    print(f'Accuracy on dataset2 with DAN : {acc2}')
    torch.cuda.empty_cache()
    return acc1, acc2

# common_transforms = transforms.Compose([transforms.Resize(32), transforms.Grayscale(), transforms.ToTensor()])
# usps = datasets.USPS(root='/home/adarsh/ADRL/datasets/usps', train=True, download=True, transform=common_transforms)
# mnist = datasets.MNIST(root='/home/adarsh/ADRL/datasets/mnist', train=True, download=True, transform=common_transforms)

# acc1, acc2 = DANN_simulation(dataset1=mnist, dataset2=usps, size=32, num_classes=10, grayscale=True, batch_size=128, epochs=5, use_resnet=True)
# results['mnist-usps'] = (acc1, acc2)

# acc1, acc2 = DANN_simulation(dataset1=usps, dataset2=mnist, size=32, num_classes=10, grayscale=True, batch_size=128, epochs=5, use_resnet=True)
# results['usps-mnist'] = (acc1, acc2)

common_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
clipart = datasets.ImageFolder('/home/adarsh/ADRL/datasets/OfficeHomeDataset_10072016/Clipart', transform=common_transforms)
real_world = datasets.ImageFolder('/home/adarsh/ADRL/datasets/OfficeHomeDataset_10072016/Real_world', transform=common_transforms)

acc1, acc2 = DANN_simulation(dataset1=clipart, dataset2=real_world, size=32, num_classes=65, grayscale=False, batch_size=128, epochs=20, use_resnet=True)
results['real-clip'] = (acc1, acc2)

acc1, acc2 = DANN_simulation(dataset1=real_world, dataset2=clipart, size=32, num_classes=65, grayscale=False, batch_size=128, epochs=20, use_resnet=True)
results['clip-real'] = (acc1, acc2)
print(results)