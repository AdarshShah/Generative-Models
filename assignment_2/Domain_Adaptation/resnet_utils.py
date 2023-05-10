from turtle import forward
import torch
import torchsummary
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Function
from torchvision.models import resnet50, ResNet50_Weights


layer=4 #3 #4

class ResNet(nn.Module):
    def __init__(self, num_classes=2, grayscale=False, size=32, use_resnet=False, gradient_reversal=False) -> None:
        super().__init__()
        self.resnet_encoder = ResNetEncoder(grayscale=grayscale, size=size, use_resnet=use_resnet, gradient_reversal=gradient_reversal)
        self.classifier = Discriminator(num_classes=num_classes, size=size, use_resnet=use_resnet, gradient_reversal=gradient_reversal)

    def forward(self, x):

        x = self.resnet_encoder(x)
        x = self.classifier(x)

        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1)
        self.skipconv = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        try:
            return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)) + self.bn(self.skipconv(x)), inplace=True)
        except:
            return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)) + self.skipconv(x), inplace=True)

class ResNetEncoder(nn.Module):

    def __init__(self, grayscale=False, size=64, use_resnet=False, gradient_reversal=False) -> None:
        super().__init__()
        self.size = size
        self.use_resnet = use_resnet
        self.grayscale = grayscale
        if not use_resnet:
            if grayscale:
                self.res1 = ResidualBlock(1, 2 ** 4) #64
            else:
                self.res1 = ResidualBlock(3, 2 ** 4) #64 16
                
            self.res2 = ResidualBlock(2 ** 4, 2 ** 5) #32 8
            self.res3 = ResidualBlock(2 ** 5, 2 ** 6) #16 4
            self.res4 = ResidualBlock(2 ** 6, 2 ** 7) #8 2
            self.res5 = ResidualBlock(2 ** 7, 2 ** 8) #4 1
            self.res6 = ResidualBlock(2 ** 8, 2 ** 9) #2
            self.res7 = ResidualBlock(2 ** 9, 2 ** 10) #1
        else:
            resnet = resnet50(ResNet50_Weights.DEFAULT)
            modules = list(resnet.children())
            # for mod in modules[:-1*(layer+2)]:
            #     mod.requires_grad_(False)
            self.resnet = nn.Sequential(*(modules[:-1*layer]))

    def forward(self, x:torch.Tensor):
        if not self.use_resnet:
            if self.size == 64:
                h = self.res7(self.res6(self.res5(self.res4(self.res3(self.res2(self.res1(x)))))))
                h = h.squeeze()
            elif self.size == 32:
                h = self.res6(self.res5(self.res4(self.res3(self.res2(self.res1(x))))))
                h = h.squeeze()
            else:
                h = self.res5(self.res4(self.res3(self.res2(self.res1(x)))))
                h = h.squeeze()
        else:
            if self.grayscale:
                x = x.repeat(1,3,1,1)
            h = self.resnet(x)
        return h

# rb = ResNet().cuda(0)
# torchsummary.summary(rb, (3,64,64))

lam = 1
class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, i):
        return i

    @staticmethod
    def backward(ctx, grad_output):
        return -lam*grad_output

class GradientReversalModule(nn.Module):

    def __init__(self) -> None:
        super(GradientReversalModule, self).__init__()
    
    def forward(self, x):
        return GradientReversalLayer.apply(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes=1, size=32, gradient_reversal=False, use_resnet=False) -> None:
        super().__init__()
        self.gradient_reversal = gradient_reversal
        self.size = size
        self.use_resnet = use_resnet
        if use_resnet:
            resnet = resnet50(ResNet50_Weights.DEFAULT)
            modules = list(resnet.children())
            self.resnet = nn.Sequential(*(modules[-1*layer:-1]))
            self.classifier = nn.Sequential(nn.Linear(2048, num_classes))
        else:
            self.classifier64 = nn.Sequential(nn.Linear(2 ** 10, 2 ** 7), nn.ReLU(inplace=True), nn.Linear(2 ** 7, num_classes), nn.Softmax(dim=1))
            self.classifier32 = nn.Sequential(nn.Linear(2 ** 9, 2 ** 6), nn.ReLU(inplace=True), nn.Linear(2 ** 6, num_classes), nn.Softmax(dim=1))
            self.classifier16 = nn.Sequential(nn.Linear(2 ** 8, 2 ** 6), nn.ReLU(inplace=True), nn.Linear(2 ** 5, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        if self.gradient_reversal:
            x = GradientReversalLayer.apply(x)
        
        if self.use_resnet:
            h = self.resnet(x).squeeze()
            return self.classifier(h)
        else:
            if self.size == 64:
                return self.classifier64(x)

            elif self.size == 32:
                return self.classifier32(x)

            else:
                return self.classifier16(x)



def resnet_train(model, train_dataset, val_dataloader=None, val_size=1, epochs=10, batch_size=128, cycle_gan=False, gen=None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    optim = torch.optim.Adam(model.parameters())
    if cycle_gan and gen:
        gen.eval()
    for ep in range(epochs):
        model.train()
        with tqdm(train_dataloader) as tepoch:
            tepoch.set_description(f'Epoch {ep}')
            for x,y in tepoch:
                if cycle_gan:
                    x = gen(x.cuda(0))
                pred = model(x.cuda(0))
                loss = F.cross_entropy(pred, y.cuda(0))
                optim.zero_grad()
                loss.backward()
                optim.step()
                tepoch.set_postfix({'loss':loss.item()})
        
        model.eval()
        if val_dataloader is not None:
            with torch.no_grad():
                count = 0
                with tqdm(val_dataloader) as tepoch:
                    for x,y in tepoch:
                        pred = model(x.cuda(0))
                        pred_y = torch.argmax(pred, dim=1)
                        _count = (pred_y == y.cuda(0)).long().sum().item()
                        count += _count

                    val_acc = count/val_size
                    print(val_acc)

                    # if val_acc > 0.98:
                    #     torch.save(model.state_dict(), '/home/adarsh/ADRL/assignment_2/Domain_Adaptation/saved_model/model' + str(val_acc) + '.pt')

@torch.no_grad()
def evaluate(model, dataset):
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    count = 0
    with tqdm(dataloader) as tepoch:
        for x,y in tepoch:
            pred = model(x.cuda(0))
            pred_y = torch.argmax(pred, dim=1)
            count += (pred_y == y.cuda(0)).long().sum().item()
    return count/len(dataset)

'''
Resnet 50 Architecture
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 16, 16]           9,408
       BatchNorm2d-2           [-1, 64, 16, 16]             128
              ReLU-3           [-1, 64, 16, 16]               0
         MaxPool2d-4             [-1, 64, 8, 8]               0
            Conv2d-5             [-1, 64, 8, 8]           4,096
       BatchNorm2d-6             [-1, 64, 8, 8]             128
              ReLU-7             [-1, 64, 8, 8]               0
            Conv2d-8             [-1, 64, 8, 8]          36,864
       BatchNorm2d-9             [-1, 64, 8, 8]             128
             ReLU-10             [-1, 64, 8, 8]               0
           Conv2d-11            [-1, 256, 8, 8]          16,384
      BatchNorm2d-12            [-1, 256, 8, 8]             512
           Conv2d-13            [-1, 256, 8, 8]          16,384
      BatchNorm2d-14            [-1, 256, 8, 8]             512
             ReLU-15            [-1, 256, 8, 8]               0
       Bottleneck-16            [-1, 256, 8, 8]               0
           Conv2d-17             [-1, 64, 8, 8]          16,384
      BatchNorm2d-18             [-1, 64, 8, 8]             128
             ReLU-19             [-1, 64, 8, 8]               0
           Conv2d-20             [-1, 64, 8, 8]          36,864
      BatchNorm2d-21             [-1, 64, 8, 8]             128
             ReLU-22             [-1, 64, 8, 8]               0
           Conv2d-23            [-1, 256, 8, 8]          16,384
      BatchNorm2d-24            [-1, 256, 8, 8]             512
             ReLU-25            [-1, 256, 8, 8]               0
       Bottleneck-26            [-1, 256, 8, 8]               0
           Conv2d-27             [-1, 64, 8, 8]          16,384
      BatchNorm2d-28             [-1, 64, 8, 8]             128
             ReLU-29             [-1, 64, 8, 8]               0
           Conv2d-30             [-1, 64, 8, 8]          36,864
      BatchNorm2d-31             [-1, 64, 8, 8]             128
             ReLU-32             [-1, 64, 8, 8]               0
           Conv2d-33            [-1, 256, 8, 8]          16,384
      BatchNorm2d-34            [-1, 256, 8, 8]             512
             ReLU-35            [-1, 256, 8, 8]               0
       Bottleneck-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 128, 8, 8]          32,768
      BatchNorm2d-38            [-1, 128, 8, 8]             256
             ReLU-39            [-1, 128, 8, 8]               0
           Conv2d-40            [-1, 128, 4, 4]         147,456
      BatchNorm2d-41            [-1, 128, 4, 4]             256
             ReLU-42            [-1, 128, 4, 4]               0
           Conv2d-43            [-1, 512, 4, 4]          65,536
      BatchNorm2d-44            [-1, 512, 4, 4]           1,024
           Conv2d-45            [-1, 512, 4, 4]         131,072
      BatchNorm2d-46            [-1, 512, 4, 4]           1,024
             ReLU-47            [-1, 512, 4, 4]               0
       Bottleneck-48            [-1, 512, 4, 4]               0
           Conv2d-49            [-1, 128, 4, 4]          65,536
      BatchNorm2d-50            [-1, 128, 4, 4]             256
             ReLU-51            [-1, 128, 4, 4]               0
           Conv2d-52            [-1, 128, 4, 4]         147,456
      BatchNorm2d-53            [-1, 128, 4, 4]             256
             ReLU-54            [-1, 128, 4, 4]               0
           Conv2d-55            [-1, 512, 4, 4]          65,536
      BatchNorm2d-56            [-1, 512, 4, 4]           1,024
             ReLU-57            [-1, 512, 4, 4]               0
       Bottleneck-58            [-1, 512, 4, 4]               0
           Conv2d-59            [-1, 128, 4, 4]          65,536
      BatchNorm2d-60            [-1, 128, 4, 4]             256
             ReLU-61            [-1, 128, 4, 4]               0
           Conv2d-62            [-1, 128, 4, 4]         147,456
      BatchNorm2d-63            [-1, 128, 4, 4]             256
             ReLU-64            [-1, 128, 4, 4]               0
           Conv2d-65            [-1, 512, 4, 4]          65,536
      BatchNorm2d-66            [-1, 512, 4, 4]           1,024
             ReLU-67            [-1, 512, 4, 4]               0
       Bottleneck-68            [-1, 512, 4, 4]               0
           Conv2d-69            [-1, 128, 4, 4]          65,536
      BatchNorm2d-70            [-1, 128, 4, 4]             256
             ReLU-71            [-1, 128, 4, 4]               0
           Conv2d-72            [-1, 128, 4, 4]         147,456
      BatchNorm2d-73            [-1, 128, 4, 4]             256
             ReLU-74            [-1, 128, 4, 4]               0
           Conv2d-75            [-1, 512, 4, 4]          65,536
      BatchNorm2d-76            [-1, 512, 4, 4]           1,024
             ReLU-77            [-1, 512, 4, 4]               0
       Bottleneck-78            [-1, 512, 4, 4]               0
           Conv2d-79            [-1, 256, 4, 4]         131,072
      BatchNorm2d-80            [-1, 256, 4, 4]             512
             ReLU-81            [-1, 256, 4, 4]               0
           Conv2d-82            [-1, 256, 2, 2]         589,824
      BatchNorm2d-83            [-1, 256, 2, 2]             512
             ReLU-84            [-1, 256, 2, 2]               0
           Conv2d-85           [-1, 1024, 2, 2]         262,144
      BatchNorm2d-86           [-1, 1024, 2, 2]           2,048
           Conv2d-87           [-1, 1024, 2, 2]         524,288
      BatchNorm2d-88           [-1, 1024, 2, 2]           2,048
             ReLU-89           [-1, 1024, 2, 2]               0
       Bottleneck-90           [-1, 1024, 2, 2]               0
           Conv2d-91            [-1, 256, 2, 2]         262,144
      BatchNorm2d-92            [-1, 256, 2, 2]             512
             ReLU-93            [-1, 256, 2, 2]               0
           Conv2d-94            [-1, 256, 2, 2]         589,824
      BatchNorm2d-95            [-1, 256, 2, 2]             512
             ReLU-96            [-1, 256, 2, 2]               0
           Conv2d-97           [-1, 1024, 2, 2]         262,144
      BatchNorm2d-98           [-1, 1024, 2, 2]           2,048
             ReLU-99           [-1, 1024, 2, 2]               0
      Bottleneck-100           [-1, 1024, 2, 2]               0
          Conv2d-101            [-1, 256, 2, 2]         262,144
     BatchNorm2d-102            [-1, 256, 2, 2]             512
            ReLU-103            [-1, 256, 2, 2]               0
          Conv2d-104            [-1, 256, 2, 2]         589,824
     BatchNorm2d-105            [-1, 256, 2, 2]             512
            ReLU-106            [-1, 256, 2, 2]               0
          Conv2d-107           [-1, 1024, 2, 2]         262,144
     BatchNorm2d-108           [-1, 1024, 2, 2]           2,048
            ReLU-109           [-1, 1024, 2, 2]               0
      Bottleneck-110           [-1, 1024, 2, 2]               0
          Conv2d-111            [-1, 256, 2, 2]         262,144
     BatchNorm2d-112            [-1, 256, 2, 2]             512
            ReLU-113            [-1, 256, 2, 2]               0
          Conv2d-114            [-1, 256, 2, 2]         589,824
     BatchNorm2d-115            [-1, 256, 2, 2]             512
            ReLU-116            [-1, 256, 2, 2]               0
          Conv2d-117           [-1, 1024, 2, 2]         262,144
     BatchNorm2d-118           [-1, 1024, 2, 2]           2,048
            ReLU-119           [-1, 1024, 2, 2]               0
      Bottleneck-120           [-1, 1024, 2, 2]               0
          Conv2d-121            [-1, 256, 2, 2]         262,144
     BatchNorm2d-122            [-1, 256, 2, 2]             512
            ReLU-123            [-1, 256, 2, 2]               0
          Conv2d-124            [-1, 256, 2, 2]         589,824
     BatchNorm2d-125            [-1, 256, 2, 2]             512
            ReLU-126            [-1, 256, 2, 2]               0
          Conv2d-127           [-1, 1024, 2, 2]         262,144
     BatchNorm2d-128           [-1, 1024, 2, 2]           2,048
            ReLU-129           [-1, 1024, 2, 2]               0
      Bottleneck-130           [-1, 1024, 2, 2]               0
          Conv2d-131            [-1, 256, 2, 2]         262,144
     BatchNorm2d-132            [-1, 256, 2, 2]             512
            ReLU-133            [-1, 256, 2, 2]               0
          Conv2d-134            [-1, 256, 2, 2]         589,824
     BatchNorm2d-135            [-1, 256, 2, 2]             512
            ReLU-136            [-1, 256, 2, 2]               0
          Conv2d-137           [-1, 1024, 2, 2]         262,144
     BatchNorm2d-138           [-1, 1024, 2, 2]           2,048
            ReLU-139           [-1, 1024, 2, 2]               0
      Bottleneck-140           [-1, 1024, 2, 2]               0
          Conv2d-141            [-1, 512, 2, 2]         524,288
     BatchNorm2d-142            [-1, 512, 2, 2]           1,024
            ReLU-143            [-1, 512, 2, 2]               0
          Conv2d-144            [-1, 512, 1, 1]       2,359,296
     BatchNorm2d-145            [-1, 512, 1, 1]           1,024
            ReLU-146            [-1, 512, 1, 1]               0
          Conv2d-147           [-1, 2048, 1, 1]       1,048,576
     BatchNorm2d-148           [-1, 2048, 1, 1]           4,096
          Conv2d-149           [-1, 2048, 1, 1]       2,097,152
     BatchNorm2d-150           [-1, 2048, 1, 1]           4,096
            ReLU-151           [-1, 2048, 1, 1]               0
      Bottleneck-152           [-1, 2048, 1, 1]               0
          Conv2d-153            [-1, 512, 1, 1]       1,048,576
     BatchNorm2d-154            [-1, 512, 1, 1]           1,024
            ReLU-155            [-1, 512, 1, 1]               0
          Conv2d-156            [-1, 512, 1, 1]       2,359,296
     BatchNorm2d-157            [-1, 512, 1, 1]           1,024
            ReLU-158            [-1, 512, 1, 1]               0
          Conv2d-159           [-1, 2048, 1, 1]       1,048,576
     BatchNorm2d-160           [-1, 2048, 1, 1]           4,096
            ReLU-161           [-1, 2048, 1, 1]               0
      Bottleneck-162           [-1, 2048, 1, 1]               0 -13 
          Conv2d-163            [-1, 512, 1, 1]       1,048,576 -12 <-
     BatchNorm2d-164            [-1, 512, 1, 1]           1,024 -11
            ReLU-165            [-1, 512, 1, 1]               0 -10
          Conv2d-166            [-1, 512, 1, 1]       2,359,296 -9
     BatchNorm2d-167            [-1, 512, 1, 1]           1,024 -8
            ReLU-168            [-1, 512, 1, 1]               0 -7
          Conv2d-169           [-1, 2048, 1, 1]       1,048,576 -6 <-
     BatchNorm2d-170           [-1, 2048, 1, 1]           4,096 -5
            ReLU-171           [-1, 2048, 1, 1]               0 -4
      Bottleneck-172           [-1, 2048, 1, 1]               0 -3
AdaptiveAvgPool2d-173           [-1, 2048, 1, 1]              0 -2 <- 
          Linear-174                 [-1, 1000]       2,049,000 -1 
================================================================
Total params: 25,557,032
Trainable params: 25,557,032
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 5.87
Params size (MB): 97.49
Estimated Total Size (MB): 103.37
----------------------------------------------------------------
'''