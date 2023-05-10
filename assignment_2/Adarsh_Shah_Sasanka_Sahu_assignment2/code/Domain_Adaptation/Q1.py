from resnet_utils import ResNet, resnet_train, evaluate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

common_transforms = transforms.Compose([transforms.Resize(32), transforms.Grayscale(), transforms.ToTensor()])

usps_train, usps_test = datasets.USPS(root='/home/adarsh/ADRL/datasets/usps', train=True, download=True, transform=common_transforms), datasets.USPS(root='/home/adarsh/ADRL/datasets/usps', train=False, download=True, transform=common_transforms)
mnist_train, mnist_test = datasets.MNIST(root='/home/adarsh/ADRL/datasets/mnist', train=True, download=True, transform=common_transforms), datasets.MNIST(root='/home/adarsh/ADRL/datasets/mnist', train=False, download=True, transform=common_transforms)

mnist_train_dataloader = DataLoader(mnist_train, batch_size=512, shuffle=True)
usps_train_dataloader = DataLoader(usps_train, batch_size=512, shuffle=True)

mnist_val_dataloader = DataLoader(mnist_test, batch_size=512, shuffle=True)
usps_val_dataloader = DataLoader(usps_test, batch_size=512, shuffle=True)

mnist_classifier = ResNet(num_classes=10, grayscale=True, size=32).cuda(0)

print(mnist_train)
print(usps_train)

resnet_train(mnist_classifier, mnist_train_dataloader, mnist_val_dataloader, len(mnist_test),10)

# print(evaluate(mnist_classifier, mnist_train))
# print(evaluate(mnist_classifier, mnist_test))
print(evaluate(mnist_classifier, usps_train))
# print(evaluate(mnist_classifier, usps_test))

# usps_classifier = ResNet(num_classes=10, grayscale=True, size=32).cuda(0)
# resnet_train(usps_classifier, usps_train_dataloader, usps_val_dataloader, len(usps_test), 10)

# print(evaluate(usps_classifier, usps_train))
# print(evaluate(usps_classifier, usps_test))
# print(evaluate(usps_classifier, mnist_train))
# print(evaluate(usps_classifier, mnist_test))

# The other dataset
# common_transforms = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
# clipart = datasets.ImageFolder('/home/adarsh/ADRL/datasets/OfficeHomeDataset_10072016/Clipart', transform=common_transforms)
# real_world = datasets.ImageFolder('/home/adarsh/ADRL/datasets/OfficeHomeDataset_10072016/Real_world', transform=common_transforms)
