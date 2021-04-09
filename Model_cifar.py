
from torch import nn



class VGG(nn.Module):
    def __init__(self, mode=None, num_classes=10):
        super(VGG, self).__init__()
        self.mode = mode
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),)
        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
           
            nn.MaxPool2d(kernel_size=2, stride=2),)
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.features5 = nn.Sequential(
            nn.Conv2d(512, 512,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=2, stride=2), )
        if self.mode is None:
            pass
        elif self.mode == 'simple':
            freeze(self.features3)
            freeze(self.features4)
            freeze(self.features5)
        elif self.mode == 'normal':
            freeze(self.features5)
        else:
            pass
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, num_classes),
           
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


