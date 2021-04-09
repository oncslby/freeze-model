
from torch import nn



class VGG16(nn.Module):
    def __init__(self, mode=None, num_classes=10):
        super(VGG16, self).__init__()
        self.mode = mode
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),)
        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),)
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.features5 = nn.Sequential(
            nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        if self.mode is None:
            pass
        elif self.mode == 'simple':
            freeze(self.features5)
            freeze(self.features3)
            freeze(self.features4)
        elif self.mode == 'normal':
            freeze(self.features5)
        else:
            pass
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            #nn.ReLU(True),
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


def make_layers_Cifar10(cfg,batch_norm=False, conv=nn.Conv2d):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = conv(in_channels, filters, kernel_size=5, padding=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)

class CNNCifar(nn.Module):
    def __init__(self):

        super(CNNCifar, self).__init__()
        self.linear = nn.Linear
        cfg = {
            9: ['64', '64', 'M', '128', '128', 'M', '256', '256', 'M'],
            11: ['64', 'M', '128', 'M', '256', '256', 'M', '512', '512', 'M', '512', '512', 'M'],
            13: ['64', '64', 'M', '128', '128', 'M', '256', '256', 'M', '512', '512', 'M', '512', '512', 'M'],
            16: ['64', '64', 'M', '128', '128', 'M', '256', '256', '256', 'M', '512', '512', '512', 'M', '512', '512', '512', 'M'],
        }
        self.conv = nn.Conv2d
        self.features = make_layers_Cifar10(cfg[11], True, self.conv)
        self.classifier=None
        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.linear(512 * 1 * 1, 10),
            # nn.ReLU(True),
            # self.linear(4096, 4096),
            # nn.ReLU(True),
            # self.linear(4096, args.num_classes),
            # nn.ReLU(True),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.classifier(x)
        return x

