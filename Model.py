
import torch.nn as nn
'''
Mnist仅为测试
'''



class CNN(nn.Module):
    def __init__(self, mode=None):
        super(CNN, self).__init__()
        self.mode = mode
        # 前面都是规定结构
        # 第一个卷积层，这里使用快速搭建发搭建网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                # 灰度图，channel为一
                out_channels=16,
                # 输出channels自己设定
                kernel_size=3,
                # 卷积核大小
                stride=1,
                # 步长
                padding=1
                # padding=（kernel_size-stride）/2   往下取整
            ),
            nn.ReLU(),
            # 激活函数，线性转意识到非线性空间
            nn.MaxPool2d(kernel_size=2)
            # 池化操作，降维，取其2x2窗口最大值代表此窗口，因此宽、高减半，channel不变
        )
        # 此时shape为[16, 14, 14]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 此时shape为[32, 7, 7]

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 此时shape为[64, 3, 3]
        # 定义全连接层，十分类，并且全连接接受两个参数，因此为[64, 3, 3, 10]
        if self.mode is None:
            pass
        elif self.mode == 'simple':
            freeze(self.conv2)
            freeze(self.conv3)
        elif self.mode == 'normal':
            freeze(self.conv3)
        else:
            pass


        self.prediction = nn.Sequential(
            nn.Linear(64*3*3, 10),
            nn.LogSoftmax(dim=1)
        )
        #前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.prediction(x)
        return output





def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
