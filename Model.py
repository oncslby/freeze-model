
import torch.nn as nn
'''
Mnist
'''



class CNN(nn.Module):
    def __init__(self, mode=None):
        super(CNN, self).__init__()
        self.mode = mode
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                
                out_channels=16,
                
                kernel_size=3,
                
                stride=1,
               
                padding=1
               
            ),
            nn.ReLU(),
           
            nn.MaxPool2d(kernel_size=2)
           
        )
        #[16, 14, 14]
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
        # [32, 7, 7]

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
        #[64, 3, 3]
        
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
