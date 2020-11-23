import torch.nn as nn


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.05,inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.05,inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

class ClassNet(nn.Module):
    def __init__(self, num_classes=100, batch_norm=True):
        super(ClassNet, self).__init__()
        self.conv1 = conv(batch_norm, 3, 64, kernel_size=3)
        self.conv2 = conv(batch_norm, 64, 128, kernel_size=3)
        self.conv3 = conv(batch_norm, 128, 256, kernel_size=3)
        self.conv4 = conv(batch_norm, 256, 512, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x