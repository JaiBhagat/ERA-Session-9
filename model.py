import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False, dilation=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias, dilation=dilation)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            depthwise_separable_conv(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, dilation=2, bias=False), # Dilated Convolution
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            depthwise_separable_conv(64, 128, 3, padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), dilation=2, bias=False), # Dilated Convolution
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            depthwise_separable_conv(64, 128, 3, padding=1, bias=False, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=2, padding=1, bias=False), # Changed dilation to 1, stride to 1 and padding to 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))  # Global Average Pooling
        self.fc = nn.Linear(64, 10) # Fully Connected Layer

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
