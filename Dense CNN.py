from torch.nn import nn
import torch

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.cat([x, out], 1)  # Concatenate the input and output
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DenseBlock(64, 128),  # Dense block 1
            nn.Conv2d(192, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DenseBlock(128, 256),  # Dense block 2
            nn.Conv2d(384, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.upsample_layers = nn.Sequential(
          nn.Conv2d(256, 1024, kernel_size=3, padding=1),

          nn.Upsample(scale_factor=2, mode='bicubic'),

          nn.Conv2d(1024, 256, kernel_size=1), # Added

          nn.Conv2d(256, 256, kernel_size=3, padding=1),  
          nn.ReLU(inplace=True),

          nn.Conv2d(256, 256, kernel_size=3, padding=1),   
          nn.BatchNorm2d(256),    
          nn.ReLU(inplace=True),

          nn.Upsample(scale_factor=2, mode='bicubic'),

          nn.Conv2d(256, 256, kernel_size=3, padding=1), 
          nn.ReLU(inplace=True),

          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),

          nn.Upsample(scale_factor=2, mode='bicubic'),

          nn.Conv2d(256, 64, kernel_size=3, padding=1),  
          nn.ReLU(inplace=True),

          nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.residual_blocks(x)
        x = self.upsample_layers(x)
        return torch.tanh(x)


