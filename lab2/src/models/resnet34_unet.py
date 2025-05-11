# 實作時有參考 https://blog.csdn.net/weixin_43977304/article/details/121497425 
# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, down=False):
        super(ResidualBlock, self).__init__()
        if down:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size=1, stride=2),
                nn.BatchNorm2d(output_size)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1, stride=2 if down else 1, padding_mode="reflect"),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class ResNet34_UNet(nn.Module):
    def __init__(self, input_size=3, output_size=1, num_block=[3, 4, 6, 3]):
        super(ResNet34_UNet, self).__init__()
        self.init = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downs = nn.ModuleList()        

        self.downs.append(nn.Sequential( #down sample block  1
            ResidualBlock(64, 64, False),
            ResidualBlock(64, 64, False),
            ResidualBlock(64, 64, False)
        ))

        self.downs.append(nn.Sequential( #down sample block  2
            ResidualBlock(64, 128, True),
            ResidualBlock(128, 128, False),
            ResidualBlock(128, 128, False)
        ))

        self.downs.append(nn.Sequential( #down sample block  3
            ResidualBlock(128, 256, True),
            ResidualBlock(256, 256, False),
            ResidualBlock(256, 256, False)
        ))

        self.downs.append(nn.Sequential( #down sample block  4
            ResidualBlock(256, 512, True),
            ResidualBlock(512, 512, False),
            ResidualBlock(512, 512, False)
        ))
        
        self.btnk = ResidualBlock(512,1024, True)
        
        self.ups = nn.ModuleList()
        for feature in reversed([64, 128, 256, 512]):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
                                
        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_size, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.init(x)
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.btnk(x)
        for up in self.ups:
            if isinstance(up, nn.ConvTranspose2d):
                x = up(x)
                x = torch.cat((x, skips.pop()), dim=1)
            else:
                x = up(x)
        x = self.output(x)
        return x

# assert False, "Not implemented yet!"