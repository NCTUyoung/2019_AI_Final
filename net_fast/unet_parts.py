# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
class shuffle(nn.Module):
    def __init__(self, groups):
        super(shuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        x = channel_shuffle(x,self.groups)
        return x
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,groups):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1,groups=groups),
            shuffle(groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3,padding=1,groups=groups),
            shuffle(groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1,groups=groups),
            shuffle(groups),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class double_conv_in(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1,groups=out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x



class down(nn.Module):
    def __init__(self, in_ch, out_ch,groups):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch,groups),
            shuffle(groups)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, groups,bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.PixelShuffle(2)

        self.conv = double_conv(in_ch, out_ch,groups)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                         diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#        print(x1.shape,x2.shape)
        x = torch.cat([x2, x1], dim=1)
        x = channel_shuffle(x,2)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3,32,3,1),
            nn.GroupNorm(4, 32),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,64,3,1),
            nn.GroupNorm(4, 64),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,64,3,1),
            nn.GroupNorm(4, 64),
            nn.MaxPool2d(2),
            nn.Conv2d(64,32,3,1),
            nn.GroupNorm(4, 32),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,3,3,1),
            nn.Tanh(),
        ) 
        

    def forward(self, x):
        validity = self.model(x)
        return validity