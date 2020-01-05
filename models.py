import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import argparse

def double_conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                         nn.ReLU(True),
                         nn.BatchNorm2d(out_channels),
                         nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
                         nn.ReLU(True),
                         nn.BatchNorm2d(out_channels))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
def init_weights(net_layer):
    print('initialization method normal')
    try:
        net_layer.apply(weights_init_normal)
    except:
        raise NotImplementedError('initialization method has not been implemented')



class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(3),
                                         nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16))
        init_weights(self.conv1)
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16),
                                         nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32))
        init_weights(self.conv2)
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32),
                                         nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(64))
        init_weights(self.conv3)
        
        '''self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(128))
        init_weights(self.conv4)
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(256))
        init_weights(self.conv5)'''

        self.linear = nn.Sequential(nn.Linear(64 * 8 * 32, 64 * 8),
                                    nn.ReLU(True),
                                    nn.Linear(64 * 8, 2 * 8),
                                    nn.ReLU(True),
                                    nn.Linear(2 * 8, 1))
        init_weights(self.linear)
        
        self.sigmoid = nn.Sigmoid()
        
        
        #self.model = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5) #can work?
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

#U-net
class generator(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
#discriminator
class basic_discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())
        
    def forward(self, x):
        x = self.dis(x)
        return x


#generator
class basic_generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())
        
    def forward(self, x):
        x = self.gen(x)
        return x


        
