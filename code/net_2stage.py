# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual_dense_block import RDB
from utils import *


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


BN_MOMENTUM = 0.1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out
        

class better_upsampling(nn.Module):
      def __init__(self, in_ch, out_ch, scale_factor):
          super(better_upsampling, self).__init__()
          self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)
          self.scale_factor = scale_factor

      def forward(self, x,y):
          x = nn.functional.interpolate(x,size= y.size()[2:], mode='nearest', align_corners=None)
          x = F.pad(x, (3 // 2, int(3 / 2),
                     3 // 2, int(3 / 2)))
          x = self.conv(x)
          return x

class down_Block(nn.Module):
    def __init__(self, in_channels, stride=2,k=1):
        kernel_size=3
        super(down_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, k*in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

class RRDB(nn.Module):
    def __init__(self):
        super(RRDB, self).__init__()
        self.conv1 = RDB(16,4,16)
        self.conv2 = RDB(16,4,16)
        self.conv3 = RDB(16,4,16)
        self.calayer=CALayer(16)
    def forward(self, x):
        
        x1 =  self.conv1(x)  
        x2 =  self.conv2(x1)
        x3 =  self.conv2(x2)
        x = x + x3
        x = self.calayer(x)
        return (x)
     
class first_Net(nn.Module):
    def __init__(self):
        super(first_Net, self).__init__()
        self.conv11 = RRDB()
        self.conv12 = RRDB()
        self.conv13 = RRDB()       
        self.conv20 = BasicBlock(16,16)
        self.conv21 = BasicBlock(16,3)

    def forward(self, x):       
        x11 = self.conv11(x)
        x12 = self.conv12 (x11)
        x13 = self.conv13 (x12)
        x13 = x13+x
        x20 = self.conv20 (x13)
        x21 = self.conv21 (x20)
        return (x21)
class final_Net(nn.Module):
    def __init__(self):
        super(final_Net, self).__init__()
        self.conv01 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv11 = nn.Conv2d(6, 16, 3, 1, 1)
        self.basic_net_1 = first_Net()
        self.basic_net_2 = first_Net()
        self.up = better_upsampling(3,3,2)
    def forward(self, x):
        down_x = F.interpolate(x, scale_factor = 0.5)
        down_x = self.conv01(down_x)
        down_x = self.basic_net_1(down_x)
        up_x = self.up (down_x,x)
        up_x = torch.cat((up_x,x),1)
        up_x = self.conv11(up_x)
        up_x = self.basic_net_2(up_x)
        return (down_x,up_x)