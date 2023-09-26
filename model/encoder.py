import torch
from .layers import *
import numpy as np
from .dct import *


class Encoder(nn.Module):
    """
    编码器，向图片中添加水印
    """
    def __init__(self, H=128, l_msg=30):
        super(Encoder, self).__init__()
        self.msg_size = int(H/8)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = DoubleConvBNRelu(3, 16)
        self.Conv2 = DoubleConvBNRelu(16, 32)
        self.Conv3 = DoubleConvBNRelu(32, 64)

        self.Up4 = UpConvBNRelu(64*3, 64)
        self.Conv7 = DoubleConvBNRelu(64*3, 64)

        self.Up3 = UpConvBNRelu(64, 32)
        self.Conv8 = DoubleConvBNRelu(32*2+64, 32)

        self.Up2 = UpConvBNRelu(32, 16)
        self.Conv9 = DoubleConvBNRelu(16*2+64, 16)

        self.Conv_1x1 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.msg_linear = nn.Linear(l_msg, self.msg_size**2)
        self.msg_conv = DoubleConvBNRelu(1,64)

    def forward(self, imgs, msg):
        # 16*128*128
        x1 = self.Conv1(imgs)
        x2 = self.Maxpool(x1)
        # 32*64*64
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        # 64*32*32
        x3 = self.Conv3(x3)
        # 64*16*16
        x4 = self.Maxpool(x3)
        x6 = self.Globalpool(x4)
        # 64*16*16
        x7 = x6.repeat(1,1,4,4)

        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = self.msg_conv(expanded_message)
        x4 = torch.cat((x4, x7, expanded_message), dim=1)

        # 64*32*32
        d4 = self.Up4(x4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d4.shape[2],d4.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d4 = torch.cat((x3, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        # 32*64*64
        d3 = self.Up3(d4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d3.shape[2],d3.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d3 = torch.cat((x2, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        # 16*128*128
        d2 = self.Up2(d3)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d2.shape[2],d2.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d2 = torch.cat((x1, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)

        return out


class Encoder_SE(nn.Module):
    """
    编码器，向图片中添加水印
    """
    def __init__(self, H=128, l_msg=30):
        super(Encoder_SE, self).__init__()
        self.msg_size = int(H/8)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = SEBlock(3, 16, 2)
        self.Conv2 = SEBlock(16, 32, 2)
        self.Conv3 = SEBlock(32, 64, 2)

        self.Up4 = UpConvBNRelu(64*3, 64)
        self.Conv7 = SEBlock(64*3, 64)

        self.Up3 = UpConvBNRelu(64, 32)
        self.Conv8 = SEBlock(32*2+64, 32)

        self.Up2 = UpConvBNRelu(32, 16)
        self.Conv9 = SEBlock(16*2+64, 16)

        self.Conv_1x1 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.msg_linear = nn.Linear(l_msg, self.msg_size**2)
        self.msg_conv = SEBlock(1,64)

    def forward(self, imgs, msg):
        imgs = dct(imgs)

        # 16*128*128
        x1 = self.Conv1(imgs)
        x2 = self.Maxpool(x1)
        # 32*64*64
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        # 64*32*32
        x3 = self.Conv3(x3)
        # 64*16*16
        x4 = self.Maxpool(x3)
        x6 = self.Globalpool(x4)
        # 64*16*16
        x7 = x6.repeat(1,1,4,4)

        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = self.msg_conv(expanded_message)
        x4 = torch.cat((x4, x7, expanded_message), dim=1)

        # 64*32*32
        d4 = self.Up4(x4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d4.shape[2],d4.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d4 = torch.cat((x3, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        # 32*64*64
        d3 = self.Up3(d4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d3.shape[2],d3.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d3 = torch.cat((x2, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        # 16*128*128
        d2 = self.Up2(d3)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d2.shape[2],d2.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d2 = torch.cat((x1, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)

        out = idct(out)

        return out


class Encoder_CA(nn.Module):
    """
    编码器，向图片中添加水印
    """
    def __init__(self, H=128, l_msg=30):
        super(Encoder_CA, self).__init__()
        self.msg_size = int(H/8)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = CABlock(3, 16, 2)
        self.Conv2 = CABlock(16, 32, 2)
        self.Conv3 = CABlock(32, 64, 2)

        self.Up4 = UpConvBNRelu(64*3, 64)
        self.Conv7 = CABlock(64*3, 64)

        self.Up3 = UpConvBNRelu(64, 32)
        self.Conv8 = CABlock(32*2+64, 32)

        self.Up2 = UpConvBNRelu(32, 16)
        self.Conv9 = CABlock(16*2+64, 16)

        self.Conv_1x1 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.msg_linear = nn.Linear(l_msg, self.msg_size**2)
        self.msg_conv = CABlock(1,64)

    def forward(self, imgs, msg):
        imgs = dct(imgs)
        # 16*128*128
        x1 = self.Conv1(imgs)
        x2 = self.Maxpool(x1)
        # 32*64*64
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        # 64*32*32
        x3 = self.Conv3(x3)
        # 64*16*16
        x4 = self.Maxpool(x3)
        x6 = self.Globalpool(x4)
        # 64*16*16
        x7 = x6.repeat(1,1,4,4)

        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = self.msg_conv(expanded_message)
        x4 = torch.cat((x4, x7, expanded_message), dim=1)

        # 64*32*32
        d4 = self.Up4(x4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d4.shape[2],d4.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d4 = torch.cat((x3, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        # 32*64*64
        d3 = self.Up3(d4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d3.shape[2],d3.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d3 = torch.cat((x2, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        # 16*128*128
        d2 = self.Up2(d3)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d2.shape[2],d2.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d2 = torch.cat((x1, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)

        out = idct(out)

        return out



class Encoder_CBAM(nn.Module):
    """
    编码器，向图片中添加水印
    """
    def __init__(self, H=128, l_msg=30):
        super(Encoder_CBAM, self).__init__()
        self.msg_size = int(H/8)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = CBAMBlock(3, 16, 2)
        self.Conv2 = CBAMBlock(16, 32, 2)
        self.Conv3 = CBAMBlock(32, 64, 2)

        self.Up4 = UpConvBNRelu(64*3, 64)
        self.Conv7 = CBAMBlock(64*3, 64)

        self.Up3 = UpConvBNRelu(64, 32)
        self.Conv8 = CBAMBlock(32*2+64, 32)

        self.Up2 = UpConvBNRelu(32, 16)
        self.Conv9 = CBAMBlock(16*2+64, 16)

        self.Conv_1x1 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.msg_linear = nn.Linear(l_msg, self.msg_size**2)
        self.msg_conv = CBAMBlock(1,64)

    def forward(self, imgs, msg):
        imgs = dct(imgs)

        # 16*128*128
        x1 = self.Conv1(imgs)
        x2 = self.Maxpool(x1)
        # 32*64*64
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        # 64*32*32
        x3 = self.Conv3(x3)
        # 64*16*16
        x4 = self.Maxpool(x3)
        x6 = self.Globalpool(x4)
        # 64*16*16
        x7 = x6.repeat(1,1,4,4)

        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = self.msg_conv(expanded_message)
        x4 = torch.cat((x4, x7, expanded_message), dim=1)

        # 64*32*32
        d4 = self.Up4(x4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d4.shape[2],d4.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d4 = torch.cat((x3, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        # 32*64*64
        d3 = self.Up3(d4)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d3.shape[2],d3.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d3 = torch.cat((x2, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        # 16*128*128
        d2 = self.Up2(d3)
        expanded_message = self.msg_linear(msg)
        expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
        expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d2.shape[2],d2.shape[3]),mode='bilinear')
        expanded_message = self.msg_conv(expanded_message)
        d2 = torch.cat((x1, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)

        out = idct(out)

        return out
    



# class Encoder_(nn.Module):
#     """
#     编码器，向图片中添加水印
#     """
#     def __init__(self, H=128, l_msg=30):
#         super(Encoder_, self).__init__()
#         self.msg_size = int(H/8)
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

#         self.Conv1 = DoubleConvBNRelu(3, 16)
#         self.Conv2 = DoubleConvBNRelu(16, 32)
#         self.Conv3 = DoubleConvBNRelu(32, 64)

#         self.Up4 = UpConvBNRelu(64*3, 64)
#         self.Conv7 = DoubleConvBNRelu(64*3, 64)

#         self.Up3 = UpConvBNRelu(64, 32)
#         self.Conv8 = DoubleConvBNRelu(32*2+64, 32)

#         self.Up2 = UpConvBNRelu(32, 16)
#         self.Conv9 = DoubleConvBNRelu(16*2+64+3, 16)

#         self.Conv_1x1 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
#         self.msg_linear = nn.Linear(l_msg, self.msg_size**2)
#         self.msg_conv = DoubleConvBNRelu(1,64)

#     def forward(self, imgs, msg):
#         # 16*128*128
#         x1 = self.Conv1(imgs)
#         x2 = self.Maxpool(x1)
#         # 32*64*64
#         x2 = self.Conv2(x2)
#         x3 = self.Maxpool(x2)
#         # 64*32*32
#         x3 = self.Conv3(x3)
#         # 64*16*16
#         x4 = self.Maxpool(x3)
#         x6 = self.Globalpool(x4)
#         # 64*16*16
#         x7 = x6.repeat(1,1,4,4)

#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = self.msg_conv(expanded_message)
#         x4 = torch.cat((x4, x7, expanded_message), dim=1)

#         # 64*32*32
#         d4 = self.Up4(x4)
#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d4.shape[2],d4.shape[3]),mode='bilinear')
#         expanded_message = self.msg_conv(expanded_message)
#         d4 = torch.cat((x3, d4, expanded_message), dim=1)
#         d4 = self.Conv7(d4)

#         # 32*64*64
#         d3 = self.Up3(d4)
#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d3.shape[2],d3.shape[3]),mode='bilinear')
#         expanded_message = self.msg_conv(expanded_message)
#         d3 = torch.cat((x2, d3, expanded_message), dim=1)
#         d3 = self.Conv8(d3)

#         # 16*128*128
#         d2 = self.Up2(d3)
#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d2.shape[2],d2.shape[3]),mode='bilinear')
#         expanded_message = self.msg_conv(expanded_message)
#         d2 = torch.cat((imgs, x1, d2, expanded_message), dim=1)
#         d2 = self.Conv9(d2)

#         out = self.Conv_1x1(d2)

#         return out



# class Encoder_reverse(nn.Module):
#     """
#     编码器，向图片中添加水印
#     """
#     def __init__(self, H=128, l_msg=30):
#         super(Encoder_reverse, self).__init__()
#         self.msg_size = int(H/8)
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

#         self.Conv1 = DoubleConvBNRelu(3, 64)
#         self.Conv2 = DoubleConvBNRelu(64, 32)
#         self.Conv3 = DoubleConvBNRelu(32, 16)

#         self.Up4 = UpConvBNRelu(96, 16)
#         self.Conv7 = DoubleConvBNRelu(96, 16)

#         self.Up3 = UpConvBNRelu(16, 32)
#         self.Conv8 = DoubleConvBNRelu(128, 32)

#         self.Up2 = UpConvBNRelu(32, 64)
#         self.Conv9 = DoubleConvBNRelu(64*3, 64)

#         self.Conv_1x1 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
#         self.msg_linear = nn.Linear(l_msg, self.msg_size**2)
#         self.msg_conv = DoubleConvBNRelu(1,64)

#     def forward(self, imgs, msg):
#         # 64*128*128
#         x1 = self.Conv1(imgs)
#         x2 = self.Maxpool(x1)
#         # 32*64*64
#         x2 = self.Conv2(x2)
#         x3 = self.Maxpool(x2)
#         # 16*32*32
#         x3 = self.Conv3(x3)
#         # 16*16*16
#         x4 = self.Maxpool(x3)
#         x6 = self.Globalpool(x4)
#         # 16*16*16
#         x7 = x6.repeat(1,1,4,4)

#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = self.msg_conv(expanded_message)
#         x4 = torch.cat((x4, x7, expanded_message), dim=1)

#         # 16*32*32
#         d4 = self.Up4(x4)
#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d4.shape[2],d4.shape[3]),mode='bilinear')
#         expanded_message = self.msg_conv(expanded_message)
#         d4 = torch.cat((x3, d4, expanded_message), dim=1)
#         d4 = self.Conv7(d4)

#         # 32*64*64
#         d3 = self.Up3(d4)
#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d3.shape[2],d3.shape[3]),mode='bilinear')
#         expanded_message = self.msg_conv(expanded_message)
#         d3 = torch.cat((x2, d3, expanded_message), dim=1)
#         d3 = self.Conv8(d3)

#         # 64*128*128
#         d2 = self.Up2(d3)
#         expanded_message = self.msg_linear(msg)
#         expanded_message = expanded_message.view(-1, 1 ,self.msg_size, self.msg_size)
#         expanded_message = torch.nn.functional.interpolate(expanded_message,size=(d2.shape[2],d2.shape[3]),mode='bilinear')
#         expanded_message = self.msg_conv(expanded_message)
#         d2 = torch.cat((x1, d2, expanded_message), dim=1)
#         d2 = self.Conv9(d2)

#         out = self.Conv_1x1(d2)

#         return out