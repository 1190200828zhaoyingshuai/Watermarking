from .layers import *
from .dct import *

class Decoder(nn.Module):
    """
    解码器，从带有水印的图片中提取水印
    """
    def __init__(self, l_msg=30):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            # block1
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            # block2
            ResBlock(64, 128, 2), 
            ResBlock(128, 128, 1), 
            # block3
            ResBlock(128, 256, 2),
            ResBlock(256, 256, 1), 
            # block4
            ResBlock(256, 128, 2), 
            ResBlock(128, 128, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, l_msg)
    
    def forward(self, imgs):
        # imgs = dct(imgs)
        output = self.conv(imgs)
        output = self.blocks(output)
        output = self.avgpool(output)
        output = self.fc(output.view(imgs.shape[0], -1))
        return output
    

class Decoder_CA(nn.Module):
    """
    解码器，从带有水印的图片中提取水印
    """
    def __init__(self, l_msg=30):
        super(Decoder_CA, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            CABlock(64, 64, 1), 
            CABlock(64, 128, 2), 
            CABlock(128, 256, 1),
            CABlock(256, 128, 2), 
            CABlock(128, 128, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, l_msg)
    
    def forward(self, imgs):
        output = self.conv(imgs)
        output = self.blocks(output)
        output = self.avgpool(output)
        output = self.fc(output.view(imgs.shape[0], -1))
        return output