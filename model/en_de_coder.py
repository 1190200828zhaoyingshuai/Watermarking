from .encoder import *
from .decoder import *
from noise import Noiser

class EncoderDecoder(nn.Module):
    """
    将编码器、噪声器和解码器组合到单个管道中
    将输入的图片和水印信息经过编码器生成水印图片
    最后将带噪声的水印图片传给解码器，尝试恢复水印信息
    返回一个二元组(img_w, message_decode)
    """
    def __init__(self, H=128,  l_msg=30):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder_SE(H, l_msg)
        self.noiser = Noiser('screen_shoot_wop')
        self.decoder = Decoder(l_msg)
    
    def forward(self, img, message, epoch, valid=False):
        img_w = self.encoder(img, message)
        img_w_n = self.noiser(img_w, epoch, valid)
        message_decode = self.decoder(img_w_n)

        return img_w, img_w_n, message_decode