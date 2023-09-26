from noise_layers import *
import torch.nn as nn

class Noiser(nn.Module):
    def __init__(self, noise_type):
        super(Noiser, self).__init__()
        self.noise = Identity()
        if noise_type == 'screen_shoot':
            self.noise = screen_shoot()
        elif noise_type == 'screen_shoot_wop':
            self.noise = screen_shoot_wop()
        elif noise_type == 'jpeg':
            self.noise = Jpeg_combined()
    
    def forward(self, imgs, epoch, valid=False):

        return self.noise(imgs, epoch, valid)