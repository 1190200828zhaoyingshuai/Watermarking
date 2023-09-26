import torch
import torch.nn as nn
from cfgs import Traincfg
import numpy as np
from loss_functions import pytorch_ssim
from loss_functions.vgg_loss import VGGLoss
from . import EncoderDecoder, Discriminator


class MyModel:
    def __init__(self, cfg:Traincfg, device:torch.device):
        super(MyModel, self).__init__()

        self.en_de_coder = EncoderDecoder(cfg.H, cfg.message_length).to(device)
        self.discriminator = Discriminator().to(device)
        self.optmizer_en_de_coder = torch.optim.Adam(self.en_de_coder.parameters())
        self.optmizer_discriminator = torch.optim.Adam(self.discriminator.parameters())

        self.cfg = cfg
        self.device = device

        self.mse_loss = nn.MSELoss().to(device) # 均方误差
        self.vgg_loss = VGGLoss(device)
        self.ssim = pytorch_ssim.SSIM() # SSIM
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device) # 二分类交叉熵损失

        self.no_watered_label = 1
        self.watered_label = 0


    def train(self, batch:list, epoch):
        """
        在一个batch上训练模型
        batch:[images, messages]
        """
        imgs, messages = batch

        batch_size = imgs.shape[0]
        self.en_de_coder.train()
        self.discriminator.train()

        imgs.requires_grad = True
        imgs_w, img_w_n, decode_messages = self.en_de_coder(imgs, messages, epoch)
        # imgs_w = self.idct(imgs_w)
        # # 生成梯度引导掩膜
        # loss_de = self.mse_loss(decode_messages, messages)
        # inputgrad = torch.autograd.grad(loss_de, imgs, create_graph=True)[0]
        # mask = torch.zeros(inputgrad.shape).to(self.device)
        # for ii in range(inputgrad.shape[0]):
        #             a = inputgrad[ii,:,:,:]
        #             a = (1-(a-a.min())/(a.max()-a.min()))+1
        #             mask[ii,:,:,:] = a.detach()
        # print(mask.shape)
        
        with torch.enable_grad():
            # -------------------------训练鉴别器------------------------
            self.optmizer_discriminator.zero_grad()
            # 不含有水印的图片上训练鉴别器
            d_target_no_watered = torch.full((batch_size, 1), self.no_watered_label, device=self.device)
            d_predict_no_watered = self.discriminator(imgs.detach())
            d_no_watered_loss = self.bce_with_logits_loss(d_predict_no_watered, d_target_no_watered.float())
            d_no_watered_loss.backward()
            # 添加水印的图片上训练鉴别器
            d_target_watered = torch.full((batch_size, 1), self.watered_label, device=self.device)
            d_predict_watered = self.discriminator(imgs_w.detach())
            d_watered_loss = self.bce_with_logits_loss(d_predict_watered, d_target_watered.float())
            d_watered_loss.backward()
            self.optmizer_discriminator.step()

            # -----------------------训练生成器(编解码器)---------------------
            self.optmizer_en_de_coder.zero_grad()
            # 为了欺骗鉴别器，生成器生成带有水印的图片的训练目标应为使鉴别器将器判断为不含水印的图片
            g_target_watered = torch.full((batch_size, 1), self.no_watered_label, device=self.device)
            d_predict_watered_for_g = self.discriminator(imgs_w)
            g_adv_loss = self.bce_with_logits_loss(d_predict_watered_for_g, g_target_watered.float())
            # 编码器和解码器loss通过计算均方误差得到
            # encoder_mse = self.mse_loss(imgs_w*mask.float(), imgs*mask.float()) + self.mse_loss(imgs_w, imgs)
            encoder_mse = self.mse_loss(imgs_w, imgs)
            encoder_vgg = self.vgg_loss(imgs_w, imgs)
            encoder_ssim = self.ssim(imgs_w, imgs)
            g_encoder_loss = self.cfg.mse_weight*encoder_mse + (1-self.cfg.mse_weight)*(1-encoder_ssim) + encoder_vgg
            # g_encoder_loss = self.cfg.mse_weight*encoder_vgg + (1-self.cfg.mse_weight)*(1-encoder_ssim)
            g_decoder_loss = self.mse_loss(decode_messages, messages)
            # 加权计算最终loss
            g_loss = self.cfg.adversarial_loss * g_adv_loss + self.cfg.encoder_loss * g_encoder_loss \
                     + self.cfg.decoder_loss * g_decoder_loss
            
            g_loss.backward()
            self.optmizer_en_de_coder.step()

        # 将信息映射至(0, 1)
        decode_messages_rounded = decode_messages.detach().cpu().numpy().round().clip(0, 1)
        # 计算比特位水印信息误差
        bitwise_avg_err = np.sum(np.abs(decode_messages_rounded - messages.detach().cpu().numpy())) / (
            batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': encoder_mse.item(),
            'encoder_ssim   ': encoder_ssim.item(),
            'decoder_mse    ': g_decoder_loss.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_adv_loss.item(),
            'discr_no_watered_bce': d_no_watered_loss.item(),
            'discr_watered_bce': d_watered_loss.item()
        }
        return losses, (imgs_w, img_w_n, decode_messages)

    def valid(self, batch:list, epoch):
        imgs, messages = batch
        # imgs = self.dct(imgs)

        batch_size = imgs.shape[0]
        self.en_de_coder.eval()
        self.discriminator.eval()

        imgs_w, img_w_n, decode_messages = self.en_de_coder(imgs, messages, epoch, valid=True)
        # imgs_w = self.idct(imgs_w)
        with torch.no_grad():
            d_target_no_watered = torch.full((batch_size, 1), self.no_watered_label, device=self.device)
            d_predict_no_watered = self.discriminator(imgs)
            d_no_watered_loss = self.bce_with_logits_loss(d_predict_no_watered, d_target_no_watered.float())

            d_target_watered = torch.full((batch_size, 1), self.watered_label, device=self.device)
            d_predict_watered = self.discriminator(imgs_w)
            d_watered_loss = self.bce_with_logits_loss(d_predict_watered, d_target_watered.float())

            g_target_watered = torch.full((batch_size, 1), self.no_watered_label, device=self.device)
            g_adv_loss = self.bce_with_logits_loss(d_predict_watered, g_target_watered.float())
            # 编码器和解码器loss通过计算均方误差得到
            encoder_mse = self.mse_loss(imgs_w, imgs)
            encoder_ssim = self.ssim(imgs_w, imgs)
            g_encoder_loss = self.cfg.mse_weight*encoder_mse + (1-self.cfg.mse_weight)*(1-encoder_ssim)
            g_decoder_loss = self.mse_loss(decode_messages, messages)
            # 加权计算最终loss
            g_loss = self.cfg.adversarial_loss * g_adv_loss + self.cfg.encoder_loss * g_encoder_loss \
                     + self.cfg.decoder_loss * g_decoder_loss

        # 将信息映射至(0, 1)
        decode_messages_rounded = decode_messages.detach().cpu().numpy().round().clip(0, 1)
        # 计算比特位水印信息误差
        bitwise_avg_err = np.sum(np.abs(decode_messages_rounded - messages.detach().cpu().numpy())) / (
        batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': encoder_mse.item(),
            'encoder_ssim   ': encoder_ssim.item(),
            'dec_mse        ': g_decoder_loss.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_adv_loss.item(),
            'discr_no_watered_bce': d_no_watered_loss.item(),
            'discr_watered_bce': d_watered_loss.item()
        }
        return losses, (imgs_w, img_w_n, decode_messages)