from model.model import EncoderDecoder
import numpy as np
import torch
from utils import load_checkpoint, AverageMeter
from collections import defaultdict
from torchvision import datasets, transforms
import torchvision.utils
import os
import torch.nn.functional as F
import csv
import cv2
import math
from skimage.metrics import structural_similarity as compare_ssim
import utils

def get_data_loaders(path):
    """
    从指定文件夹中加载图片数据并进行处理
    """
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    test_images = datasets.ImageFolder(path, data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=10, 
                                                shuffle=False, num_workers=4)

    print('数据读取成功！')

    return test_loader

def get_real_loaders(path):
    """
    从指定文件夹中加载图片数据并进行处理
    """
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    test_images = datasets.ImageFolder(path, data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=20, 
                                                shuffle=False, num_workers=4)

    print('数据读取成功！')

    return test_loader

def save_images(imgs, folder, num=0, resize_to=None):
    """
    保存第epoch轮验证过程第一个batch的前imgs_num张图片
    """
    # scale values to range [0, 1] from original range of [-1, 1]
    imgs = (imgs + 1) / 2

    if resize_to is not None:
        imgs = F.interpolate(imgs, size=resize_to)
    for i in range(num, num+imgs.shape[0]):
        filename = os.path.join(folder, 'results-{}.png'.format(i))
        torchvision.utils.save_image(imgs[i-num], filename, normalize=False)
    

def write_losses(file_name, losses_accu):
    """
    保存各loss值
    """
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_to_write = ['{:.4f}'.format(1-loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)


def encode(test_data, device, save_path='./real_tests/encoded_pics/pics', orign_path='./real_tests/orign_pics/pics'):
    ed = EncoderDecoder().to(device)
    checkpoint = load_checkpoint('./weights')
    ed.load_state_dict(checkpoint['enc-dec-model'])
    num = 1
    ed.eval()
    x = torch.Tensor([1,0,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,0,0,0,0,0,1,0,1,1,1,1,0,0]).unsqueeze_(0)
    for imgs, _ in test_data:
        imgs = imgs.to(device)
        messages = x.expand(imgs.shape[0], -1).to(device)
        imgs_w = ed.encoder(imgs, messages).clamp(-1, 1)
        save_images(imgs_w, save_path, num, (128, 128))
        save_images(imgs, orign_path, num, (128, 128))
        num += imgs_w.shape[0]
        if num > 100:
            break


def decode(test_data, device, save_path = './test.csv'):
    ed = EncoderDecoder().to(device)
    checkpoint = load_checkpoint('./weights')
    x = torch.Tensor([1,0,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,0,0,0,0,0,1,0,1,1,1,1,0,0]).unsqueeze_(0)
    ed.load_state_dict(checkpoint['enc-dec-model'])
    ed.eval()
    test_loss = defaultdict(AverageMeter)
    for imgs_w_n, _ in test_data:
        batch_size = imgs_w_n.shape[0]
        imgs_w_n = imgs_w_n.to(device)
        target = x.expand(imgs_w_n.shape[0], -1).to(device)
        messages = ed.decoder(imgs_w_n)
        # 将信息映射至(0, 1)
        messages_rounded = messages.detach().cpu().numpy().round().clip(0, 1)
        # 计算比特位水印信息误差
        bitwise_avg_err = np.sum(np.abs(messages_rounded - target.detach().cpu().numpy())) / (
            batch_size * messages.shape[1])
        test_loss['bitwise_avg_err'].update(bitwise_avg_err)
    write_losses(os.path.join(save_path), test_loss)


def psnr2(img1, img2):
   mse = np.mean((img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 生成初始水印图片
    # test_data = get_data_loaders('data/test')
    # encode(test_data, device)

    # 计算PSNR
    image_names = os.listdir('./real_tests/encoded_pics/pics')
    psnr_mean = 0
    ssim_mean = 0
    for name in image_names:
        img1 = cv2.imread(os.path.join('./real_tests/encoded_pics/pics', name))
        img2 = cv2.imread(os.path.join('./real_tests/orign_pics/pics', name))
        psnr_mean += psnr2(img1, img2)
        ssim_mean += compare_ssim(img1, img2, channel_axis=2)
    psnr_mean /= len(image_names)
    ssim_mean /= len(image_names)
    print(psnr_mean, ssim_mean)

    # 读取翻拍图片并进行解码,test文件夹待构建
    test_data = get_real_loaders('./real_tests/shoot_pics')
    # num = 1
    # for imgs, _ in test_data:
    #     imgs = imgs.to(device)
    #     save_images(imgs, './pics', num, (128, 128))
    #     num += imgs.shape[0]
    decode(test_data, device)
