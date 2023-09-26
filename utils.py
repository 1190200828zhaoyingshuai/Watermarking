import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils
import numpy as np
import os
import csv

from model.model import MyModel
from cfgs import *


class AverageMeter(object):
    """
    各loss值计算
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def get_data_loaders(train_cfg:Traincfg):
    """
    从指定文件夹中加载图片数据并进行处理
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((train_cfg.H, train_cfg.W), pad_if_needed=True),
            torchvision.transforms.RandomHorizontalFlip(),    # 水平镜像
            torchvision.transforms.RandomVerticalFlip(),      # 竖直镜像
            torchvision.transforms.RandomRotation(45),        # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((train_cfg.H, train_cfg.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_cfg.train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_cfg.batch_size,
                                                shuffle=True, num_workers=8)

    valid_images = datasets.ImageFolder(train_cfg.valid_folder, data_transforms['test'])
    valid_loader = torch.utils.data.DataLoader(valid_images, batch_size=train_cfg.batch_size, 
                                                shuffle=False, num_workers=8)

    print('数据读取成功！')
    return train_loader, valid_loader


def save_valid_images(imgs, imgs_w, imgs_num, epoch, folder, resize_to=None):
    """
    保存第epoch轮验证过程第一个batch的前imgs_num张图片
    """
    images = imgs[:imgs_num, :, :, :].cpu()
    imgs_w = imgs_w[:imgs_num, :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    imgs_w = (imgs_w + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        imgs_w = F.interpolate(imgs_w, size=resize_to)

    stacked_images = torch.cat([images, imgs_w], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, normalize=False)


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


def write_losses(file_name, losses_accu, epoch):
    """
    保存各loss值
    """
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.5f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)


def save_checkpoint(model: MyModel, epoch: int, train_cfg:Traincfg):
    """
    保存模型
    """

    checkpoint_filename = os.path.join(train_cfg.save_folder, 'weights/best{}.pth'.format(epoch))
    checkpoint = {
        'enc-dec-model': model.en_de_coder.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'enc-dec-optim': model.optmizer_en_de_coder.state_dict(),
        'discrim-optim': model.optmizer_discriminator.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)


def load_checkpoint(folder):
    """
    读取保存的最新模型
    """
    pths = os.listdir(folder)
    count = []
    for pth in pths:
        count.append(int(pth[4:-4]))
    count = sorted(count)
    last = count[-1]

    checkpoint_filepath = os.path.join(folder, 'best{}.pth'.format(last))
    checkpoint = torch.load(checkpoint_filepath, map_location='cuda:0')
    return checkpoint


