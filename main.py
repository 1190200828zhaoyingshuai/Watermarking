import torch
from collections import defaultdict
from model.model import MyModel
from cfgs import *
from utils import *
import argparse
import time


def train(model:MyModel, device:torch.device, train_cfg:Traincfg):
    """
    训练脚本
    """
    train_data, val_data = get_data_loaders(train_cfg)
    # 计算一轮中总的step数，以便后续输出log
    file_count = len(train_data.dataset)
    if file_count % train_cfg.batch_size == 0:
        step_total = file_count // train_cfg.batch_size
    else:
        step_total = file_count // train_cfg.batch_size + 1

    output_each = 10 # 每10轮输出一次结果
    images_to_save = 8 # 每轮保存图片的组合体包括的张数
    saved_images_size = (512, 512) # 每张图片调整为(512, 512)

    for epoch in range(train_cfg.start_epoch, train_cfg.number_of_epochs+1):
        print('第{}轮训练开始：'.format(epoch))
        train_losses = defaultdict(AverageMeter)
        step = 1
        t1 = time.time()
        for imgs, _ in train_data:
            imgs = imgs.to(device)
            messages = torch.Tensor(np.random.choice([0, 1], (imgs.shape[0], train_cfg.message_length))).to(device)
            losses, _ = model.train([imgs, messages], epoch)
            for name, loss in losses.items():
                train_losses[name].update(loss)
            if step % output_each == 0 or step == step_total:
                t2 = time.time()
                print('第{}轮训练：{}/{}，用时{:.3f}s'.format(epoch, step, step_total, t2-t1))
                t1 = t2
            step += 1
            # if step > 2:
            #     break
        write_losses(os.path.join(train_cfg.save_folder, 'losses/train.csv'), train_losses, epoch)
    
        valid_losses = defaultdict(AverageMeter)
        save_imgs = True # 每轮只在保存第一个bacth的图片
        print('第{}轮训练：正在验证...'.format(epoch))
        for imgs, _ in val_data:
            imgs = imgs.to(device)
            messages = torch.Tensor(np.random.choice([0, 1], (imgs.shape[0], train_cfg.message_length))).to(device)
            losses, (imgs_w, imgs_w_n, decode_messages) = model.valid([imgs, messages], epoch)
            for name, loss in losses.items():
                valid_losses[name].update(loss)
            if save_imgs:
                save_valid_images(imgs, imgs_w, images_to_save, epoch, 
                            os.path.join(train_cfg.save_folder, 'images'),
                            resize_to=saved_images_size)
                save_imgs = False
                # break
        write_losses(os.path.join(train_cfg.save_folder,'losses/valid.csv'), valid_losses, epoch)
        print('第{}轮训练结束...'.format(epoch))
        save_checkpoint(model, epoch, train_cfg)
        print('模型保存成功...')    
            
def main():
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser('训练')
    parser.add_argument('--new', default=True, type=str)
    args = parser.parse_args()

    train_cfg = Traincfg(
        batch_size=40,
        number_of_epochs=500,
        train_folder='data/train',
        valid_folder='data/valid',
        save_folder='3.9.4',
        start_epoch=1,
        H=128, W=128,
        message_length=30,
        decoder_loss=3,
        mse_weight=0.5,
        encoder_loss=0.5,
        adversarial_loss=1e-3
    )


    model = MyModel(train_cfg, device)
    if args.new == 'False':
        print('正在读取权重...')
        checkpoint = load_checkpoint(train_cfg.save_folder+'/weights')
        model.en_de_coder.load_state_dict(checkpoint['enc-dec-model'])
        model.discriminator.load_state_dict(checkpoint['discrim-model'])
        model.optmizer_en_de_coder.load_state_dict(checkpoint['enc-dec-optim'])
        model.optmizer_discriminator.load_state_dict(checkpoint['discrim-optim'])
        train_cfg.start_epoch = checkpoint['epoch'] + 1
        print('读取完毕...')
    
    train(model, device, train_cfg)


if __name__ == '__main__':
    main()
