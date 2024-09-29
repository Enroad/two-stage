import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import glob
import logging
import torch.utils
import argparse
import time
import dataloader
import model

import Myloss
import numpy as np
from test_1stage import lowlight
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import shutil
from thop import profile
# from torchstat import stat
from torchstat import stat
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'),exist_ok=True)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
def reflection(im):
    mr, mg, mb = torch.split(im, 1, dim=1)
    r = mr / (mr + mg + mb + 0.0001)
    g = mg / (mr + mg + mb + 0.0001)
    b = mb / (mr + mg + mb + 0.0001)
    return torch.cat([r, g, b], dim=1)


def luminance(s):
    return ((s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :])).unsqueeze(1)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train(config):
    os.environ['CUDA_VISIBLE_DEVICES']= config.gpu

    config.save = config.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py'))
    model_path =config.save + '/model_epochs/'
    os.makedirs(model_path, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("train file name = %s", os.path.split(__file__))

    # light_net = model.light_net()
    light_net = model.light_net().cuda()


    # light_net = model.VisualAttentionNetwork().cuda()
    optimizer = torch.optim.Adam(light_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.load_pretrain == True:
        light_net.load_state_dict(torch.load(config.pretrain_dir)['net'])
        optimizer.load_state_dict(torch.load(config.pretrain_dir)['optimizer'])
    print(config.lowlight_images_path,config.lowlightVAN_images_path)

    train_dataset= dataloader.lowlight_loader(config.lowlight_images_path,config.lowlightVAN_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    L_exp = Myloss.L_exp(2, 0.8)
    L_smo = Myloss.L_SMO()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.77)

    light_net.train()
    loss_idx_value = 0
    for epoch in range(config.num_epochs):
        losses=[]
        # for img_lowlight, imgVAN_lowlight in zip(train_loader, train_attention_loader):
        for iteration, dataset in enumerate(train_loader):
            # img_lowlight = img_lowlight
            img_lowlight,imgVAN_lowlight=dataset[0],dataset[1]
            img_lowlight = img_lowlight.cuda()
            imgVAN_lowlight=imgVAN_lowlight.cuda()
            ## flops, params
            flops, params = profile(light_net, inputs=(img_lowlight,imgVAN_lowlight))

            print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            print('Params = ' + str(params / 1000 ** 2) + 'M')
            enhanced_image,  rr1, rr2  = light_net(img_lowlight,imgVAN_lowlight)
            # loss_smo = 1000 * L_smo(rr1) + 5000 * L_smo(rr2)  # 1000
            # loss_exp = 5 * torch.mean(L_exp(enhanced_image, img_lowlight))  # 5
            # loss_locl = 1000 * torch.mean(L_locl(enhanced_image, img_lowlight))  # 800
            # loss_gcol = 1500 * torch.mean(L_gcolor(enhanced_image))  # 20
            loss_smo = 1000*L_smo(rr1) + 5000*L_smo(rr2)  # 1000
            loss_exp =5*torch.mean(L_exp(enhanced_image, img_lowlight))  # 5


            # logging.info('loss check',loss_gcol,loss_locl,loss_exp,loss_smo)
            # loss = loss_exp + loss_smo + loss_locl+ loss_gcol# + loss_spa#+ loss_col# + Loss_TV#+ loss_spa# + loss_smooth #+ loss_noise + loss_exp+ loss_col+ loss_spa  Loss_TV +  loss_locl
            loss = loss_exp + loss_smo
            loss_idx_value += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(light_net.parameters(),config.grad_clip_norm)
            optimizer.step()
            losses.append(loss.item())
            logging.info('train-epoch %03d %03d %f', epoch,iteration, loss)
            scheduler.step()
        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        torch.save({'net': light_net.state_dict(), 'optimizer': optimizer.state_dict()},
                     os.path.join(model_path, 'weights_%d.pth' % epoch))
        with torch.no_grad():
            filePath = 'data/testdata/'
            file_list = os.listdir(filePath)
            for file_name in file_list:
                test_list = glob.glob(filePath + file_name + "/*")
                for i in range(len(test_list)):
                    test_list[i] = test_list[i].replace("\\", "/")
                for image in test_list:
                    lowlight(image,image.replace("testdata","testdata_MYVAN20"),os.path.join(model_path, 'weights_%d.pth' % epoch),config.save+'/result_epoch'+str(epoch))

if __name__ == "__main__":
    start1 = time.perf_counter()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_lowimage/")
    parser.add_argument('--lowlightVAN_images_path', type=str, default="data/train_lowimage_MYVAN20/")
    # parser.add_argument('--lowlightEDG_images_path', type=str, default="data/train_lowimage_edg/")
    parser.add_argument('--lr', type=float, default=0.00001)#0.00001
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    parser.add_argument('--num_epochs', type=int, default=100)  # initial value 200
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_light/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots/pretrained.pth")
    parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')
    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)
    end1 = time.perf_counter()
    #3.7 train-epoch 021 34.500875
    print("final is in : %s Seconds " % (end1 - start1))








