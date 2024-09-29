import torch
import torch.nn as nn
import torchvision
from skimage.color import rgb2lab,lab2rgb
import torch.backends.cudnn as cudnn
import torch.optim

import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def join_two_image(img_1, img_2, save_path,flag='horizontal'):
    img1 = Image.open(img_1)
    img2 = Image.open(img_2)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_path)
def lowlight(image_path,imageVAN_path, model_path, resultimage_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    original_image_path=image_path
    data_lowlight = Image.open(image_path)
    data_lowlight_lab = rgb2lab(data_lowlight)
    #[0, 1]
    data_lowlight_lab[:, :, 0] = data_lowlight_lab[:, :, 0]/100
     #[-1, 1]
    data_lowlight_lab[:, :, 1:] = data_lowlight_lab[:, :, 1:] /128.0
    data_lowlight_lab = torch.from_numpy(data_lowlight_lab.transpose(2,0,1)).float().unsqueeze(0)

    dataVAN_lowlight = Image.open(imageVAN_path)
    dataVAN_lowlight_lab = rgb2lab(dataVAN_lowlight)
    #[0, 1]
    dataVAN_lowlight_lab[:, :, 0] = dataVAN_lowlight_lab[:, :, 0]/100
     #[-1, 1]
    dataVAN_lowlight_lab[:, :, 1:] = dataVAN_lowlight_lab[:, :, 1:] /128.0
    dataVAN_lowlight_lab = torch.from_numpy(dataVAN_lowlight_lab.transpose(2,0,1)).float().unsqueeze(0)
    # denoise_net.load_state_dict(torch.load('snapshots_denoise/Epoch24.pth')['net'])
    light_net=model.light_net()
    light_net.load_state_dict(torch.load(model_path)['net'])
    start = time.time()
    img_highlight_lab, _, _ = light_net(data_lowlight_lab,dataVAN_lowlight_lab)
    # img_highlight_lab = torch.cat((img_highlight_l, data_lowlight_lab[:, 1:,:, :]), dim=1)

    img_highlight_lab[:, 0, :,:] = img_highlight_lab[:, 0,:, :] *100
    img_highlight_lab[:, 1:,:, :] = img_highlight_lab[:,1:, :, :] * 128.0
    img_rgb=lab2rgb(img_highlight_lab.squeeze(0).permute(1,2, 0))
    img_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float()
    img_tensor = img_tensor.unsqueeze(0)

    # img_rgb_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))
    # img_final = denoise_net(img_normlight_noise)
    # enhanced_image = img_lu_de * img_re_de

    end_time = (time.time() - start)

    print(end_time)
    image_path = image_path.replace('testdata', resultimage_path)
    # image_path = image_path.replace('LSRW', 'result_LSRW')
    result_path=image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    # img_rgb_pil.save(result_path)
    torchvision.utils.save_image(img_tensor, result_path)
    # print(result_path[:-4]+'(1)'+ result_path[-4:])
    join_two_image(original_image_path, result_path, result_path[:-4]+'(1)'+ result_path[-4:], flag='horizontal')



if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = 'E:\ye/'
        # filePath = 'data/LSRW/'
        model_path = 'E:\ye\codes_SelfDACE\ZeroDE_Lab\ZeroDE_Lab\EXP\Train-20240718-211620\model_epochs\weights_4.pth'
        resultimage_path = 'E:\ye\lol_dataset\zeroLAN'
        file_list = os.listdir(filePath)
        print(file_list)
        # for file_name in file_list:
        #     test_list = glob.glob(filePath + file_name + "/*")
        #     print()
        #     # test_list = glob.glob(filePath + file_name + "/66.jpg")
        #     for i in range(len(test_list)):
        #         test_list[i] = test_list[i].replace("\\", "/")
        #     for image in test_list:
        #         print(image, image.replace("val","val_VAN"))
        #         lowlight(image, image.replace("val","val_VAN"),model_path, resultimage_path):
        test_list = glob.glob(filePath  + "/*")
        print(test_list)
         # test_list = gl
        # xsob.glob(filePath + file_name + "/66.jpg")
        for i in range(len(test_list)):
            test_list[i] = test_list[i].replace("\\", "/")
        for image in test_list:
            print(image, image.replace("lol_dataset", "lol_dataset_van"))
            lowlight(image, image.replace("lol_dataset", "lol_dataset_van"), model_path, resultimage_path)




