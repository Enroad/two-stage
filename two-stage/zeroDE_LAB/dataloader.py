import os
import sys
from skimage.color import rgb2lab,lab2rgb
import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):

	# image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	#
	# train_list = image_list_lowlight

	# random.shuffle(train_list)
	files = os.listdir(lowlight_images_path)
	files.sort()
	train_list = [os.path.join(lowlight_images_path, file) for file in files]

	return train_list

	

class lowlight_loader(data.Dataset):
	def __init__(self, lowlight_images_path,lowlightVAN_images_path):
		self.train_list = populate_train_list(lowlight_images_path)
		self.trainVAN_list = populate_train_list(lowlightVAN_images_path)
		self.size = 256
		self.data_list = self.train_list
		self.dataVAN_list = self.trainVAN_list
		print("Total training examples:", len(self.train_list))
	def __getitem__(self, index):
		data_lowlight_path = self.data_list[index]
		data_lowlight = Image.open(data_lowlight_path)
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		# 转换为LAB色彩空间
		data_lowlight_lab = rgb2lab(data_lowlight)
		# 将L通道标准化到 [0, 1]
		data_lowlight_lab[:, :, 0] = data_lowlight_lab[:, :, 0] / 100.0
		# 将a和b通道标准化到 [-1, 1]
		data_lowlight_lab[:, :, 1:] = data_lowlight_lab[:, :, 1:]/ 128.0
		data_lowlight_lab = torch.from_numpy(data_lowlight_lab.transpose(2,0,1)).float()

		dataVAN_lowlight_path = self.dataVAN_list[index]
		dataVAN_lowlight = Image.open(dataVAN_lowlight_path)
		dataVAN_lowlight = dataVAN_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		# 转换为LAB色彩空间
		dataVAN_lowlight_lab = rgb2lab(dataVAN_lowlight)
		# 将L通道标准化到 [0, 1]
		dataVAN_lowlight_lab[:, :, 0] = dataVAN_lowlight_lab[:, :, 0] / 100.0
		# 将a和b通道标准化到 [-1, 1]
		dataVAN_lowlight_lab[:, :, 1:] = dataVAN_lowlight_lab[:, :, 1:]/ 128.0
		dataVAN_lowlight_lab = torch.from_numpy(dataVAN_lowlight_lab.transpose(2,0,1)).float()


		return data_lowlight_lab,dataVAN_lowlight_lab
	def __len__(self):
		return len(self.data_list)



class CustomDataset(data.Dataset):
    def __init__(self, folder_path_original, folder_path_processed):
        self.folder_path_original = folder_path_original
        self.folder_path_processed = folder_path_processed

        # 获取两个文件夹中的文件名列表
        self.original_files = os.listdir(folder_path_original)
        self.processed_files = os.listdir(folder_path_processed)

    def __len__(self):
        return min(len(self.original_files), len(self.processed_files))

    def __getitem__(self, idx):
        # 加载原始图像和处理过的图像
        original_img = Image.open(os.path.join(self.folder_path_original, self.original_files[idx]))
        processed_img = Image.open(os.path.join(self.folder_path_processed, self.processed_files[idx]))

        return original_img, processed_img
