import random
import os
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time


class Dataset(object):


    def __init__(self, data_dir, fold, input_size=[321, 321], normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1]):

        self.data_dir = data_dir
        self.fold = fold
        self.input_size = input_size
        self.input_list = self.get_input_list(fold=self.fold)

        self.history_mask_list = [None] * len(self.input_list)
        self.query_support_list=[None] * len(self.input_list)
        for index in range (1, len(self.input_list)):
            query_name=self.input_list[index]
            support_name = self.input_list[0] 

            self.query_support_list[index-1]=[query_name,support_name]

        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass

    def get_input_list(self, fold):
        input_list = []

        f = open(os.path.join(self.data_dir, fold, 'val.txt'))
        while True:
            item = f.readline().strip()
            if item == '':
                break
            input_list.append(item)
        return input_list


    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def __getitem__(self, index):

        query_name = self.query_support_list[index][0]
        support_name=self.query_support_list[index][1]

        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = input_size#int(random.uniform(1,1.5)*input_size)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0#random.random()
        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, self.fold, 'img', support_name + '.jpg'))))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, self.fold, 'mask', support_name + '.png')))))

        margin_h = 0#random.randint(0, scaled_size - input_size)
        margin_w = 0#random.randint(0, scaled_size - input_size)

        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        # random scale and crop for query
        scaled_size = 321

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0#random.random()

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, self.fold, 'img', query_name + '.jpg'))))))

        query = np.array(scale_transform_rgb(
                Image.open(os.path.join(self.data_dir, self.fold, 'img', query_name + '.jpg'))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, self.fold, 'mask', support_name + '.png')))))

        margin_h = 0#random.randint(0, scaled_size - input_size)
        margin_w = 0#random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        if self.history_mask_list[index] is None:

            history_mask=torch.zeros(2,41,41).fill_(0.0)

        else:

            history_mask=self.history_mask_list[index]



        return query_rgb, query_mask, support_rgb, support_mask,history_mask,index,query

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return len(self.input_list)-1
