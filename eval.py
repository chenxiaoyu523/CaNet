from torch.utils import data
import torch.backends.cudnn as cudnn
import os.path as osp
import tqdm
from dataset_mask_val import Dataset as Dataset_val
import os
import torch
from one_shot_network import Res_Deeplab
import torch.nn as nn
import numpy as np
import cv2
import argparse
import torch.nn.functional as F
from utils import *


data_dir = 'PASCAL-5i'
model_path = 'checkpoint/fo=1/model/best.pth'

parser = argparse.ArgumentParser()

parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=1)

parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')

parser.add_argument('-iter_time',
                    type=int,
                    default=5)

parser.add_argument('-bs_val',
                    type=int,
                    help='batchsize for val',
                    default=1)

options = parser.parse_args()

#set gpus
gpu_list = [int(x) for x in options.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

torch.backends.cudnn.benchmark = True

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 3

input_size = (321, 321)
cudnn.enabled = True

# Create network.
model = Res_Deeplab(num_classes=num_class)
model.load_state_dict(torch.load(model_path))

checkpoint_dir = 'checkpoint/fo=%d/'% options.fold
check_dir(checkpoint_dir)

# this only a quick val dataset where all images are 321*321.
valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=4,
                            drop_last=False)

model.cuda()
model.eval()


valset.history_mask_list=[None] * 1000

for eva_iter in range(options.iter_time):
    all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
    for i_iter, batch in enumerate(valloader):

        query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index, query = batch

        query_rgb = (query_rgb).cuda(0)
        support_rgb = (support_rgb).cuda(0)
        support_mask = (support_mask).cuda(0)
        query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use

        query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
        history_mask = (history_mask).cuda(0)

        pred = model(query_rgb, support_rgb, support_mask,history_mask)
        pred_softmax = F.softmax(pred, dim=1).data.cpu()

        # update history mask
        for j in range(support_mask.shape[0]):
            sub_index = index[j]
            valset.history_mask_list[sub_index] = pred_softmax[j]

            pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                align_corners=True)  #upsample  # upsample

        _, pred_label = torch.max(pred, 1)
        inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
        for j in range(query_mask.shape[0]):#batch size
            all_inter[sample_class[j] - (options.fold * 5 + 1)] += inter_list[j]
            all_union[sample_class[j] - (options.fold * 5 + 1)] += union_list[j]
        '''
        if eva_iter==0:
            cv2.imwrite(os.path.join(checkpoint_dir, 'pred_img', str(index[0].data.numpy()) + '_' + str(eva_iter) + '.jpg'), (query.squeeze().numpy())[:,:,::-1])
        cv2.imwrite(os.path.join(checkpoint_dir, 'pred_img', str(index[0].data.numpy()) + '_' + str(eva_iter+1) + '.jpg'), pred_label.squeeze().data.cpu().numpy()*255)
        '''

    IOU = [0] * 5

    for j in range(5):
        IOU[j] = all_inter[j] / all_union[j]

    mean_iou = np.mean(IOU)
    print('IOU:%.4f' % (mean_iou))


