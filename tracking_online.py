from torch.utils import data
import torch.backends.cudnn as cudnn
import os.path as osp
import tqdm
from dataset_tracking import Dataset as Dataset_val
import os
import torch
from one_shot_network import Res_Deeplab
import torch.nn as nn
import numpy as np
import cv2
import argparse
import torch.nn.functional as F
from utils import *
import torch.optim as optim
import time
import matplotlib.pyplot as plt

data_dir = 'vot'
model_path = 'checkpoint/fo=1/model/best.pth'

parser = argparse.ArgumentParser()

parser.add_argument('-lr',
                    type=float,
                    help='learning rate',
                    default=0.00005)

parser.add_argument('-fold',
                    type=str,
                    help='fold',
                    default='basketball')

parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')

parser.add_argument('-iter_time',
                    type=int,
                    default=1)

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
num_class = 2
learning_rate = options.lr  # 0.000025#0.00025
input_size = (321, 321)
weight_decay = 0.0005
momentum = 0.9
power = 0.9
online_iter = 0

input_size = (321, 321)
cudnn.enabled = True

# Create network.
model = Res_Deeplab(num_classes=num_class)

optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': 10 * learning_rate}],
                          lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

checkpoint_dir = 'checkpoint/fo=%s/'% options.fold
check_dir(checkpoint_dir)

# this only a quick val dataset where all images are 321*321.
valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=4,
                            drop_last=False)

model.cuda()
model.eval()

valset.history_mask_list=[None] * len(valloader)

all_inter = 0
all_union = 0

def us_forward(query_rgb, support_rgb, support_mask, history_mask, index, model):
    for eva_iter in range(options.iter_time):
        if valset.history_mask_list[index] is not None:
            history_mask = valset.history_mask_list[index].unsqueeze(0)

        history_mask = (history_mask).cuda(0)

        pred = model(query_rgb, support_rgb, support_mask,history_mask)
        pred_softmax = F.softmax(pred, dim=1).data.cpu()

        # update history mask
        for j in range(support_mask.shape[0]):
            sub_index = index[j]
            valset.history_mask_list[sub_index] = pred_softmax[j]

            pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                align_corners=True)  #upsample  # upsample
    return pred

t_start = time.time()

for i_iter, batch in enumerate(valloader):
    model.load_state_dict(torch.load(model_path), strict=False)
    query_rgb, query_mask, support_rgb, support_mask, history_mask, index, query = batch
    '''
    plt.figure()
    plt.imshow(query_rgb.squeeze().permute(1,2,0).numpy()[:,:,::-1])
    plt.figure()
    plt.imshow(support_rgb.squeeze().permute(1,2,0).numpy()[:,:,::-1])
    plt.show()
    '''
    query_rgb = (query_rgb).cuda(0)
    support_rgb = (support_rgb).cuda(0)
    support_mask = (support_mask).cuda(0)
    query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use
    query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use

    # iterate adaptation
    for n in range(online_iter):

        optimizer.zero_grad()

        pred = us_forward(query_rgb, support_rgb, support_mask, history_mask, index, model)
        pred_soft = torch.softmax(pred, 1)

        history_mask=torch.zeros(1,2,41,41).fill_(0.0)
        valset.history_mask_list = [None] * len(valloader)

        loss_mid = loss_calc_v1(pred, query_mask.squeeze(1), 0)

        pred_inv = us_forward(support_rgb, query_rgb, pred_soft[:,1:,:,:].repeat(1,256,1,1), history_mask, index, model)

        loss = loss_calc_v1(pred_inv, support_mask.squeeze(1), 0)
        loss.backward()
        optimizer.step()
        print(loss_mid.item(), loss.item())

        history_mask=torch.zeros(1,2,41,41).fill_(0.0)
        valset.history_mask_list = [None] * len(valloader)

    pred = us_forward(query_rgb, support_rgb, support_mask, history_mask, index, model)
    _, pred_label = torch.max(pred, 1)

    t_period = time.time() - t_start   
    print(i_iter, t_period/(i_iter+1))
    inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
    for j in range(query_mask.shape[0]):#batch size
        all_inter += inter_list[j]
        all_union += union_list[j]

    cv2.imwrite(os.path.join(checkpoint_dir, 'pred_img', str(index[0].data.numpy()) + '_c.jpg'), (query.squeeze().numpy())[:,:,::-1])
    cv2.imwrite(os.path.join(checkpoint_dir, 'pred_img', str(index[0].data.numpy()) + '_n.jpg'), pred_label.squeeze().data.cpu().numpy()*255)

IOU = all_inter / all_union
print(IOU)

mean_iou = np.mean(IOU)
print('IOU:%.4f' % (mean_iou))


