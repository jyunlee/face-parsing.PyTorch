#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(respth, 'cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    no_iter = str(int(cp.split('_')[0]))
    org_respth = respth[:]
    respth = os.path.join(respth, no_iter)

    if not os.path.exists(respth):
        os.makedirs(respth)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ''' added '''
    cropsize = [448, 448]
    n_img_per_gpu = 16

    data_root = '/home/jihyun/workspace/face_parsing/dataset/CelebAMask-HQ/'
    ds = FaceMask(data_root, cropsize=cropsize, mode='val')
    dl = DataLoader(ds, batch_size=16, shuffle=False, drop_last=True)

    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    score_thres = 0.7
    ignore_idx = -100

    loss_avg = []
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    with torch.no_grad():
        
        for i, sample in enumerate(dl):
            im, lb = sample
            im = im.cuda()
            lb = lb.cuda()
            lb = torch.squeeze(lb, 1)
            out, out16, out32 = net(im)
            lossp = LossP(out, lb)
            loss2 = Loss2(out16, lb)
            loss3 = Loss3(out32, lb)
            loss = lossp + loss2 + loss3

            loss_avg.append(loss.item())
        
        loss_avg = sum(loss_avg) / len(loss_avg)

        f = open(osp.join(org_respth, 'loss.log'), 'a')
        f.write(' eval_loss: ' + str(loss_avg) + '\n')
        f.close()

        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out, out16, out32 = net(img)

            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))
        

if __name__ == "__main__":
    setup_logger('./res')
    evaluate()
