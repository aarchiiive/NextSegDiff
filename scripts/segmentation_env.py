
import sys
import random
sys.path.append(".")
from guided_diffusion.utils import staple

from tqdm import tqdm
import pandas as pd
import glob
import numpy
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import math
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from guided_diffusion.utils import staple
import argparse

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz


def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one ㅔㅛgradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred,true_mask_p,threshold = (0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--org_pth") # AMOS2D/imagesVa
    argParser.add_argument("--inp_pth") # AMOS2D/preds/diff_100_ens_5
    argParser.add_argument("--out_pth") # AMOS2D/labelsVa
    argParser.add_argument("--data_name")
    
    args = argParser.parse_args()
    mix_res = (0,0)
    num = 0
    
    org_path = args.org_pth
    pred_path = args.inp_pth
    gt_path = args.out_pth
    
    res_path = os.path.join(pred_path, "dice")
    if not os.path.isdir(res_path): os.mkdir(res_path)
    pred_images = glob.glob(os.path.join(pred_path, "*.jpg"))
    pred_images = [p for p in pred_images if "output" in p]
    
    results_list = []
    
    for pred_img in tqdm(pred_images):
        num += 1
        org_name = '_'.join(os.path.basename(pred_img).split('_')[:4]) + '.png'
        org = Image.open(os.path.join(org_path, org_name)).convert('RGB')
        org = torchvision.transforms.PILToTensor()(org)
        org = torch.unsqueeze(org, 0).float()
        org = torchvision.transforms.Resize((256,256))(org)
        
        pred = Image.open(pred_img).convert('L')
        gt_name = '_'.join(os.path.basename(pred_img).split('_')[:4]) + '.png'
        gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')
        pred = torchvision.transforms.PILToTensor()(pred)
        pred = torch.unsqueeze(pred,0).float() 
        pred = pred / pred.max()
        # if args.debug:
        #     print('pred max is', pred.max())
        #     vutils.save_image(pred, fp = os.path.join('./results/' + str(ind)+'pred.jpg'), nrow = 1, padding = 10)
        gt = torchvision.transforms.PILToTensor()(gt)
        gt = torchvision.transforms.Resize((256,256))(gt)
        gt = torch.unsqueeze(gt,0).float() / 255.0
        # if args.debug:
        #     vutils.save_image(gt, fp = os.path.join('./results/' + str(ind)+'gt.jpg'), nrow = 1, padding = 10)
        temp = eval_seg(pred, gt)
        iou, dice = temp
        mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
        
        merged = th.cat((org, pred.repeat(1, 3, 1, 1), gt.repeat(1, 3, 1, 1)), dim=2).squeeze().permute(2, 1, 0).numpy()
        merged = Image.fromarray(merged.astype(np.uint8) * 255)
        
        draw = ImageDraw.Draw(merged)
        draw.text((10, 10), f"IoU: {iou:.4f} Dice: {dice:.4f}", 255) 
        merged.save(os.path.join(res_path, gt_name.split(".")[0]+f'_iou_{iou:.4f}_dice_{dice:.4f}.png'))
        
        results_list.append([gt_name, round(iou, 6), round(dice, 6)])
        
    iou, dice = tuple([a/num for a in mix_res])
    print('iou is',iou)
    print('dice is', dice)
    
    df = pd.DataFrame(results_list, columns=['gt_name', 'iou', 'dice'])

    csv_path = os.path.join(res_path, 'results.csv')
    df.to_csv(csv_path, index=False)

    print(f'Results saved to: {csv_path}')

if __name__ == "__main__":
    main()
