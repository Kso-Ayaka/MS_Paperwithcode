import os
import sys
import time
import random
import logging
import argparse
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
from collections import OrderedDict

import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
from msadapter.pytorch.utils.data import DataLoader
from msadapter.torchvision import datasets, transforms
from msadapter.torchvision.transforms.functional import InterpolationMode
 
import mindspore as ms

from msadapter.pytorch.utils.data import Dataset

from datasets import *
import moco.builder


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def valid_one_epoch(epoch, model, val_loader, device, args):
    model.eval()

    d = {}
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))
    for step, (imgs, targets, _id) in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)
        feat = model(imgs)
        feat = F.normalize(feat).view(-1, args.num_attr, feat.size(1))

        for i, j in zip(_id, feat):
            d[i.item()] = j.cpu().numpy()

    return d


def build_dataloader(args):
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=args.resize_size),
        torchvision.transforms.CenterCrop(size=args.crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    dataset = CUB200Id(root=args.root, train=True, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return loader


def main(args):
    device = torch.device('cuda')

    val_loader = build_dataloader(args)
    model = moco.builder.MoCo_CLIP(256, 2048, 0.2, args.num_attr)
    model.to(device)

    eval_path = f'gkb/cub200/{args.num_attr}/checkpoint_0299.pth.tar'

    checkpoint = torch.load(eval_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        state_dict[k] = v
    model.load_state_dict(state_dict)
    with torch.no_grad():
        d = valid_one_epoch(-1, model, val_loader, device, args)
    pkl.dump(d, open(f'gkb/cub200/{args.num_attr}/gkb.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/4T/dataset/CUB_200_2011/', type=str)
    parser.add_argument('--dataset', default='vegfru', choices=['cub200', 'aircraft', 'food101', 'dog', 'vegfru'])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--resize_size', default=256, type=int)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_attr', default=4, type=int)
    #parser.add_argument('--info', default='clip1', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)

    main(args)
