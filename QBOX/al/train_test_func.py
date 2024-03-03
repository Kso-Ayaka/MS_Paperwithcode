import time
import datetime
import pickle
import numpy as np

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from models_da import *
from utils.datasets import *
from utils.utils import *
from config import *

# global params
mixed_precision = False
arc = 'defaultpw'
# Hyperparameters (j-series, 50.5 mAP yolov3-320) evolved by @ktian08 https://github.com/ultralytics/yolov3/issues/310
hyp = {'giou': 1.582,  # giou loss gain
       'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
       'obj_pw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.2635,  # iou training threshold
       'lr0': 9e-5,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.97,  # SGD momentum
       'weight_decay': 0.0004569,  # optimizer weight decay
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)

if 'pw' not in arc:  # remove BCELoss positive weights
    hyp['cls_pw'] = 1.
    hyp['obj_pw'] = 1.

class QBoxParam:
    def __init__(self, dataset):
        self.clsw = 0.5
        self.st_thres = 1e-3 if dataset != 'covid' else 1e-5
        self.posinsw = 3.91
        self.objw = 60
        self.boxrw = 1.5 if dataset not in ['covid', 'city'] else 1.0
        self.neg_cls_loss = True
        self.border_neg = True
        self.cache_img = True
        self.pos_ins_weight = 0.05
        self.da_tradeoff = 1 if dataset == 'city' else 0.1

# Initialize
device = torch_utils.select_device(apex=mixed_precision)

# device = "cuda:0"
def init_model(pkl_path='/data/saved_model/init_da_yolo_coco.pkl',
               cfg='~/yolov3/cfg/yolov3-spp-voc.cfg', parallel=True, parallel_port=9999,
               init_group=True):
    global weights_path
    # Initialize model
    model = Darknet(cfg, arc=arc).to(device)
    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    optimizer.add_param_group({'params': pg1})
    del pg0, pg1

    # check for pkl file
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            pf = pickle.load(f)
        model.load_state_dict(pf['model'])
        optimizer.load_state_dict(pf['optimizer'])
    else:
        # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        cutoff = load_darknet_weights(model, weights_path)

        tp_dict = {'model': model.module.state_dict() if type(
            model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                   'optimizer': optimizer.state_dict()}

        with open(pkl_path, 'wb') as f:
            pickle.dump(tp_dict, f)

    if parallel:
        # Initialize distributed training
        if torch.cuda.device_count() > 1:
            if init_group:
                dist.init_process_group(backend='nccl',  # 'distributed backend'
                                        init_method='tcp://127.0.0.1:' + str(parallel_port),
                                        # distributed training init method
                                        world_size=1,  # number of nodes for distributed training
                                        rank=0)  # distributed training node rank
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    return model, optimizer


def load_voc_model(pt_path='/data/saved_model/saved_ckpt_50.pt',
                   cfg='cfg/yolov3-spp-coco.cfg',
                   parallel=True, parallel_port=9999, init_group=True):
    model = Darknet(cfg, arc=arc).to(device)

    if os.path.exists(pt_path):
        if pt_path.split('.')[-1] == 'pkl':
            with open(pt_path, 'rb') as f:
                chkpt = pickle.load(f)
        else:
            chkpt = torch.load(pt_path, map_location=device)
        res = 1
    else:
        raise ValueError(f"file does not exist: {pt_path}")
    model.load_state_dict(chkpt['model'])

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    optimizer.add_param_group({'params': pg1})
    del pg0, pg1
    optimizer.load_state_dict(chkpt['optimizer'])

    del chkpt

    if parallel:
        # Initialize distributed training
        if torch.cuda.device_count() > 1:
            if init_group:
                dist.init_process_group(backend='nccl',  # 'distributed backend'
                                        init_method='tcp://127.0.0.1:' + str(parallel_port),
                                        # distributed training init method
                                        world_size=1,  # number of nodes for distributed training
                                        rank=0)  # distributed training node rank
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    return model, optimizer, res


def get_gt_dataloader(data='data/coco.data', data_item='source_train', img_size=416, batch_size=32,
                      rect=False, img_weights=False, cache_images=True, shuffle=False,
                      augment=False, data_root='/data/cityscape/coco/images', num_worker=None, fold=0):
    # init_seeds(seed=0)
    assert data_item in ['source_train', 'target_train',
                         'valid'], "data_item  must in ['source_train', 'target_train', 'valid']"
    data_dict = parse_data_cfg(data)
    train_path = data_dict[data_item]
    if fold > 0:
        train_path = train_path.split('.')[0] + "_"+str(fold) + ".txt"
    nc = int(data_dict['classes'])  # number of classes

    # Dataset
    datasetype = 'voc' if data.split('/')[-1] == 'voc_coco.data' else 'oter'

    # Dataset
    if datasetype == 'voc':
        dataset = LoadImagesAndLabels(train_path,
                                      img_size,
                                      batch_size,
                                      augment=augment,
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      image_weights=img_weights,
                                      cache_images=cache_images,
                                      DATA_ROOT=data_root,
                                      data_type='voc' if data.split('/')[-1] == 'voc_coco.data' else 'oter')
    else:
        init_lab = None
        with open(train_path, 'r') as f:
            init_lab = f.read().splitlines()

        dataset = LoadImagesAndLabelsByImgFiles(
            img_files=init_lab,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,  # augmentation hyperparameters
            rect=False,  # rectangular training
            image_weights=False,
            cache_images=cache_images
        )

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min(os.cpu_count(),
                                                             batch_size) if num_worker is None else num_worker,
                                             shuffle=shuffle,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    return nc, dataset, dataloader


def get_gt_loader_by_names(img_names, img_size=416, batch_size=32, num_workers=0, augment=True, cache_images=True):
    # calc initial performance point
    ini_dataset = LoadImagesAndLabelsByImgFiles(
        img_files=img_names,
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,  # augmentation hyperparameters
        rect=False,  # rectangular training
        image_weights=False,
        cache_images=cache_images
    )
    # Dataloader
    ini_dataloader = torch.utils.data.DataLoader(ini_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=False,  # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=ini_dataset.collate_fn)
    return ini_dataloader


def train_mix(model, optimizer, dataloader, tgt_dataloader,
              start_epoch, epochs, nc, batch_size, 
              src2tgt_label_map, save_epoch=tuple(), notest=True, test_dl=None,
              class_weights=None, ins_gain=5, best_save_name='/data/saved_model/best.pt',
              save_prefix='saved_ckpt_', saved_map_dir='/data/saved_model/saved_map.txt',
              verbose=True, save_pt=False):
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    best_fitness = 0
    best_fitness_20 = 0
    if os.path.exists(saved_map_dir):
        open(saved_map_dir, 'w')

    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model, report='summary')  # 'full' or 'summary'
    if class_weights is not None:
        model.class_weights = class_weights
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    # t0 = time.time()
    print('Starting %s for %g epochs...' % ('training', epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # training (source first)
        if dataloader is not None:
            src_iter = iter(dataloader)
            nb = len(dataloader)
        else:
            nb = -1
        tgt_iter = iter(tgt_dataloader)
        nbt = len(tgt_dataloader)
        end_src = False
        end_tgt = False

        print(f"running epoch {epoch + 1}...")
        batch_count = 0
        src_total_batch = nb
        tgt_total_batch = nbt

        ######################### SOURCE DOMAIN ##########################################
        while batch_count < max(nb, nbt):
            batch_count += 1
            if batch_count <= nb:
                try:
                    imgs, targets, paths, _ = next(src_iter)
                except StopIteration:
                    end_src = True
            if end_src:
                end_src = False
                src_iter = iter(dataloader)
                imgs, targets, paths, _ = next(src_iter)
            if not end_src:
                imgs = imgs.to(device)
                assert src2tgt_label_map is not None
                tgt_mask = [True] * targets.shape[0]
                for ti, target in enumerate(targets):
                    if target[1] >= 0:  # non-background, non-outlier classes
                        if int(target[1]) in src2tgt_label_map.keys():
                            target[1] = src2tgt_label_map[int(target[1])]
                        else:
                            tgt_mask[ti] = False
                targets = targets[tgt_mask, :]
                targets = targets.to(device)

                # Run model
                domain_label = torch.ones(imgs.shape[0]).to(device)
                _, pred, da_pred, da_lab = model(imgs, domain_label=domain_label)
                loss, loss_items = compute_loss(pred, targets, model)

                # compute DA loss
                mask_list = build_obj_target_mask(targets, batch_size=imgs.shape[0])
                for da_i in range(len(da_pred)):
                    base_prob1 = F.log_softmax(da_pred[da_i], dim=1)
                    total_da_loss = None
                    if da_i > 0:  # instance DA
                        total_da_loss = F.nll_loss(base_prob1, mask_list[da_i - 1].long().to(device),
                                                    ignore_index=0, reduction='mean') * ins_gain
                    else:  # image level DA
                        total_da_loss = F.nll_loss(base_prob1, da_lab[da_i], reduction='mean')
                        if len(targets) == 0:
                            total_da_loss *= 0
                    loss += total_da_loss
                    loss_items = torch.cat((loss_items, total_da_loss.reshape(1))).detach()

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Compute gradient
                loss.backward()

            ######################### TARGET DOMAIN ##########################################
            if not end_tgt:
                try:
                    imgs, targets, paths, _ = next(tgt_iter)
                except StopIteration:
                    end_tgt = True
            if end_tgt:
                end_tgt = False
                tgt_iter = iter(tgt_dataloader)
                imgs, targets, paths, _ = next(tgt_iter)
            if True:
                imgs = imgs.to(device)
                targets = targets.to(device)

                # Run model
                domain_label = torch.zeros(imgs.shape[0]).to(device)
                _, pred, da_pred, da_lab = model(imgs, domain_label=domain_label)
                loss, loss_items = compute_loss(pred, targets, model)

                mask_list = build_obj_target_mask(targets, batch_size=imgs.shape[0])
                for da_i in range(len(da_pred)):
                    base_prob1 = F.log_softmax(da_pred[da_i], dim=1)
                    total_da_loss = None
                    if da_i > 0:  # instance DA
                        total_da_loss = F.nll_loss(base_prob1, (1 - mask_list[da_i - 1]).long().to(device),
                                                    ignore_index=1,
                                                    reduction='mean') * ins_gain
                    else:  # image level DA
                        total_da_loss = F.nll_loss(base_prob1, da_lab[da_i], reduction='mean')
                    loss += total_da_loss
                    loss_items = torch.cat((loss_items, total_da_loss.reshape(1))).detach()

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Compute gradient
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        # Update scheduler
        scheduler.step()

        if not notest and epoch >= epochs * 0.80:
            # if not notest:
            assert test_dl is not None
            with torch.no_grad():
                results, maps = test(model,
                                     dataloader=test_dl,
                                     nc=nc,
                                     batch_size=batch_size,
                                     img_size=416,
                                     iou_thres=0.5,
                                     conf_thres=0.1,
                                     nms_thres=0.5)  # results[2] map

            with open(saved_map_dir, 'a') as file:
                file.write(str(results))
                file.write(os.linesep)

            fitness = results[2]  # mAP
            if fitness > best_fitness and save_pt:
                best_fitness = fitness

                # save best model
                with open(best_save_name, 'wb') as file:
                    # Create checkpoint
                    chkpt = {'model': model.module.state_dict() if type(
                        model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    # Save last checkpoint
                    torch.save(chkpt, file)

        if epoch + 1 in save_epoch:
            with open(f'/data/saved_model/{save_prefix}{epoch + 1}.pt', 'wb') as file:
                # Create checkpoint
                chkpt = {'model': model.module.state_dict() if type(
                    model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': optimizer.state_dict()}

                # Save last checkpoint
                torch.save(chkpt, file)

    return model, best_fitness


def train_mix_partial(model, optimizer, dataloader, tgt_dataloader, queried_dataloader,
                      start_epoch, epochs, nc, batch_size, partial_loss_hyp,
                      src2tgt_label_map=None, test_dl=None,
                      class_weights=None, ins_gain=5,
                      saved_map_dir='/data/saved_model/saved_map.txt'):
    # init_seeds(seed=0)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1
    assert src2tgt_label_map is not None
    best_fitness = 0
    if os.path.exists(saved_map_dir):
        open(saved_map_dir, 'w')

    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model, report='summary')  # 'full' or 'summary'
    if class_weights is not None:
        model.class_weights = class_weights
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    # t0 = time.time()
    print('Starting %s for %g epochs...' % ('training', epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        # start_time = time.clock()
        # training (source first)
        if dataloader is not None:
            src_iter = iter(dataloader)
            nb = len(dataloader)
        else:
            nb = -1
        tgt_iter = iter(tgt_dataloader)
        nbt = len(tgt_dataloader)
        q_iter = iter(queried_dataloader)
        nbq = len(queried_dataloader)
        end_src = False
        end_tgt = False
        end_q = False
        src_dtype = 1  # ini data

        print(f"running epoch {epoch + 1}...")
        batch_count = 0
        src_total_batch = nb
        tgt_total_batch = nbt

        ######################### SOURCE DOMAIN ##########################################
        while batch_count < max((nb + nbq, nbt)):
            batch_count += 1
            if not end_src:
                try:
                    imgs, targets, paths, _ = next(src_iter)
                except StopIteration:
                    end_src = True
                    imgs, targets, paths, _ = next(q_iter)
                    src_dtype = 2
            elif not end_q:
                try:
                    imgs, targets, paths, _ = next(q_iter)
                    src_dtype = 2
                except StopIteration:
                    end_q = True
            if (not end_src) or (not end_q):
                if len(targets) == 0:
                    print(paths)
                    print(src_dtype)
                    print(f"empty targets for source domain in {batch_count} batch. exit.")
                    print(queried_dataloader.dataset.get_item_by_path(paths[0]))
                    # exit()
                imgs = imgs.to(device)

                # Run model
                domain_label = torch.ones(imgs.shape[0]).to(device)
                inf_out, pred, da_pred, da_lab = model(imgs, domain_label=domain_label)

                ############################ init data ###################################
                # COMPUTE LOSS FOR DIFFERENT SRC DATA TYPES!
                if src_dtype == 1:  # init dataset, fully supervised
                    tgt_mask = [True] * targets.shape[0]
                    for ti, target in enumerate(targets):
                        if target[1] >= 0:  # non-background, non-outlier classes
                            if int(target[1]) in src2tgt_label_map.keys():
                                target[1] = src2tgt_label_map[int(target[1])]
                            else:
                                tgt_mask[ti] = False
                    targets = targets[tgt_mask, :]
                    targets = targets.to(device)

                    loss, loss_items = compute_loss(pred, targets, model)

                    # compute DA loss
                    mask_list = build_obj_target_mask(targets, batch_size=imgs.shape[0])
                    for da_i in range(len(da_pred)):
                        base_prob1 = F.log_softmax(da_pred[da_i], dim=1)
                        if da_i > 0:  # instance DA
                            total_da_loss = F.nll_loss(base_prob1, mask_list[da_i - 1].long().to(device),
                                                       ignore_index=0, reduction='mean') * ins_gain
                        else:  # image level DA
                            total_da_loss = F.nll_loss(base_prob1, da_lab[da_i], reduction='mean')
                            if len(targets) == 0:
                                total_da_loss *= 0
                            # total_da_loss2 = F.nll_loss(base_prob1, da_lab[da_i], reduce=False)
                            # tp = torch.sum(total_da_loss2)    # equal

                        loss += total_da_loss
                        loss_items = torch.cat((loss_items, total_da_loss.reshape(1))).detach()

                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, ending training ', loss_items)
                        return results
                ################################# partially labeled #################################################################
                else:  # partially labeled
                    # split targets into pos, neg, outliers
                    tgt_pos_mask = [False] * targets.shape[0]
                    tgt_neg_mask = [False] * targets.shape[0]
                    tgt_out_mask = [False] * targets.shape[0]
                    for ti, target in enumerate(targets):
                        if target[1] >= 0:  # non-background, non-outlier classes
                            if int(target[1]) in src2tgt_label_map.keys():
                                target[1] = src2tgt_label_map[int(target[1])]
                                tgt_pos_mask[ti] = True
                            else:
                                # not queried outlier
                                raise ValueError(f"Unexpected label {target[1]} in {paths}")
                        elif target[1] == -1:  # neg class
                            tgt_neg_mask[ti] = True
                        elif target[1] == -2:  # out class
                            tgt_out_mask[ti] = True
                        else:
                            raise (f"un-recognized class number: {target[1]}")

                    gt_tar_tmp = targets[tgt_pos_mask]
                    targets = targets.to(device)
                    loss, loss_items = compute_partial_loss(p=pred, pos_targets=targets[tgt_pos_mask],
                                                            neg_targets=targets[tgt_neg_mask],
                                                            outlier_targets=targets[tgt_out_mask], model=model,
                                                            partial_loss_hyp=partial_loss_hyp)
                    # compute DA loss
                    if len(targets) > 0:
                        mask_list = build_obj_target_mask(targets[np.asarray(tgt_pos_mask) | np.asarray(tgt_out_mask)],
                                                          batch_size=imgs.shape[0])
                    else:
                        mask_list = build_obj_target_mask(targets, batch_size=imgs.shape[0])
                    img_level_mask = build_img_level_mask(targets=targets, batch_size=imgs.shape[0])
                    for da_i in range(len(da_pred)):
                        base_prob1 = F.log_softmax(da_pred[da_i], dim=1)
                        total_da_loss = None
                        if da_i > 0:  # instance DA
                            total_da_loss = F.nll_loss(base_prob1, mask_list[da_i - 1].long().to(device),
                                                       ignore_index=0, reduction='mean') * ins_gain
                        else:  # image level DA
                            img_label_mask = da_lab[da_i]
                            img_label_mask[img_level_mask] = -1
                            total_da_loss = F.nll_loss(base_prob1, da_lab[da_i], ignore_index=-1, reduction='mean')
                            # total_da_loss2 = F.nll_loss(base_prob1, da_lab[da_i], reduce=False)
                            # tp = torch.sum(total_da_loss2)    # equal

                        loss += total_da_loss
                        loss_items = torch.cat((loss_items, total_da_loss.reshape(1))).detach()
                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, ending training ', loss_items)
                        return results

                # Compute gradient
                loss.backward()

            ######################### TARGET DOMAIN ##########################################
            if not end_tgt:
                try:
                    imgs, targets, paths, _ = next(tgt_iter)
                except StopIteration:
                    end_tgt = True
            if end_tgt:
                end_tgt = False
                tgt_iter = iter(tgt_dataloader)
                imgs, targets, paths, _ = next(tgt_iter)
            if True:
                if len(targets) == 0:
                    print(paths)
                    print(f"empty targets for target domain in {batch_count} batch. exit.")
                    print(tgt_dataloader.dataset.get_item_by_path(paths[0]))
                    # exit()
                imgs = imgs.to(device)
                targets = targets.to(device)

                # Run model
                domain_label = torch.zeros(imgs.shape[0]).to(device)
                inf_out, pred, da_pred, da_lab = model(imgs, domain_label=domain_label)
                loss, loss_items = compute_loss(pred, targets, model)

                # compute DA loss
                mask_list = build_obj_target_mask(targets, batch_size=imgs.shape[0])
                for da_i in range(len(da_pred)):
                    base_prob1 = F.log_softmax(da_pred[da_i], dim=1)
                    total_da_loss = None
                    if da_i > 0:  # instance DA
                        total_da_loss = F.nll_loss(base_prob1, (1 - mask_list[da_i - 1]).long().to(device),
                                                   ignore_index=1,
                                                   reduction='mean') * ins_gain
                    else:  # image level DA
                        total_da_loss = F.nll_loss(base_prob1, da_lab[da_i], reduction='mean')
                    loss += total_da_loss
                    loss_items = torch.cat((loss_items, total_da_loss.reshape(1))).detach()

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                loss.backward()

            if (end_src & end_q) + end_tgt == 1:
                # accumulate 64
                if batch_count % 2 == 0 or True:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                # batch size is 64
                optimizer.step()
                optimizer.zero_grad()
        # Update scheduler
        scheduler.step()

        if epoch >= epochs * 0.80:
            assert test_dl is not None
            with torch.no_grad():
                results, maps = test(model,
                                     dataloader=test_dl,
                                     nc=nc,
                                     batch_size=batch_size,
                                     img_size=416,
                                     iou_thres=0.5,
                                     conf_thres=0.1,
                                     nms_thres=0.5)  # results[2] map

            with open(saved_map_dir, 'a') as file:
                file.write(str(results))
                file.write(os.linesep)

            with open(saved_map_dir.split('.')[0] + '_aps.txt', 'a') as file:
                file.write(str(maps))
                file.write(os.linesep)

    return model, best_fitness


def test(model,
         dataloader,
         nc,
         batch_size=16,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.01,
         nms_thres=0.5,
         data_root="/data/cityscape/coco/images"):
    # init_seeds(seed=0)
    seen = 0
    model.eval()

    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        # if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
        #     plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Run model
        domain_label = torch.zeros(imgs.shape[0]).to(device)
        inf_out, train_out, _ = model(imgs, domain_label)  # inference and training outputs

        # Compute loss
        # if hasattr(model, 'hyp'):  # if model has loss hyperparameters
        #     loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    # if verbose and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


def train_plain(model, optimizer, dataloader,
              epochs, nc, src2tgt_label_map=None, clsw=0):
    init_seeds(seed=0)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    # scheduler.last_epoch = start_epoch - 1

    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model, report='summary')  # 'full' or 'summary'
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    # t0 = time.time()
    print('Starting %s for %g epochs...' % ('training', epochs))
    for epoch in range(epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # training (source first)
        if dataloader is not None:
            src_iter = iter(dataloader)
            nb = len(dataloader)
        else:
            nb = -1

        print(f"running epoch {epoch + 1}...")
        batch_count = 0

        ######################### SOURCE DOMAIN ##########################################
        while batch_count < nb:
            batch_count += 1
            if batch_count <= nb:
                try:
                    imgs, targets, paths, _ = next(src_iter)
                except StopIteration:
                    end_src = True
            if 1:
                imgs = imgs.to(device)
                if src2tgt_label_map is not None:
                    tgt_mask = [True] * targets.shape[0]
                    for ti, target in enumerate(targets):
                        if target[1] >= 0:  # non-background, non-outlier classes
                            if int(target[1]) in src2tgt_label_map.keys():
                                target[1] = src2tgt_label_map[int(target[1])]
                            else:
                                tgt_mask[ti] = False
                    targets = targets[tgt_mask, :]
                targets = targets.to(device)

                # Run model
                domain_label = torch.ones(imgs.shape[0]).to(device)
                _, pred, da_pred, da_lab = model(imgs, domain_label=domain_label)
                # return order: img, scale13, 26, 52

                # Compute detection loss
                # print(imgs.shape)
                # print(targets.shape)
                # print(pred[0].shape)
                loss, loss_items = compute_loss(pred, targets, model, clsw=clsw)
                # Scale loss by nominal batch_size of 64
                # if end_src + end_tgt != 1:
                #     loss *= batch_size / 64

                if 0:
                    # compute DA loss
                    mask_list = build_obj_target_mask(targets, batch_size=imgs.shape[0])
                    for da_i in range(len(da_pred)):
                        base_prob1 = F.log_softmax(da_pred[da_i], dim=1)
                        total_da_loss = None
                        if da_i > 0:  # instance DA
                            # custom function to filter out non-object loss
                            # DA_img_loss_cls1 = F.nll_loss(base_prob1, da_lab[da_i], reduce=False, reduction='None')  # bs, scale, scale
                            # total_da_loss = torch.sum(DA_img_loss_cls1*(mask_list[da_i - 1].to(device))) / torch.sum(mask_list[da_i - 1]).to(device)
                            # MASK loss for each example. Only retain the object grid
                            total_da_loss = F.nll_loss(base_prob1, mask_list[da_i - 1].long().to(device),
                                                       ignore_index=0, reduction='mean') * ins_gain
                        else:  # image level DA
                            total_da_loss = F.nll_loss(base_prob1, da_lab[da_i], reduction='mean')
                            if len(targets) == 0:
                                total_da_loss *= 0
                            # total_da_loss2 = F.nll_loss(base_prob1, da_lab[da_i], reduce=False)
                            # tp = torch.sum(total_da_loss2)    # equal

                        loss += total_da_loss
                        loss_items = torch.cat((loss_items, total_da_loss.reshape(1))).detach()
                else:
                    for da_i in range(len(da_pred)):
                        base_prob1 = F.log_softmax(da_pred[da_i], dim=1)
                        total_da_loss = F.nll_loss(base_prob1, da_lab[da_i])
                        loss += total_da_loss * 0
                        loss_items = torch.cat((loss_items, total_da_loss.reshape(1) * 0)).detach()

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Compute gradient
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # print progress
            # print(f"\rsource data: {min(1.0, batch_count / src_total_batch):.3f}\t target data: {min(1.0, batch_count / tgt_total_batch):.3f}\t\t time elaps: {datetime.timedelta(seconds=int(time.clock() - start_time))}", end='')

        # Update scheduler
        scheduler.step()
        with open(f'/data/saved_model/covid_xls_pretrain.pt', 'wb') as file:
            # Create checkpoint
            chkpt = {'model': model.module.state_dict() if type(
                model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, file)
            # pickle.dump(pklf, chkpt)

    return model
