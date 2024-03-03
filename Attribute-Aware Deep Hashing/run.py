import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
from msadapter.pytorch.utils.data import DataLoader
from msadapter.torchvision import datasets, transforms
from msadapter.torchvision.transforms.functional import InterpolationMode
import mindspore as ms
import argparse
from loguru import logger
from data.data_loader import load_data
import os
import numpy as np
import random
import A_2_Net_train as A_2_Net
import A_2_Net_finetune


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def run():
    # seed_everything(68)
    seed_everything(3)
    args = load_config()
    if args.arch != 'test':
        if not os.path.exists('logs/' + args.info):
            os.makedirs('logs/' + args.info)
        # logger.add('logs/' + args.info + '/-{time}_' + args.arch + '_' + args.dataset + '.log', rotation='500 MB', level='INFO')
        # logger.add('logs/' + args.info + '/' + str(args.code_length) + '/food101-finet.log')
        logger.add('logs/' + args.info + '/' + str(args.code_length) + '/' + 'train.log')
        # logger.add('logs/'+args.info+'/'+str(args.code_length)+'/resonly-test.log')
        logger.info(args)

    torch.backends.cudnn.benchmark = True

    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_samples,
        args.batch_size,
        args.num_workers,
    )

    if args.arch == 'a2net':
        net_arch = A_2_Net
    elif args.arch == 'a2net_ft':
        net_arch = A_2_Net_finetune

    for code_length in args.code_length:
        mAP = net_arch.train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args)
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size.(default: 16)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='Weight Decay.(default: 1e-4)')
    parser.add_argument('--optim', default='Adam', type=str,
                        help='Optimizer')
    parser.add_argument('--code-length', default='12,24,32,48', type=str,
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-iter', default=2, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=5, type=int,
                        help='Number of epochs.(default: 5)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of loading data threads.(default: 4)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--arch', default='baseline',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--lr-step', default='40, 50', type=str,
                        help='lr decrease step.(default: 40,50)')
    parser.add_argument('--align-step', default=50, type=int,
                        help='Step of start aligning.(default: 50)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--quan-loss', action='store_true',
                        help='Using quan_loss')
    parser.add_argument('--lambd-sp', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd-ch', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--momen', default=0.9, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Using SGD nesterov')
    parser.add_argument('--num_classes', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num_train', default=2000, type=int)
    parser.add_argument('--margin', default=4, type=float)
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
    # args.device = torch.device("cuda:0")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    args.lr_step = list(map(int, args.lr_step.split(',')))

    return args


if __name__ == '__main__':
    run()
