import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.FTR_trainer import ZITS, LaMa
from src.config import Config


def main_worker(gpu, args):
    rank = args.node_rank * args.gpus + gpu
    torch.cuda.set_device(gpu)

    if args.DDP:
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=args.world_size,
                                rank=rank,
                                group_name='mtorch')

    # load config file
    config = Config(args.config_path)
    config.MODE = 1
    config.nodes = args.nodes
    config.gpus = args.gpus
    config.GPU_ids = args.GPU_ids
    config.DDP = args.DDP
    if config.DDP:
        config.world_size = args.world_size
    else:
        config.world_size = 1

    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    if args.lama:
        model = LaMa(config, gpu, rank)
    else:
        model = ZITS(config, gpu, rank)

    # model training
    if rank == 0:
        config.print()
        print('\nstart training...\n')
    model.train()

    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default=None,
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--config_file', type=str, default='./config_list/config_LAMA_MPE_HR.yml',
                        help='The config file of each experiment ')
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
    parser.add_argument('--DDP', action='store_true', help='DDP')
    parser.add_argument('--lama', action='store_true', help='train the lama first')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    os.makedirs(args.path, exist_ok=True)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile(args.config_file, config_path)  ## Training, always copy

    args.config_path = config_path

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    if args.DDP:
        args.world_size = args.nodes * args.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22323'
    else:
        args.world_size = 1

    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
