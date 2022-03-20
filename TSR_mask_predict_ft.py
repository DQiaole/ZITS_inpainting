import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

from datasets.dataset_TSR import ContinuousEdgeLineDatasetMaskFinetune
from src.TSR_trainer import TrainerConfig, TrainerForEdgeLineFinetune
from src.models import EdgeLineGPT256RelBCE, EdgeLineGPTConfig
from src.utils import set_seed


def main_worker(rank, opts):
    set_seed(42)
    gpu = torch.device("cuda")

    if rank == 0:
        os.makedirs(os.path.dirname(opts.ckpt_path), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(sh)
    logger.propagate = False
    fh = logging.FileHandler(os.path.join(opts.ckpt_path, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Define the dataset
    train_dataset = ContinuousEdgeLineDatasetMaskFinetune(opts.data_path, mask_path=opts.mask_path, is_train=True,
                                                          mask_rates=opts.mask_rates, image_size=opts.image_size)
    test_dataset = ContinuousEdgeLineDatasetMaskFinetune(opts.validation_path, test_mask_path=opts.valid_mask_path,
                                                         is_train=False, image_size=opts.image_size)
    # Define the model
    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head)
    IGPT_model = EdgeLineGPT256RelBCE(model_config)

    iterations_per_epoch = len(train_dataset.image_id_list) // opts.batch_size
    train_epochs = opts.train_epoch
    train_config = TrainerConfig(max_epochs=train_epochs, batch_size=opts.batch_size,
                                 learning_rate=opts.lr, betas=(0.9, 0.95),
                                 weight_decay=0, lr_decay=True,
                                 warmup_iterations=1000,
                                 final_iterations=train_epochs * iterations_per_epoch / opts.world_size,
                                 ckpt_path=opts.ckpt_path, num_workers=12, GPU_ids=opts.GPU_ids,
                                 world_size=opts.world_size,
                                 AMP=opts.AMP, print_freq=opts.print_freq)

    trainer = TrainerForEdgeLineFinetune(IGPT_model, train_dataset, test_dataset, train_config, gpu, rank,
                      iterations_per_epoch, logger=logger)
    loaded_ckpt = trainer.load_checkpoint(opts.resume_ckpt)
    trainer.train(loaded_ckpt)
    print("Finish the training ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='places2_continous_edgeline', help='The name of this exp')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--data_path', type=str,
                        default=None, help='Indicate where is the training set')
    parser.add_argument('--mask_path', type=list, default=['irregular_mask_list.txt', 'coco_mask_list.txt'])
    parser.add_argument('--mask_rates', type=list, default=[0.4, 0.8, 1.0],
                        help='irregular rate, coco rate, addition rate')
    parser.add_argument('--batch_size', type=int, default=32, help='16*8 maybe suitable for V100')
    parser.add_argument('--train_epoch', type=int, default=100, help='how many epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='While training, the freq of printing log')
    parser.add_argument('--validation_path', type=str, default=None, help='where is the validation set of ImageNet')
    parser.add_argument('--valid_mask_path', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=256, help='input sequence length = image_size*image_size')
    parser.add_argument('--resume_ckpt', type=str, default='latest.pth', help='start from where, the default is latest')
    # Define the size of transformer
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--lr', type=float, default=6e-4)
    # DDP+AMP
    parser.add_argument('--DDP', action='store_true', help='using DDP rather than normal data parallel')
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--AMP', action='store_true', help='Automatic Mixed Precision')
    parser.add_argument('--local_rank', type=int, default=-1, help='the id of this machine')

    opts = parser.parse_args()
    opts.ckpt_path = os.path.join(opts.ckpt_path, opts.name)
    opts.resume_ckpt = os.path.join(opts.ckpt_path, opts.resume_ckpt)
    os.makedirs(opts.ckpt_path, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.GPU_ids

    opts.world_size = opts.nodes * opts.gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12340'

    dist.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    main_worker(rank, opts)
