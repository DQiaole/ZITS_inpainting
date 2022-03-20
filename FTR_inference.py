import argparse

from tqdm import tqdm

from src.FTR_trainer import *
from src.config import Config
from src.inpainting_metrics import get_inpainting_metrics
from src.utils import SampleEdgeLineLogits


def main():

    args, config = load_config()
    rank = torch.distributed.get_rank() if config.DDP else 0
    torch.cuda.set_device(rank)
    gpu = torch.device("cuda")

    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    inpaint_model = DefaultInpaintingTrainingModule(config, gpu=gpu, rank=rank, **kwargs).to(gpu)
    data = torch.load(os.path.join(config.PATH, 'InpaintingModel_best_gen_HR.pth'), map_location='cpu')
    inpaint_model.generator.load_state_dict(data['generator'])
    inpaint_model.str_encoder.load_state_dict(data['str_encoder'])
    if config.min_sigma is None:
        min_sigma = 2.0
    else:
        min_sigma = config.min_sigma
    if config.max_sigma is None:
        max_sigma = 2.5
    else:
        max_sigma = config.max_sigma
    test_dataset = DynamicDataset(config.TEST_FLIST, mask_path=None, pos_num=config.rel_pos_num,
                                  batch_size=config.BATCH_SIZE, augment=False, training=False,
                                  test_mask_path=config.TEST_MASK_FLIST, eval_line_path=config.eval_line_path,
                                  add_pos=config.use_MPE, input_size=config.EVAL_SIZE,
                                  min_sigma=min_sigma, max_sigma=max_sigma)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                             batch_size=8, num_workers=16)

    inpaint_model.eval()

    test_path = args.path + '/test'
    os.makedirs(test_path, exist_ok=True)

    with torch.no_grad():
        for items in tqdm(test_loader):
            items['image'] = items['image'].to(gpu)
            items['mask'] = items['mask'].to(gpu)
            items['rel_pos'] = items['rel_pos'].to(gpu)
            items['direct'] = items['direct'].to(gpu)
            b, _, _, _ = items['image'].size()
            edge_pred, line_pred = SampleEdgeLineLogits(inpaint_model.transformer,
                                                        context=[items['img_256'][:b, ...],
                                                                 items['edge_256'][:b, ...],
                                                                 items['line_256'][:b, ...]],
                                                        mask=items['mask_256'][:b, ...].clone(),
                                                        iterations=5,
                                                        add_v=0.05, mul_v=4,
                                                        device=gpu)
            edge_pred, line_pred = edge_pred[:b, ...].detach().to(torch.float32), \
                                   line_pred[:b, ...].detach().to(torch.float32)
            if config.fix_256 is None or config.fix_256 is False:
                edge_pred = inpaint_model.structure_upsample(edge_pred)[0]
                edge_pred = torch.sigmoid((edge_pred + 2) * 2)
                line_pred = inpaint_model.structure_upsample(line_pred)[0]
                line_pred = torch.sigmoid((line_pred + 2) * 2)

            items['edge'][:b, ...] = edge_pred.detach()
            items['line'][:b, ...] = line_pred.detach()
            items['edge'] = items['edge'].to(gpu)
            items['line'] = items['line'].to(gpu)

            # eval
            items = inpaint_model(items)
            outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
            # save
            outputs_merged *= 255.0
            outputs_merged = outputs_merged.permute(0, 2, 3, 1).int().cpu().numpy()
            for img_num in range(b):
                cv2.imwrite(test_path + '/' + items['name'][img_num], outputs_merged[img_num, :, :, ::-1])

    our_metric = get_inpainting_metrics(test_path, config.gt256test_flist, None, fid_test=True)
    print("PSNR: %f, SSIM: %f, FID: %f" % (
        float(our_metric['psnr']), float(our_metric['ssim']), float(our_metric['fid'])))


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./ckpt/LAMA_places2',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--config_file', type=str, default='./config_list/config_LAMA_MPE.yml',
                        help='The config file of each experiment ')

    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--local_rank', type=int, default=-1, help='the id of this machine')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--AMP', action='store_true', help='Automatic Mixed Precision')
    parser.add_argument('--DDP', action='store_true', help='Automatic Mixed Precision')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # load config file
    config = Config(config_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids

    return args, config


if __name__ == "__main__":
    main()
