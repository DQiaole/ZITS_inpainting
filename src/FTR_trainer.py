import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.dataset_FTR import *
from src.models.FTR_model import *
from .inpainting_metrics import get_inpainting_metrics
from .utils import Progbar, create_dir, stitch_images, SampleEdgeLineLogits


class LaMa:
    def __init__(self, config, gpu, rank, test=False):
        self.config = config
        self.device = gpu
        self.global_rank = rank

        self.model_name = 'inpaint'

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = LaMaInpaintingTrainingModule(config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu)

        self.train_dataset = ImgDataset(config.TRAIN_FLIST, config.INPUT_SIZE, config.MASK_RATE, config.TRAIN_MASK_FLIST,
                                        augment=True, training=True, test_mask_path=None)
        if config.DDP:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config.world_size,
                                                    rank=self.global_rank, shuffle=True)
        # else:
        #     self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)
        self.val_dataset = ImgDataset(config.VAL_FLIST, config.INPUT_SIZE, mask_rates=None, mask_path=None, augment=False,
                                      training=False, test_mask_path=config.TEST_MASK_FLIST)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.val_path = os.path.join(config.PATH, 'validation')
        create_dir(self.val_path)

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

        self.best = float("inf") if self.inpaint_model.best is None else self.inpaint_model.best

    def save(self):
        if self.global_rank == 0:
            self.inpaint_model.save()

    def train(self):
        if self.config.DDP:
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE // self.config.world_size,
                                      num_workers=12, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE, num_workers=12, shuffle=True)

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset) // self.config.world_size

        if total == 0 and self.global_rank == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while keep_training:
            epoch += 1
            if self.config.DDP:
                self.train_sampler.set_epoch(epoch + 1)  # Shuffle each epoch
            epoch_start = time.time()
            if self.global_rank == 0:
                print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'loss_scale'],
                              verbose=1 if self.global_rank == 0 else 0)

            for _, items in enumerate(train_loader):
                self.inpaint_model.train()

                items['image'] = items['image'].to(self.device)
                items['mask'] = items['mask'].to(self.device)

                # train
                outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                if self.config.No_Bar:
                    pass
                else:
                    progbar.add(len(items['image']),
                                values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 1 and self.global_rank == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 1 and self.global_rank == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 1:
                    if self.global_rank == 0:
                        print('\nstart eval...\n')
                        print("Epoch: %d" % epoch)
                    psnr, ssim, fid = self.eval()
                    if self.best > fid and self.global_rank == 0:
                        self.best = fid
                        print("current best epoch is %d" % epoch)
                        print('\nsaving %s...\n' % self.inpaint_model.name)
                        raw_model = self.inpaint_model.generator.module if \
                            hasattr(self.inpaint_model.generator, "module") else self.inpaint_model.generator
                        torch.save({
                            'iteration': self.inpaint_model.iteration,
                            'generator': raw_model.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_gen.pth'))
                        raw_model = self.inpaint_model.discriminator.module if \
                            hasattr(self.inpaint_model.discriminator, "module") else self.inpaint_model.discriminator
                        torch.save({
                            'discriminator': raw_model.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_dis.pth'))

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 1 and self.global_rank == 0:
                    self.save()
            if self.global_rank == 0:
                print("Epoch: %d, time for one epoch: %d seconds" % (epoch, time.time() - epoch_start))
                logs = [('Epoch', epoch), ('time', time.time() - epoch_start)]
                self.log(logs)
        print('\nEnd training....')

    def eval(self):
        if self.config.DDP:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE // self.config.world_size,  ## BS of each GPU
                                    num_workers=12)
        else:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE, num_workers=12)

        total = len(self.val_dataset)

        self.inpaint_model.eval()

        if self.config.No_Bar:
            pass
        else:
            progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        with torch.no_grad():
            for items in tqdm(val_loader):
                iteration += 1
                items['image'] = items['image'].to(self.device)
                items['mask'] = items['mask'].to(self.device)
                b, _, _, _ = items['image'].size()

                # inpaint model
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
                # save
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 2, 3, 1).int().cpu().numpy()
                for img_num in range(b):
                    cv2.imwrite(self.val_path + '/' + items['name'][img_num], outputs_merged[img_num, :, :, ::-1])

        our_metric = get_inpainting_metrics(self.val_path, self.config.GT_Val_FOLDER, None, fid_test=True)

        if self.global_rank == 0:
            print("iter: %d, PSNR: %f, SSIM: %f, FID: %f, LPIPS: %f" %
                  (self.inpaint_model.iteration, float(our_metric['psnr']), float(our_metric['ssim']),
                   float(our_metric['fid']), float(our_metric['lpips'])))
            logs = [('iter', self.inpaint_model.iteration), ('PSNR', float(our_metric['psnr'])),
                    ('SSIM', float(our_metric['ssim'])), ('FID', float(our_metric['fid'])), ('LPIPS', float(our_metric['lpips']))]
            self.log(logs)
        return float(our_metric['psnr']), float(our_metric['ssim']), float(our_metric['fid'])

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            items['image'] = items['image'].to(self.device)
            items['mask'] = items['mask'].to(self.device)

            # inpaint model
            iteration = self.inpaint_model.iteration
            inputs = (items['image'] * (1 - items['mask']))
            items = self.inpaint_model(items)
            outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        images = stitch_images(
            self.postprocess(items['image'].cpu()),
            self.postprocess(inputs.cpu()),
            self.postprocess(items['mask'].cpu()),
            self.postprocess(items['predicted_image'].cpu()),
            self.postprocess(outputs_merged.cpu()),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[0]) + '\t' + str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


class ZITS:
    def __init__(self, config, gpu, rank, test=False, single_img_test=False):
        self.config = config
        self.device = gpu
        self.global_rank = rank

        self.model_name = 'inpaint'

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = DefaultInpaintingTrainingModule(config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu)

        if config.min_sigma is None:
            min_sigma = 2.0
        else:
            min_sigma = config.min_sigma
        if config.max_sigma is None:
            max_sigma = 2.5
        else:
            max_sigma = config.max_sigma
        if config.round is None:
            round = 1
        else:
            round = config.round

        if not test:
            self.train_dataset = DynamicDataset(config.TRAIN_FLIST, mask_path=config.TRAIN_MASK_FLIST,
                                                batch_size=config.BATCH_SIZE // config.world_size,
                                                pos_num=config.rel_pos_num, augment=True, training=True,
                                                test_mask_path=None, train_line_path=config.train_line_path,
                                                add_pos=config.use_MPE, world_size=config.world_size,
                                                min_sigma=min_sigma, max_sigma=max_sigma, round=round)
            if config.DDP:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config.world_size,
                                                        rank=self.global_rank, shuffle=True)
            else:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)

            self.samples_path = os.path.join(config.PATH, 'samples')
            self.results_path = os.path.join(config.PATH, 'results')

            self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

            self.best = float("inf") if self.inpaint_model.best is None else self.inpaint_model.best

        if not single_img_test:
            self.val_dataset = DynamicDataset(config.VAL_FLIST, mask_path=None, pos_num=config.rel_pos_num,
                                              batch_size=config.BATCH_SIZE, augment=False, training=False,
                                              test_mask_path=config.TEST_MASK_FLIST,
                                              eval_line_path=config.eval_line_path,
                                              add_pos=config.use_MPE, input_size=config.INPUT_SIZE,
                                              min_sigma=min_sigma, max_sigma=max_sigma)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
            self.val_path = os.path.join(config.PATH, 'validation')
            create_dir(self.val_path)

    def save(self):
        if self.global_rank == 0:
            self.inpaint_model.save()

    def train(self):
        if self.config.DDP:
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE // self.config.world_size,
                                      num_workers=12, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE, num_workers=12,
                                      sampler=self.train_sampler)
        epoch = self.inpaint_model.iteration // len(train_loader)
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset) // self.config.world_size

        if total == 0 and self.global_rank == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while keep_training:

            epoch += 1
            if self.config.DDP or self.config.DP:
                self.train_sampler.set_epoch(epoch + 1)
            if self.config.fix_256 is None or self.config.fix_256 is False:
                self.train_dataset.reset_dataset(self.train_sampler)

            epoch_start = time.time()
            if self.global_rank == 0:
                print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'loss_scale',
                                                                 'g_lr', 'd_lr', 'str_lr', 'img_size'],
                              verbose=1 if self.global_rank == 0 else 0)

            for _, items in enumerate(train_loader):
                iteration = self.inpaint_model.iteration

                self.inpaint_model.train()
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                image_size = items['image'].shape[2]
                random_add_v = random.random() * 1.5 + 1.5
                random_mul_v = random.random() * 1.5 + 1.5  # [1.5~3]

                # random mix the edge and line
                if iteration > int(self.config.MIX_ITERS):
                    b, _, _, _ = items['edge'].shape
                    if int(self.config.MIX_ITERS) < iteration < int(self.config.Turning_Point):
                        pred_rate = (iteration - int(self.config.MIX_ITERS)) / \
                                    (int(self.config.Turning_Point) - int(self.config.MIX_ITERS))
                        b = np.clip(int(pred_rate * b), 2, b)
                    iteration_num_for_pred = int(random.random() * 5) + 1
                    edge_pred, line_pred = SampleEdgeLineLogits(self.inpaint_model.transformer,
                                                                context=[items['img_256'][:b, ...],
                                                                         items['edge_256'][:b, ...],
                                                                         items['line_256'][:b, ...]],
                                                                mask=items['mask_256'][:b, ...].clone(),
                                                                iterations=iteration_num_for_pred,
                                                                add_v=0.05, mul_v=4)
                    edge_pred = edge_pred.detach().to(torch.float32)
                    line_pred = line_pred.detach().to(torch.float32)
                    if self.config.fix_256 is None or self.config.fix_256 is False:
                        if image_size < 300 and random.random() < 0.5:
                            edge_pred = F.interpolate(edge_pred, size=(image_size, image_size), mode='nearest')
                            line_pred = F.interpolate(line_pred, size=(image_size, image_size), mode='nearest')
                        else:
                            edge_pred = self.inpaint_model.structure_upsample(edge_pred)[0]
                            edge_pred = torch.sigmoid((edge_pred + random_add_v) * random_mul_v)
                            edge_pred = F.interpolate(edge_pred, size=(image_size, image_size), mode='bilinear',
                                                      align_corners=False)
                            line_pred = self.inpaint_model.structure_upsample(line_pred)[0]
                            line_pred = torch.sigmoid((line_pred + random_add_v) * random_mul_v)
                            line_pred = F.interpolate(line_pred, size=(image_size, image_size), mode='bilinear',
                                                      align_corners=False)
                    items['edge'][:b, ...] = edge_pred.detach()
                    items['line'][:b, ...] = line_pred.detach()

                # train
                outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs = [("epoch", epoch), ("iter", iteration)] + \
                       [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                logs.append(("g_lr", self.inpaint_model.g_scheduler.get_lr()[0]))
                logs.append(("d_lr", self.inpaint_model.d_scheduler.get_lr()[0]))
                logs.append(("str_lr", self.inpaint_model.str_scheduler.get_lr()[0]))
                logs.append(("img_size", batch['size_ratio'][0].item() * 256))
                progbar.add(len(items['image']),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0 and self.global_rank == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration > 0 and iteration % self.config.SAMPLE_INTERVAL == 0 and self.global_rank == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration > 0 and iteration % self.config.EVAL_INTERVAL == 0 and self.global_rank == 0:
                    print('\nstart eval...\n')
                    print("Epoch: %d" % epoch)
                    psnr, ssim, fid = self.eval()
                    if self.best > fid:
                        self.best = fid
                        print("current best epoch is %d" % epoch)
                        print('\nsaving %s...\n' % self.inpaint_model.name)
                        raw_model = self.inpaint_model.generator.module if \
                            hasattr(self.inpaint_model.generator, "module") else self.inpaint_model.generator
                        raw_encoder = self.inpaint_model.str_encoder.module if \
                            hasattr(self.inpaint_model.str_encoder, "module") else self.inpaint_model.str_encoder
                        torch.save({
                            'iteration': self.inpaint_model.iteration,
                            'generator': raw_model.state_dict(),
                            'str_encoder': raw_encoder.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH,
                                        self.inpaint_model.name + '_best_gen_HR.pth'))
                        raw_model = self.inpaint_model.discriminator.module if \
                            hasattr(self.inpaint_model.discriminator, "module") else self.inpaint_model.discriminator
                        torch.save({
                            'discriminator': raw_model.state_dict()
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_dis_HR.pth'))

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration > 0 and iteration % self.config.SAVE_INTERVAL == 0 and self.global_rank == 0:
                    self.save()
            if self.global_rank == 0:
                print("Epoch: %d, time for one epoch: %d seconds" % (epoch, time.time() - epoch_start))
                logs = [('Epoch', epoch), ('time', time.time() - epoch_start)]
                self.log(logs)
        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                batch_size=self.config.BATCH_SIZE, num_workers=12)

        self.inpaint_model.eval()

        with torch.no_grad():
            for items in tqdm(val_loader):
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)
                b, _, _, _ = items['edge'].shape
                edge_pred, line_pred = SampleEdgeLineLogits(self.inpaint_model.transformer,
                                                            context=[items['img_256'][:b, ...],
                                                                     items['edge_256'][:b, ...],
                                                                     items['line_256'][:b, ...]],
                                                            mask=items['mask_256'][:b, ...].clone(),
                                                            iterations=5,
                                                            add_v=0.05, mul_v=4,
                                                            device=self.device)
                edge_pred, line_pred = edge_pred[:b, ...].detach().to(torch.float32), \
                                       line_pred[:b, ...].detach().to(torch.float32)
                if self.config.fix_256 is None or self.config.fix_256 is False:
                    edge_pred = self.inpaint_model.structure_upsample(edge_pred)[0]
                    edge_pred = torch.sigmoid((edge_pred + 2) * 2)
                    line_pred = self.inpaint_model.structure_upsample(line_pred)[0]
                    line_pred = torch.sigmoid((line_pred + 2) * 2)
                items['edge'][:b, ...] = edge_pred.detach()
                items['line'][:b, ...] = line_pred.detach()
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
                # save
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 2, 3, 1).int().cpu().numpy()
                for img_num in range(b):
                    cv2.imwrite(self.val_path + '/' + items['name'][img_num], outputs_merged[img_num, :, :, ::-1])

        our_metric = get_inpainting_metrics(self.val_path, self.config.GT_Val_FOLDER, None, fid_test=True)

        if self.global_rank == 0:
            print("iter: %d, PSNR: %f, SSIM: %f, FID: %f, LPIPS: %f" %
                  (self.inpaint_model.iteration, float(our_metric['psnr']), float(our_metric['ssim']),
                   float(our_metric['fid']), float(our_metric['lpips'])))
            logs = [('iter', self.inpaint_model.iteration), ('PSNR', float(our_metric['psnr'])),
                    ('SSIM', float(our_metric['ssim'])), ('FID', float(our_metric['fid'])),
                    ('LPIPS', float(our_metric['lpips']))]
            self.log(logs)
        return float(our_metric['psnr']), float(our_metric['ssim']), float(our_metric['fid'])

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            b, _, _, _ = items['edge'].shape
            edge_pred, line_pred = SampleEdgeLineLogits(self.inpaint_model.transformer,
                                                        context=[items['img_256'][:b, ...],
                                                                 items['edge_256'][:b, ...],
                                                                 items['line_256'][:b, ...]],
                                                        mask=items['mask_256'][:b, ...].clone(),
                                                        iterations=5,
                                                        add_v=0.05, mul_v=4,
                                                        device=self.device)
            edge_pred, line_pred = edge_pred[:b, ...].detach().to(torch.float32), \
                                   line_pred[:b, ...].detach().to(torch.float32)
            if self.config.fix_256 is None or self.config.fix_256 is False:
                edge_pred = self.inpaint_model.structure_upsample(edge_pred)[0]
                edge_pred = torch.sigmoid((edge_pred + 2) * 2)
                line_pred = self.inpaint_model.structure_upsample(line_pred)[0]
                line_pred = torch.sigmoid((line_pred + 2) * 2)
            items['edge'][:b, ...] = edge_pred.detach()
            items['line'][:b, ...] = line_pred.detach()
            # inpaint model
            iteration = self.inpaint_model.iteration
            inputs = (items['image'] * (1 - items['mask']))
            items = self.inpaint_model(items)
            outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        images = stitch_images(
            self.postprocess((items['image']).cpu()),
            self.postprocess((inputs).cpu()),
            self.postprocess(items['edge'].cpu()),
            self.postprocess(items['line'].cpu()),
            self.postprocess(items['mask'].cpu()),
            self.postprocess((items['predicted_image']).cpu()),
            self.postprocess((outputs_merged).cpu()),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(6) + ".jpg")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[0]) + '\t' + str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
