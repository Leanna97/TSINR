"""
    A basic trainer.

    The general procedure in run() is:
        make_datasets()
            create . train_loader, test_loader, dist_samplers
        make_model()
            create . model_ddp, model
        train()
            create . optimizer, epoch, log_buffer
            for epoch = 1 ... max_epoch:
                adjust_learning_rate()
                train_epoch()
                    train_step()
                evaluate_epoch()
                    evaluate_step()
                visualize_epoch()
                save_checkpoint()
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import os.path as osp
import time

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt

import datasets
import models
import utils
from trainers import register
from utils.tools import adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

@register('base_trainer')
class BaseTrainer():

    def __init__(self, rank, cfg):
        self.rank = rank
        self.cfg = cfg
        self.is_master = (rank == 0)

        env = cfg['env']
        self.tot_gpus = env['tot_gpus']
        self.distributed = (env['tot_gpus'] > 1)

        # Setup log, tensorboard, wandb
        if self.is_master:
            logger, writer = utils.set_save_dir(env['save_dir'], replace=False)
            with open(osp.join(env['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)

            self.log = logger.info

            self.enable_tb = True
            self.writer = writer

            if env['wandb_upload']:
                self.enable_wandb = True
                with open('wandb.yaml', 'r') as f:
                    wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                os.environ['WANDB_DIR'] = env['save_dir']
                os.environ['WANDB_NAME'] = env['exp_name']
                os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
                wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfg)
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        # Setup distributed devices
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        if self.distributed:
            dist_url = f"tcp://localhost:{env['port']}"
            dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=self.tot_gpus, rank=rank)
            self.log(f'Distributed training enabled.')

        cudnn.benchmark = env['cudnn']

        self.log(f'Environment setup done.')

    def run(self):
        self.anomaly_ratio = self.cfg['anomaly_ratio']

        self.make_datasets()
        self.make_model()
        self.train()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

            Cfg example:

            train/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        cfg = self.cfg
        self.dist_samplers = []

        def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
            sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset,
                batch_size // self.tot_gpus,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers // self.tot_gpus,
                pin_memory=True)
            return loader, sampler

        if cfg.get('all_train_dataset') is not None:
            tmp = datasets.make(cfg['all_train_dataset'])
            train_dataset = tmp.data_set
            self.train_loader = tmp.data_loader
            self.log(f'Train dataset: len={len(train_dataset)}')

        if cfg.get('all_test_dataset') is not None:
            tmp = datasets.make(cfg['all_test_dataset'])
            test_dataset = tmp.data_set
            self.test_loader = tmp.data_loader
            self.log(f'Test dataset: len={len(test_dataset)}')

        if cfg.get('ucr_train_dataset') is not None:
            train_dataset = datasets.make(cfg['ucr_train_dataset'])
            self.log(f'Train dataset: len={len(train_dataset)}')
            l = cfg['ucr_train_dataset']['loader']
            self.train_loader, train_sampler = make_distributed_loader(
                train_dataset, l['batch_size'], l['num_workers'], shuffle=True, drop_last=True)
            self.dist_samplers.append(train_sampler)

        if cfg.get('ucr_test_dataset') is not None:
            test_dataset = datasets.make(cfg['ucr_test_dataset'])
            self.log(f'Test dataset: len={len(test_dataset)}')
            l = cfg['ucr_test_dataset']['loader']
            self.test_loader, test_sampler = make_distributed_loader(
                test_dataset, l['batch_size'], l['num_workers'], shuffle=False, drop_last=False)
            self.dist_samplers.append(test_sampler)


    def make_model(self, model_spec=None, load_sd=False):
        if model_spec is None:
            model_spec = self.cfg['model']
        model = models.make(model_spec, load_sd=load_sd)
        self.log(f'Model: #params={utils.compute_num_params(model)}')
        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank])
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), cfg['optimizer'])

        max_epoch = cfg['max_epoch']
        eval_epoch = cfg.get('eval_epoch', max_epoch + 1)
        save_epoch = cfg.get('save_epoch', max_epoch + 1)
        epoch_timer = utils.EpochTimer(max_epoch)

        tmp = 0
        for epoch in range(1, max_epoch + 1):
            self.epoch = epoch
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)

            self.adjust_learning_rate()

            self.t_data, self.t_model = 0, 0
            self.train_epoch()

            if epoch % eval_epoch == 0:
                ret = self.anomaly_evaluate_epoch()
                tmp = self.evaluate_epoch()
                result = {}
                for k, v in tmp.items():
                    result[k] = v.item()

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')
            self.save_checkpoint('epoch-last.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model)
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

        result = {}
        for k, v in tmp.items():
            result[k] = v.item()


    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        self.log_temp_scalar('lr', self.optimizer.param_groups[0]['lr'])

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({k: v}, step=t)

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)

    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor(ave_scalars[k].item(), dtype=torch.float32, device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.item()
            ave_scalars[k].n *= self.tot_gpus

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = dict()

        pbar = self.train_loader
        t1 = time.time()

        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.train_step(data)
            self.t_model += time.time() - t0

            B = data[0].shape[0]
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'train:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('train/' + k, v.item())
        self.log_buffer.append(logtext)


    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.test_loader

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.evaluate_step(data)
            self.t_model += time.time() - t0

            B = data[0].shape[0]
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        tmp_each_epoch = ave_scalars

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('test/' + k, v.item())
        self.log_buffer.append(logtext)
        return tmp_each_epoch


    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_spec = self.cfg['model']
        model_spec['sd'] = self.model.state_dict()
        optimizer_spec = self.cfg['optimizer']
        optimizer_spec['sd'] = self.optimizer.state_dict()
        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': self.epoch,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, osp.join(self.cfg['env']['save_dir'], filename))

    def anomaly_evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.train_loader

        t1 = time.time()
        attens_energy = []
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for data in pbar:
                t0 = time.time()
                self.t_data += t0 - t1
                ret, pred, gt = self.evaluate_step(data, test=True)
                self.t_model += time.time() - t0

                # criterion
                score = torch.mean(self.anomaly_criterion(gt, pred), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        pbar = self.test_loader
        attens_energy = []
        test_labels = []
        for data in pbar:
            ret, pred, gt = self.evaluate_step(data, test=True)
            B = data[0].shape[0]
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            # criterion
            score = torch.mean(self.anomaly_criterion(gt, pred), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(data[1])

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        x = [i for i in range(combined_energy.shape[0])]

        threshold = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))


        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        tmp_each_epoch = ave_scalars

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'

        logtext += '\n'
        logtext = logtext + 'Threshold: ' + str(threshold) + '\n'
        logtext = logtext + 'Accuracy: ' + str(accuracy) + '\n'
        logtext = logtext + 'Precision: ' + str(precision) + '\n'
        logtext = logtext + 'Recall: ' + str(recall) + '\n'
        logtext = logtext + 'F-score: ' + str(f_score) + '\n'

        self.log_buffer.append(logtext)
        return tmp_each_epoch

