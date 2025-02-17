"""
    Generate a cfg object according to a cfg file and args, then spawn Trainer(rank, cfg).
"""

import argparse
import os

import yaml
import torch
import torch.multiprocessing as mp

import utils
import trainers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--load-root', default='data')
    parser.add_argument('--save-root', default='save_ucr')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--port-offset', '-p', type=int, default=0)
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    args = parser.parse_args()

    return args


def make_cfg(args, dataset=None, num=0):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                d[k] = v.replace('$load_root$', args.load_root)
    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag

    exp_name += '_'
    exp_name += str(num)
    env = dict()
    env['exp_name'] = exp_name
    env['save_dir'] = os.path.join(args.save_root, exp_name)
    env['tot_gpus'] = torch.cuda.device_count()
    env['cudnn'] = args.cudnn
    env['port'] = str(29600 + args.port_offset)
    env['wandb_upload'] = args.wandb_upload
    cfg['env'] = env

    cfg['ucr_train_dataset']['args']['args']['data'] = dataset
    cfg['ucr_test_dataset']['args']['args']['data'] = dataset

    return cfg


def main_worker(rank, cfg):
    trainer = trainers.trainers_dict[cfg['trainer']](rank, cfg)
    trainer.run()


def main(dataset=None, i=0):
    args = parse_args()

    cfg = make_cfg(args, dataset=dataset, num=i)
    utils.ensure_path(cfg['env']['save_dir'])

    if cfg['env']['tot_gpus'] > 1:
        mp.spawn(main_worker, args=(cfg,), nprocs=cfg['env']['tot_gpus'])
    else:
        main_worker(0, cfg)


if __name__ == '__main__':
    root_dir = './all_datasets/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere'
    total_dir = []
    for file in os.listdir(os.path.join(root_dir, 'AnomalyArchive')):
        tmp_dir = '_'.join(file.split('_')[:4])
        total_dir.append(tmp_dir)
    total_dir = sorted(total_dir)
    for i in range(250):
        print('dataset: ', total_dir[i])
        main(total_dir[i], i)
