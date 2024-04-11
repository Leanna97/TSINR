import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

from datasets import register
from ucr_anomaly_data_provider.data_loader import load_data
from ucr_anomaly_data_provider.loader import Loader


@register('ucr_timeseries')
class UCRTimeseries(Dataset):

    def __init__(self, split, args):
        DATASET = 'anomaly_archive'  # 'anomaly_archive' OR 'smd'
        ENTITY = args['data']  # Name of timeseries in UCR or machine in SMD
        data_set = load_data(dataset=DATASET, group=split, entities=[ENTITY], downsampling=None, root_dir=args['root_path'], normalize=True, verbose=True)

        self.X = data_set.entities[0].Y.transpose((1,0))
        self.test_labels = data_set.entities[0].labels.squeeze(0)

        if split == 'train':
            self.train = self.X
            self.val = self.test = None
        else:
            self.test = self.X
            self.train = self.val = None

        self.win_size = args['win_size']
        self.step = args['step']
        self.flag = split

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



