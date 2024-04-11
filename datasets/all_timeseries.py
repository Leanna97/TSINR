import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

from datasets import register
from data_provider.data_factory import data_provider


@register('all_timeseries')
class AllTimeseries(Dataset):

    def __init__(self, split, args):
        data_set, data_loader = data_provider(args, split)
        self.data_set = data_set
        self.data_loader = data_loader

