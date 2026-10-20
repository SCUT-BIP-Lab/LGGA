"""Shared configuration and sampling contract for DHGA dataset protocols."""

import torch
from torch.utils.data import Dataset

from src.dataset.dataset_utils import get_slice_list


class DHGABaseDataset(Dataset):
    def __init__(self, conf, phase):
        self.conf = conf
        self.phase = phase
        self.get_slice_list = get_slice_list
        self.gpu_video_transform = (
            self.as_bool(conf.get('dataset', {}).get('gpu_video_transform', True))
            and torch.device(conf.get('device', 'cpu')).type == 'cuda'
        )

    @staticmethod
    def as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    @staticmethod
    def normalize_subsets(subsets):
        if isinstance(subsets, str):
            return [item.strip() for item in subsets.split(',') if item.strip()]
        if isinstance(subsets, (list, tuple)):
            return [str(item).strip() for item in subsets if str(item).strip()]
        return []

    def __len__(self):
        return self.total_num
