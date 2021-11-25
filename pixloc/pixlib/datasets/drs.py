from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle
import json

from .base_dataset import BaseDataset
from ..geometry import Camera, Pose
from ...settings import DATA_PATH

logger = logging.getLogger(__name__)


class DRS(BaseDataset):

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _DRS_Dataset(self.conf)


class _DRS_Dataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        with open("/tmp/correspondences.json", "rb") as f:
            self.data = json.loads(f.read())

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self.data.__len__()
