from pathlib import Path

import cv2
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle
import json
from ..geometry import Camera, Pose


from .base_dataset import BaseDataset
from ..geometry import Camera, Pose
from ...settings import DATA_PATH
from pixloc.utils.quaternions import qvec2rotmat

logger = logging.getLogger(__name__)


class LiveSensors(BaseDataset):

  def _init(self, conf):
    print("LiveSensors Setup")
    pass

  def get_dataset(self, split):
    return _LiveSensors_Dataset()


class _LiveSensors_Dataset(torch.utils.data.Dataset):
  def __init__(self):
    pass

  def setup(self, cam0_intrinsics, cam1_intrinsics):
    pass

  def set_inputs(self, cam0, cam1, lidar_points_in_cam0_frame):
    print(lidar_points_in_cam0_frame.shape)
    data = dict()
    data['ref'] = dict()
    data['query'] = dict()
    data['ref']['image'] = None
    data['ref']['camera'] = None
    data['ref']['T_w2cam'] = None
    data['ref']['points3D'] = None
    data['query']['image'] = None
    data['query']['camera'] = None
    data['query']['T_w2cam'] = None
    data['T_r2q_init'] = None

  def __getitem__(self, idx):

    return None

  def __len__(self):
    return 1
