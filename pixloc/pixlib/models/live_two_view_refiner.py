from pathlib import Path

import cv2
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle
import json

from ..geometry import Camera, Pose
from ...settings import DATA_PATH
from pixloc.utils.quaternions import qvec2rotmat
from pixloc.pixlib.utils.experiments import load_experiment

logger = logging.getLogger(__name__)


def list_rotate(l, n):
  return l[n:] + l[:n]

def drs_q_t_to_T(q, t):
  q = list_rotate(list(q), 3)
  q = np.array(q)
  t = np.array(t)
  rot_mat = qvec2rotmat(q)
  return np.vstack((np.hstack((rot_mat, t[:, None])), np.array([0, 0, 0, 1])[None, :]))

class LiveTwoViewRefiner(object):
  def __init__(self, cam0_intrinsics, cam1_intrinsics):
    exp = "pixloc_author_reference"
    conf = {
      'normalize_dt': False,
      'optimizer': {'num_iters': 20, },
    }
    self.refiner = load_experiment(exp, conf)

    B_r_BL = [0.001, 0.000, 0.091]
    q_BL = [0.0, 0.0, 0.0, 1.0]

    T_base_lidar = drs_q_t_to_T(q_BL, B_r_BL)
    q_BC = [-0.499, 0.501, -0.499, 0.501]  # the base here is the bottom of NUC
    B_r_BC = [0.082, 0.053, 0.077]

    T_base_right_camera = drs_q_t_to_T(q_BC, B_r_BC)
    self.T_right_camera_lidar = np.linalg.inv(T_base_right_camera).dot(T_base_lidar)

    calib0 = [720, 540, 353.84, 353.08, 354.96, 261.97]
    calib1 = [720, 540, 353.65, 353.02, 362.44, 288.49]
    self.calib1 = Camera(torch.tensor(calib1, dtype=torch.float32))
    self.calib0 = Camera(torch.tensor(calib0, dtype=torch.float32))
    pass

  def process_inputs(self, cam0, cam1, lidar_points_in_lidar_frame):
    # print(lidar_points_in_lidar_frame.shape)

    data = dict()
    data['ref'] = dict()
    data['query'] = dict()
    data['ref']['image'] = (torch.from_numpy(cam0).permute(2,0,1) / 255.).unsqueeze(0).type(torch.float32)
    data['ref']['camera'] = self.calib0
    data['ref']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))

    vertices_in_cam0 = self.T_right_camera_lidar.dot(np.vstack((lidar_points_in_lidar_frame,
                                                                        np.ones_like(lidar_points_in_lidar_frame[0, :]))))
    data['ref']['points3D'] = torch.tensor(vertices_in_cam0[:3, :].T, dtype=torch.float32)
    k = 512
    perm = torch.randperm(data['ref']['points3D'].shape[0])
    idx = perm[:k]
    data['ref']['points3D'] = data['ref']['points3D'][idx].unsqueeze(0)

    data['query']['image'] = (torch.from_numpy(cam1).permute(2,0,1) / 255.).unsqueeze(0).type(torch.float32)
    data['query']['camera'] = self.calib1
    data['query']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))
    data['T_r2q_init'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))
    pred = self.refiner(data)
    return pred
