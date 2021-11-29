from pathlib import Path

import cv2
from plyfile import PlyData, PlyElement
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


class DRS(BaseDataset):

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _DRS_Dataset(self.conf)


def list_rotate(l, n):
    return l[n:] + l[:n]

def drs_q_t_to_T(q, t):
    q = list_rotate(list(q), 3)
    q = np.array(q)
    t = np.array(t)
    rot_mat = qvec2rotmat(q)
    return np.vstack((np.hstack((rot_mat, t[:,None])), np.array([0,0,0,1])[None,:]))


class _DRS_Dataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        with open("/tmp/correspondences.json", "rb") as f:
            self.data = json.loads(f.read())

        B_r_BL = [0.001, 0.000, 0.091]
        q_BL = [0.0, 0.0, 0.0, 1.0]

        T_base_lidar = drs_q_t_to_T(q_BL, B_r_BL)
        q_BC = [-0.499, 0.501, -0.499, 0.501]  # the base here is the bottom of NUC
        B_r_BC = [0.082, 0.053, 0.077]

        T_base_right_camera = drs_q_t_to_T(q_BC, B_r_BC)
        self.T_right_camera_lidar = np.linalg.inv(T_base_right_camera).dot(T_base_lidar)
        self.T_right_camera_left_camera = np.array(
            [[0.9999004450657571, 0.008840078761063054, 0.010997861828491322, 0.11117119807885806],
             [-0.008915175838954632, 0.9999371504175603, 0.006798150819857261, 0.0004908485221584779],
             [-0.01093707442879055, -0.006895521902452946, 0.9999164125968888, -0.0005279697447933163],
             [0.0, 0.0, 0.0, 1.0]]).astype(dtype=np.single)

        right_calib = [720, 540, 353.84, 353.08, 354.96, 261.97]
        left_calib = [720, 540, 353.65, 353.02, 362.44, 288.49]
        self.left_camera = Camera(torch.tensor(left_calib, dtype=torch.float32))
        self.right_camera = Camera(torch.tensor(right_calib, dtype=torch.float32))

    def __getitem__(self, idx):
        item = self.data[list(self.data.keys())[idx]]
        left_image = cv2.imread(item["left_image"])
        right_image = cv2.imread(item["right_image"])
        with open(item["lidar_points"], "rb") as f:
            lidar_ply = PlyData.read(f)
        vertices = np.stack((lidar_ply["vertex"]["x"],
                             lidar_ply["vertex"]["y"],
                             lidar_ply["vertex"]["z"]))
        vertices_in_right_camera = self.T_right_camera_lidar.dot(np.vstack((vertices,
                                                                      np.ones_like(vertices[0, :]))))

        datum = dict()
        datum['ref'] = dict()
        datum['query'] = dict()
        datum['ref']['image'] = torch.tensor(right_image,dtype=torch.float32).permute(2,0,1) / 255.
        datum['ref']['camera'] = self.right_camera
        datum['ref']['points3D'] = torch.tensor(vertices_in_right_camera[:3,:].T, dtype=torch.float32)
        k = 512
        perm = torch.randperm(datum['ref']['points3D'].shape[0])
        idx = perm[:k]
        datum['ref']['points3D'] = datum['ref']['points3D'][idx]

        datum['query']['image'] = torch.tensor(left_image, dtype=torch.float32).permute(2,0,1) / 255.
        datum['query']['camera'] = self.left_camera

        datum['T_r2q_init'] = Pose.from_4x4mat(torch.eye(4,dtype=torch.float32))
        datum['T_r2q_gt'] = Pose.from_4x4mat(torch.from_numpy(self.T_right_camera_left_camera))
        datum['ref']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4,dtype=torch.float32))
        datum['query']['T_w2cam'] = datum['T_r2q_gt']
        datum['scene'] = torch.tensor([0])
        return datum

    def __len__(self):
        return self.data.__len__()
