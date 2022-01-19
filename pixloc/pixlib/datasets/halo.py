from pathlib import Path

import cv2
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle
import json
from ouster import client
import rosbag
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError

from .base_dataset import BaseDataset
from ..geometry import Camera, Pose, OusterLidar
from ...settings import DATA_PATH
from pixloc.utils.quaternions import qvec2rotmat

logger = logging.getLogger(__name__)


class Halo(BaseDataset):

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Halo_Dataset(self.conf)


def drs_q_t_to_T(q, t):
    def list_rotate(l, n):
        return l[n:] + l[:n]

    q = list_rotate(list(q), 3)
    q = np.array(q)
    t = np.array(t)
    rot_mat = qvec2rotmat(q)
    return np.vstack((np.hstack((rot_mat, t[:, None])), np.array([0, 0, 0, 1])[None, :]))


class _Halo_Dataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        self.BAG_PATH = "/home/fu/Desktop/sample_ouster_frame_data/halo_data.bag"
        self.bridge = CvBridge()
        with rosbag.Bag(self.BAG_PATH) as bag:
            for topic, msg, t in bag.read_messages(topics=['data']):
                corresponding_tuple = msg
                break
        self.data = corresponding_tuple

        CONFIG_PATH = "/home/fu/catkin_ws/src/lidar_undistortion/lidar_undistortion/config/os0_128_gen2_fw_114b12_sn_992015000018.json"
        with open(CONFIG_PATH, "r") as f:
            metadata = client.SensorInfo(f.read())

        W = 500  # metadata.format.columns_per_frame
        H = metadata.format.pixels_per_column
        left_x = 350
        top_y = 0
        n_azimuth_beams = metadata.format.columns_per_frame
        n_altitude_beams = metadata.format.pixels_per_column
        first_altitude_angle = metadata.beam_altitude_angles[0] * np.pi / 180.0
        last_altitude_angle = metadata.beam_altitude_angles[-1] * np.pi / 180.0
        zero_pix_offset_azimuth = -10.97 * np.pi / 180.0
        lidar_origin_to_beam_origin_m = 0.02767
        R = [-1., 0., 0., 0., -1., 0., 0., 0., 1.]
        t = [0., 0., 0.03618]

        ouster_config = torch.tensor([W, H, left_x, top_y,
                             n_azimuth_beams, n_altitude_beams, first_altitude_angle, last_altitude_angle,
                             zero_pix_offset_azimuth, lidar_origin_to_beam_origin_m, *R, *t])
        self.ouster_sensor = OusterLidar(ouster_config.float())

        B_r_BL = [0.001, 0.000, 0.091]
        q_BL = [0.0, 0.0, 0.0, 1.0]

        T_base_lidar = drs_q_t_to_T(q_BL, B_r_BL)
        q_BC = [-0.499, 0.501, -0.499, 0.501]  # the base here is the bottom of NUC
        B_r_BC = [0.082, 0.053, 0.077]

        T_base_left_camera = drs_q_t_to_T(q_BC, B_r_BC)
        self.T_left_camera_lidar = np.linalg.inv(T_base_left_camera).dot(T_base_lidar)

        left_calib = [720, 540, 353.65, 353.02, 362.44, 288.49]
        self.left_camera = Camera(torch.tensor(left_calib, dtype=torch.float32))
        left_K = np.array([[353.65, 0, 362.44], [0, 353.02, 288.49], [0, 0, 1]])
        left_D = (-0.0391, -0.0084, 0.0069, -0.0022)
        self.left_camera_map1, self.left_camera_map2 = cv2.fisheye.initUndistortRectifyMap(left_K, left_D, np.eye(3),
                                                                                           left_K, left_calib[:2],
                                                                                           cv2.CV_16SC2)

    def __getitem__(self, idx):
        camera_image = self.bridge.imgmsg_to_cv2(self.data.camera_image, "mono8")
        camera_image = cv2.remap(camera_image, self.left_camera_map1, self.left_camera_map2,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        camera_image = camera_image.astype(np.float32)
        near_ir_image = self.bridge.imgmsg_to_cv2(self.data.lidar_nearir_image, "mono16").astype(np.float32)
        points = [point[:3] for point in point_cloud2.read_points(self.data.lidar_points, skip_nans=True)]

        datum = dict()
        datum['ref'] = dict()
        datum['query'] = dict()
        datum['ref']['image'] = torch.tensor(near_ir_image, dtype=torch.float32).permute(2, 0, 1)
        datum['ref']['camera'] = self.ouster_sensor
        datum['ref']['points3D'] = torch.tensor(points, dtype=torch.float32)
        k = 512
        perm = torch.randperm(datum['ref']['points3D'].shape[0])
        idx = perm[:k]
        datum['ref']['points3D'] = datum['ref']['points3D'][idx]

        datum['query']['image'] = torch.tensor(camera_image, dtype=torch.float32).permute(2, 0, 1)
        datum['query']['camera'] = self.left_camera  # TODO: Are we using the left camera? Also fill in the poses below.

        datum['T_r2q_init'] = Pose.from_4x4mat(torch.eye(4, dtype=torch.float32))
        datum['T_r2q_gt'] = Pose.from_4x4mat(torch.from_numpy(self.T_left_camera_lidar))
        datum['ref']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4, dtype=torch.float32))
        datum['query']['T_w2cam'] = datum['T_r2q_gt']
        datum['scene'] = torch.tensor([0])
        return datum

    def __len__(self):
        return 1
