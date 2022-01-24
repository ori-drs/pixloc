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

    default_conf = {
        'dataset_dir': 'halo/',
        'info_dir': 'halo_trainin/',

        'train_slices': [None],
        'val_slices': [None],
        'train_num_per_slice': 1000,
        'val_num_per_slice': 80,

        'two_view': True,
        'min_overlap': 0.3,
        'max_overlap': 1.,
        'min_baseline': None,
        'max_baseline': None,
        'sort_by_overlap': False,

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': True,
        'seed': 0,

        'max_num_points3D': 512,
        'force_num_points3D': False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Halo_Dataset(self.conf, split)


def drs_q_t_to_T(q, t):
    def list_rotate(l, n):
        return l[n:] + l[:n]

    q = list_rotate(list(q), 3)
    q = np.array(q)
    t = np.array(t)
    rot_mat = qvec2rotmat(q)
    return np.vstack((np.hstack((rot_mat, t[:, None])), np.array([0, 0, 0, 1])[None, :]))


class _Halo_Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.seed = 0
        self.conf = conf
        self.split = split
        self.BAG_PATH = "/home/fu/Desktop/sample_ouster_frame_data/halo_data.bag"
        self.bridge = CvBridge()
        self.data = list()
        with rosbag.Bag(self.BAG_PATH) as bag:
            i = 0
            for topic, msg, t in bag.read_messages(topics=['data']):
                self.data.append(msg)
                i += 1
                if self.split != "train":
                    break
                elif i >= 100:
                    break

        CONFIG_PATH = "/home/fu/catkin_ws/src/lidar_undistortion/lidar_undistortion/config/os0_128_gen2_fw_114b12_sn_992015000018.json"
        with open(CONFIG_PATH, "r") as f:
            metadata = client.SensorInfo(f.read())

        W = 250  # metadata.format.columns_per_frame
        H = metadata.format.pixels_per_column
        left_x = 450
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
        self.lidar_scale = 4.0
        self.ouster_sensor = OusterLidar(ouster_config.float()).scale(self.lidar_scale)

        B_r_BL = [0.001, 0.000, 0.091]
        q_BL = [0.0, 0.0, 0.0, 1.0]

        T_base_lidar = drs_q_t_to_T(q_BL, B_r_BL)
        q_BC = [-0.499, 0.501, -0.499, 0.501]  # the base here is the bottom of NUC
        B_r_BC = [0.082, 0.053, 0.077]

        T_base_left_camera = drs_q_t_to_T(q_BC, B_r_BC)
        self.T_left_camera_lidar = np.linalg.inv(T_base_left_camera).dot(T_base_lidar).astype(np.float32)

        left_calib = [720, 540, 353.65, 353.02, 362.44, 288.49]
        self.left_camera = Camera(torch.tensor(left_calib, dtype=torch.float32))
        left_K = np.array([[353.65, 0, 362.44], [0, 353.02, 288.49], [0, 0, 1]])
        left_D = (-0.0391, -0.0084, 0.0069, -0.0022)
        self.left_camera_map1, self.left_camera_map2 = cv2.fisheye.initUndistortRectifyMap(left_K, left_D, np.eye(3),
                                                                                           left_K, left_calib[:2],
                                                                                           cv2.CV_16SC2)

    def __getitem__(self, idx):
        camera_image = self.bridge.imgmsg_to_cv2(self.data[idx].camera_image, "mono8")
        camera_image = cv2.remap(camera_image, self.left_camera_map1, self.left_camera_map2,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        camera_image = (camera_image.astype(np.float32) / 255. - 0.5)[:, :, None]
        zeros = np.zeros_like(camera_image)
        camera_image = np.dstack((camera_image, zeros, zeros))
        near_ir_image = self.bridge.imgmsg_to_cv2(self.data[idx].lidar_nearir_image, "mono16").astype(np.float32)
        near_ir_image = cv2.resize(near_ir_image, None, None, self.lidar_scale, self.lidar_scale)
        near_ir_image = near_ir_image[:, int(self.ouster_sensor.top_left[0].item()):
                                         int(self.ouster_sensor.top_left[0] + self.ouster_sensor.size[0])]
        near_ir_image -= np.mean(near_ir_image)
        ir_span = max(near_ir_image.max(), -near_ir_image.min())
        near_ir_image /= ir_span
        near_ir_image = near_ir_image[:, :, None]
        zeros = np.zeros_like(near_ir_image)
        near_ir_image = np.dstack((zeros, near_ir_image, zeros))
        points = [point[:3] for point in point_cloud2.read_points(self.data[idx].lidar_points, skip_nans=True)]
        points = np.array(points).reshape(-1, 3)

        datum = dict()
        datum['ref'] = dict()
        datum['query'] = dict()
        datum['ref']['image'] = torch.tensor(near_ir_image, dtype=torch.float32).permute(2, 0, 1)
        datum['ref']['camera'] = self.ouster_sensor
        datum['ref']['points3D'] = torch.tensor(points, dtype=torch.float32)

        datum['query']['image'] = torch.tensor(camera_image, dtype=torch.float32).permute(2, 0, 1)
        datum['query']['camera'] = self.left_camera  # TODO: Are we using the left camera? Also fill in the poses below.

        datum['T_r2q_gt'] = Pose.from_4x4mat(torch.from_numpy(self.T_left_camera_lidar))

        p3d_q = datum['T_r2q_gt'].transform(datum['ref']['points3D'])
        _, valid_in_camera = self.left_camera.world2image(p3d_q)
        p3d_q = p3d_q[valid_in_camera]
        valid_in_gradients = self.left_camera.has_gradient_information(p3d_q,
                                                                       camera_image[:, :, 0].astype(np.float64))
        p3d_q = p3d_q[valid_in_gradients]
        datum['ref']['points3D'] = datum['T_r2q_gt'].inv().transform(p3d_q)
        torch.manual_seed(self.seed)
        perm = torch.randperm(datum['ref']['points3D'].shape[0])
        k = min(50, len(perm))
        idx = perm[:k]
        logger.info('Indices used: ' + idx.numpy().__str__())
        datum['ref']['points3D'] = datum['ref']['points3D'][idx]

        pose_error = np.array([[1,0,0,0],
                               [0,1,0,1.0],  #y-dimension error in lidar frame is easiest
                               [0,0,1,0],
                               [0,0,0,1]]).astype(np.float32)

        datum['T_r2q_init'] = Pose.from_4x4mat(torch.from_numpy(self.T_left_camera_lidar.dot(pose_error)))

        datum['ref']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4, dtype=torch.float32))
        datum['query']['T_w2cam'] = datum['T_r2q_gt']
        #TODO: Implement symmetric loss so that the lidar image is seen by the network optimiser.
        if True:  # Switch query vs. reference
            datum["ref"], datum["query"] = datum["query"], datum["ref"]
            del datum["query"]["points3D"]
            datum["ref"]["points3D"] = p3d_q
            datum['T_r2q_gt'] = datum['T_r2q_gt'].inv()
            datum['T_r2q_init'] = datum['T_r2q_init'].inv()

        datum['scene'] = torch.tensor([0])
        datum['query']["index"] = torch.tensor([0])
        datum['ref']["index"] = torch.tensor([0])
        return datum

    def __len__(self):
        return len(self.data)

    def no_op(self, seed):
        self.seed = seed