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
from pixloc.visualization.viz_2d import (
    plot_images, plot_keypoints, plot_matches, cm_RdGn,
    features_to_RGB, add_text)

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
  def __init__(self):
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

  def process_inputs(self, image_0, image_1, lidar_points_in_lidar_frame, camera_0, camera_1, logger=None):
    data = dict()
    data['ref'] = dict()
    data['query'] = dict()
    data['ref']['image'] = (torch.from_numpy(image_0).permute(2, 0, 1) / 255.).unsqueeze(0).type(torch.float32)
    data['ref']['camera'] = camera_0
    data['ref']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))

    vertices_in_cam0 = self.T_right_camera_lidar.dot(np.vstack((lidar_points_in_lidar_frame,
                                                                        np.ones_like(lidar_points_in_lidar_frame[0, :]))))
    data['ref']['points3D'] = torch.tensor(vertices_in_cam0[:3, :].T, dtype=torch.float32)
    k = 512
    perm = torch.randperm(data['ref']['points3D'].shape[0])
    idx = perm[:k]
    data['ref']['points3D'] = data['ref']['points3D'][idx].unsqueeze(0)

    data['query']['image'] = (torch.from_numpy(image_1).permute(2, 0, 1) / 255.).unsqueeze(0).type(torch.float32)
    data['query']['camera'] = camera_1
    data['query']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))
    data['T_r2q_init'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))
    data['T_r2q_gt'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))
    data['scene'] = torch.tensor([0])
    pred = self.refiner(data)
    self.log_refinement(data, pred, self.refiner, logger)
    return pred

  def log_refinement(self, data, pred, refiner, logger):
    if logger is None:
      return
    cam_q = data['query']['camera']
    p3D_r = data['ref']['points3D']

    p2D_r, valid_r = data['ref']['camera'].world2image(p3D_r)
    p2D_q_gt, valid_q = cam_q.world2image(data['T_r2q_gt'] * p3D_r)
    p2D_q_init, _ = cam_q.world2image(data['T_r2q_init'] * p3D_r)
    p2D_q_opt, _ = cam_q.world2image(pred['T_r2q_opt'][-1] * p3D_r)
    valid = valid_q & valid_r

    # losses = refiner.loss(pred_, data_)
    # mets = refiner.metrics(pred_, data_)
    # errP = f"ΔP {losses['reprojection_error/init'].item():.2f} -> {losses['reprojection_error'].item():.3f} px; "
    # errR = f"ΔR {mets['R_error/init'].item():.2f} -> {mets['R_error'].item():.3f} deg; "
    # errt = f"Δt {mets['t_error/init'].item():.2f} -> {mets['t_error'].item():.3f} m"
    # print(errP, errR, errt)

    imr, imq = data['ref']['image'][0].permute(1, 2, 0), data['query']['image'][0].permute(1, 2, 0)
    n_points_plot = -1
    plot_images([imr, imq],
                dpi=100,  # set to 100-200 for higher res
                titles=[(data['scene'].item(), valid_r.sum().item(), valid_q.sum().item()), 0.0])
    plot_keypoints([p2D_r[valid_r][:n_points_plot], p2D_q_gt[valid][:n_points_plot]],
                   colors=[cm_RdGn(valid[valid_r][:n_points_plot]), 'lime'])
    plot_keypoints([np.empty((0, 2)), p2D_q_init[valid][:n_points_plot]], colors='red')
    plot_keypoints([np.empty((0, 2)), p2D_q_opt[valid][:n_points_plot]], colors='blue')
    add_text(0, 'reference')
    add_text(1, 'query')

    #     continue
    for i, (F0, F1) in enumerate(zip(pred['ref']['feature_maps'], pred['query']['feature_maps'])):
      C_r, C_q = pred['ref']['confidences'][i][0], pred['query']['confidences'][i][0]
      plot_images([C_r, C_q], cmaps=mpl.cm.Greys, dpi=100)
      add_text(0, f'Level {i}')

      axes = plt.gcf().axes
      axes[0].imshow(imr, alpha=0.2, extent=axes[0].images[0]._extent)
      axes[1].imshow(imq, alpha=0.2, extent=axes[1].images[0]._extent)
      plot_images(features_to_RGB(F0.numpy(), F1.numpy(), skip=1), dpi=100)
