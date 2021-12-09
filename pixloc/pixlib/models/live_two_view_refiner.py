import cv2
import numpy as np
import logging
import torch

from ..geometry import Camera, Pose
from pixloc.pixlib.utils.experiments import load_experiment
from pixloc.visualization.viz_2d import features_to_RGB

logger = logging.getLogger(__name__)


def normalise_to_uint8(I):
  return cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def plot_circles(img, centres, color):
  for centre in centres:
    cv2.circle(img, centre, 3, color, thickness=cv2.FILLED)
  return img

class LiveTwoViewRefiner(object):
  def __init__(self):
    exp = "pixloc_author_reference"
    conf = {
      'normalize_dt': False,
      'optimizer': {'num_iters': 20, },
    }
    self.refiner = load_experiment(exp, conf)
    self.refiner.eval()

  def process_inputs(self, image_0, image_1, lidar_points_in_camera_0, camera_0, camera_1, logger=None):
    with torch.no_grad():
      data = dict()
      data['ref'] = dict()
      data['query'] = dict()
      data['ref']['image'] = (torch.from_numpy(image_0).permute(2, 0, 1) / 255.).unsqueeze(0).type(torch.float32)
      data['ref']['camera'] = camera_0
      data['ref']['T_w2cam'] = Pose.from_4x4mat(torch.eye(4,4,dtype=torch.float32))
      data['ref']['points3D'] = torch.tensor(lidar_points_in_camera_0, dtype=torch.float32)
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
      self.log_refinement(data, pred, logger)
    return pred

  def log_refinement(self, data, pred, logger):
    if logger is None:
      return
    cam_q = data['query']['camera']
    p3D_r = data['ref']['points3D']

    p2D_r, valid_r = data['ref']['camera'].world2image(p3D_r)
    p2D_q_gt, valid_q = cam_q.world2image(data['T_r2q_gt'] * p3D_r)
    p2D_q_init, _ = cam_q.world2image(data['T_r2q_init'] * p3D_r)
    p2D_q_opt, _ = cam_q.world2image(pred['T_r2q_opt'][-1] * p3D_r)

    valid = valid_q & valid_r

    imr, imq = data['ref']['image'][0].permute(1, 2, 0), data['query']['image'][0].permute(1, 2, 0)
    imr, imq = normalise_to_uint8(imr.numpy()), normalise_to_uint8(imq.numpy())
    imr = plot_circles(imr, p2D_r[valid].numpy().astype(np.int), (255,0,0))
    imq = plot_circles(imq, p2D_q_opt[valid].numpy().astype(np.int), (255,0,0))
    imq = plot_circles(imq, p2D_q_init[valid].numpy().astype(np.int), (0,0,255))
    logger["inputs"] = np.hstack((imr, imq))
    for i, (F0, F1) in enumerate(zip(pred['ref']['feature_maps'], pred['query']['feature_maps'])):
      logger[i] = dict()
      logger[i]["F0"] = features_to_RGB(F0.numpy()[0])[0]
      logger[i]["F1"] = features_to_RGB(F1.numpy()[0])[0]
      logger[i]["features"] = np.hstack((logger[i]["F0"], logger[i]["F1"]))
