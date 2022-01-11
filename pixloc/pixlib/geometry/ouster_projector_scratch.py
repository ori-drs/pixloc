import torch_interpolations
import torch

from scipy.interpolate import RBFInterpolator
import numpy as np
import json
import matplotlib.pyplot as plt

rng = np.random.default_rng()
OUSTER_CONFIG_PATH = "/home/fu/catkin_ws/src/lidar_undistortion/lidar_undistortion/config/os0_128_gen2_fw_114b12_sn_992015000018.json"

with open(OUSTER_CONFIG_PATH, "r") as f:
    ouster_config = json.loads(f.read())

azimuth_angles = ouster_config["beam_azimuth_angles"]
# plt.plot(azimuth_angles)

points = [torch.arange(0, azimuth_angles.__len__(), 1) * 1.]
values = torch.tensor(azimuth_angles)
gi = torch_interpolations.RegularGridInterpolator(points, values)
points_to_interp = (torch.arange(-0, 128, 1) * 1.).reshape(1, -1)
points_to_interp.requires_grad = True
fx = gi(points_to_interp)
plt.figure(figsize=(20, 20))
plt.plot(points_to_interp[0].detach().numpy(), fx.detach().numpy(), "r+")
plt.scatter(points[0].numpy(), values.numpy())

l = fx.sum()
l.backward()
print("torch: ", points_to_interp.grad)
print("Forward euler: ", np.array(azimuth_angles[1:]) - np.array(azimuth_angles[:-1]))