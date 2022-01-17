"""
Convenience classes for an SE3 pose and a pinhole Camera with lens distortion.
Based on PyTorch tensors: differentiable, batched, with GPU support.
"""

import functools
import inspect
import math
from typing import Union, Tuple, List, Dict, NamedTuple
import torch
import numpy as np
import os
import cv2

from .optimization import skew_symmetric, so3exp_map
from .utils import undistort_points, J_undistort_points


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
       if they are numpy arrays. Use the device and dtype of the wrapper.
    """
    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device('cpu')
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        else:
            return NotImplemented


class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        '''Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        '''
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_aa(cls, aa: torch.Tensor, t: torch.Tensor):
        '''Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            aa: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        '''
        assert aa.shape[-1] == 3
        assert t.shape[-1] == 3
        assert aa.shape[:-1] == t.shape[:-1]
        return cls.from_Rt(so3exp_map(aa), t)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        '''Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        '''
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    def from_colmap(cls, image: NamedTuple):
        '''Pose from a COLMAP Image.'''
        return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        '''Underlying translation vector with shape (..., 3).'''
        return self._data[..., -3:]

    def inv(self) -> 'Pose':
        '''Invert an SE(3) pose.'''
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C.'''
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        '''
        assert p3d.shape[-1] == 3
        # assert p3d.shape[:-2] == self.shape  # allow broadcasting
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points: T_A2B * p3D_A -> p3D_B.'''
        return self.transform(p3D)

    def __matmul__(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C.'''
        return self.compose(other)

    @autocast
    def J_transform(self, p3d_out: torch.Tensor):
        # [[1,0,0,0,-pz,py],
        #  [0,1,0,pz,0,-px],
        #  [0,0,1,-py,px,0]]
        J_t = torch.diag_embed(torch.ones_like(p3d_out))
        J_rot = -skew_symmetric(p3d_out)
        J = torch.cat([J_t, J_rot], dim=-1)
        return J  # N x 3 x 6

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> Tuple[torch.Tensor]:
        '''Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation anngle in degrees.
            dt: translation distance in meters.
        '''
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.acos(cos).abs() / math.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def __repr__(self):
        return f'Pose: {self.shape} {self.dtype} {self.device}'


class Camera(TensorWrapper):
    eps = 1e-4

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6, 8, 10, 22}
        super().__init__(data)

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]):
        '''Camera from a COLMAP Camera tuple or dictionary.
        We assume that the origin (0, 0) is the center of the top-left pixel.
        This is different from COLMAP.
        '''
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera['model']
        params = camera['params']

        if model in ['OPENCV', 'PINHOLE']:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL']:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == 'SIMPLE_RADIAL':
                params = np.r_[params, 0.]
        else:
            raise NotImplementedError(model)

        data = np.r_[camera['width'], camera['height'],
                     fx, fy, cx - 0.5, cy - 0.5, params]
        return cls(data)

    @property
    def size(self) -> torch.Tensor:
        '''Size (width height) of the images, with shape (..., 2).'''
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        '''Focal lengths (fx, fy) with shape (..., 2).'''
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        '''Principal points (cx, cy) with shape (..., 2).'''
        return self._data[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        '''Distortion parameters, with shape (..., {0, 2, 4}).'''
        return self._data[..., 6:]

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        '''Update the camera parameters after resizing an image.'''
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        data = torch.cat([
            self.size*s,
            self.f*s,
            (self.c+0.5)*s-0.5,
            self.dist], -1)
        return self.__class__(data)

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        '''Update the camera parameters after cropping an image.'''
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([
            size,
            self.f,
            self.c - left_top,
            self.dist], -1)
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        '''Check if 2D points are within the image boundaries.'''
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Project 3D points into the camera plane and check for visibility.'''
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid

    def J_project(self, p3d: torch.Tensor):
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        J = torch.stack([
            1 / z, zero, -x / z ** 2,
            zero, 1 / z, -y / z ** 2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    @autocast
    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Undistort normalized 2D coordinates
           and check for validity of the distortion model.
        '''
        assert pts.shape[-1] == 2
        # assert pts.shape[:-2] == self.shape  # allow broadcasting
        return undistort_points(pts, self.dist)

    def J_undistort(self, pts: torch.Tensor):
        return J_undistort_points(pts, self.dist)  # N x 2 x 2

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        '''Convert normalized 2D coordinates into pixel coordinates.'''
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    def J_denormalize(self):
        return torch.diag_embed(self.f).unsqueeze(-3)  # 1 x 2 x 2

    @autocast
    def world2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Transform 3D points into 2D pixel coordinates.'''
        p2d, visible = self.project(p3d)
        p2d, mask = self.undistort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    def J_world2image(self, p3d: torch.Tensor):
        p2d_dist, valid = self.project(p3d)
        J = (self.J_denormalize()
             @ self.J_undistort(p2d_dist)
             @ self.J_project(p3d))
        return J, valid

    def __repr__(self):
        return f'Camera {self.shape} {self.dtype} {self.device}'


class OusterLidarToBeamTransformer(object):
    def __init__(self):
        self.R = torch.from_numpy(np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])).float()
        self.t = torch.from_numpy(np.array([0., 0., 0.03618])).float()
        self.zero_pix_offset_azimuth = -10.97 * np.pi / 180.0
        self.lidar_origin_to_beam_origin_m = 0.02767

    def calculate_encoder_angle(self, p3d_beam_frame):
        return torch.fmod(torch.atan2(p3d_beam_frame[..., 1], p3d_beam_frame[..., 0]) + 2.0 * torch.pi +
                          self.zero_pix_offset_azimuth, 2.0 * np.pi).unsqueeze(-1)

    def transform_to_emitter_frame(self, p3d: torch.Tensor):
        p3d_beam_frame = (p3d - self.t).matmul(self.R)
        encoder = self.calculate_encoder_angle(p3d_beam_frame)
        offset_correction = torch.concat(
            (np.cos(encoder), np.sin(encoder), torch.zeros_like(encoder)), dim=-1) * self.lidar_origin_to_beam_origin_m
        spherical_points_tensor = p3d_beam_frame - offset_correction
        return spherical_points_tensor, encoder


def transform_to_emitter_frame(p3d: torch.Tensor):
    return OusterLidarToBeamTransformer().transform_to_emitter_frame(p3d)


class OusterData(object):
    def __init__(self):
        OUSTER_FRAME_PATH = "/home/fu/Desktop/sample_ouster_frame_data"
        RANGE_IMAGE_PATH = os.path.join(OUSTER_FRAME_PATH, "latest_range.tif")
        X_IMAGE_PATH = os.path.join(OUSTER_FRAME_PATH, "latest_x.tif")
        Y_IMAGE_PATH = os.path.join(OUSTER_FRAME_PATH, "latest_y.tif")
        Z_IMAGE_PATH = os.path.join(OUSTER_FRAME_PATH, "latest_z.tif")
        self.depth_image = cv2.imread(RANGE_IMAGE_PATH, cv2.IMREAD_ANYDEPTH)
        self.x_image = cv2.imread(X_IMAGE_PATH, cv2.IMREAD_ANYDEPTH)
        self.y_image = cv2.imread(Y_IMAGE_PATH, cv2.IMREAD_ANYDEPTH)
        self.z_image = cv2.imread(Z_IMAGE_PATH, cv2.IMREAD_ANYDEPTH)


class OusterLidar(Camera):
    # eps = 1e-4

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 22
        super().__init__(data)

    @property
    def size(self) -> torch.Tensor:
        return self._data[..., :2]

    @property
    def top_left(self) -> torch.Tensor:
        return self._data[..., 2:4]

    @property
    def n_azimuth_beams(self):
        return self._data[..., 4:5]

    @property
    def n_altitude_beams(self):
        return self._data[..., 5:6]

    @property
    def first_altitude_angle(self):
        return self._data[..., 6:7]

    @property
    def last_altitude_angle(self):
        return self._data[..., 7:8]

    @property
    def zero_pix_offset_azimuth(self):
        return self._data[..., 8:9]

    @property
    def lidar_origin_to_beam_origin_m(self):
        return self._data[..., 9:10]

    @property
    def R_lidar_to_sensor(self) -> torch.Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., 10:19]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t_lidar_to_sensor(self) -> torch.Tensor:
        '''Underlying translation vector with shape (..., 3).'''
        return self._data[..., 19:]

    def is_in_image(self, p2d: torch.Tensor) -> torch.Tensor:
        '''Check if 2D points are within the image boundaries.'''
        assert p2d.shape[-1] == 2
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        '''Update the ouster parameters after resizing an image.'''
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        data = torch.cat([self.size*s, self.top_left*s, self.n_azimuth_beams*s[0], self.n_altitude_beams*s[1],
                          self.first_altitude_angle, self.last_altitude_angle,
                          self.zero_pix_offset_azimuth, self.lidar_origin_to_beam_origin_m,
                          self.R_lidar_to_sensor.flatten(), self.t_lidar_to_sensor])
        return self.__class__(data)

    @autocast
    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        '''Update the camera parameters after cropping an image.'''
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([size, left_top, self.n_azimuth_beams, self.n_altitude_beams,
                          self.first_altitude_angle, self.last_altitude_angle,
                          self.zero_pix_offset_azimuth, self.lidar_origin_to_beam_origin_m,
                          self.R_lidar_to_sensor.flatten(), self.t_lidar_to_sensor])

        return self.__class__(data)

    @autocast
    def to_sensor_frame(self, p3d: torch.Tensor) -> torch.Tensor:
        return (p3d - self.t_lidar_to_sensor).matmul(self.R_lidar_to_sensor)

    @autocast
    def calculate_azimuth_angle(self, p3d_sensor_frame: torch.Tensor) -> torch.Tensor:
        return torch.atan2(p3d_sensor_frame[..., 1], p3d_sensor_frame[..., 0]).unsqueeze(-1)

    @autocast
    def to_beam_frame(self, p3d_sensor_frame: torch.Tensor) -> torch.Tensor:
        xy_norm = torch.linalg.vector_norm(p3d_sensor_frame[..., :2], dim=-1)
        xy_norm = torch.clip(xy_norm, min=self.eps)
        azimuth_heading = p3d_sensor_frame[..., :2] / xy_norm.unsqueeze(-1)
        offset_correction = torch.concat(
            (azimuth_heading, torch.zeros_like(azimuth_heading[..., :1])), dim=-1) * self.lidar_origin_to_beam_origin_m
        p3d_emitter_frame = p3d_sensor_frame - offset_correction
        return p3d_emitter_frame

    @autocast
    def to_spherical_angles(self, p3d):
        emitter_ranges = torch.linalg.vector_norm(p3d, dim=-1)
        emitter_ranges = torch.clip(emitter_ranges, self.eps)
        ps2 = p3d / emitter_ranges.reshape(*p3d.shape[:-1], 1)
        theta_psi = torch.concat(
            (torch.atan2(ps2[..., 1], ps2[..., 0]).unsqueeze(-1), torch.asin(ps2[..., 2]).unsqueeze(-1)), dim=-1)
        return theta_psi

    @autocast
    def calculate_altitude(self, ps2) -> torch.Tensor:
        altitudes = torch.asin(ps2[..., 2]).unsqueeze(-1)
        return altitudes

    @autocast
    def calculate_vu(self, spherical_angles) -> torch.Tensor:
        azimuth = spherical_angles[..., :1]
        altitude = spherical_angles[..., 1:2]
        u = self.n_altitude_beams * (altitude - self.first_altitude_angle) / \
            (self.last_altitude_angle - self.first_altitude_angle)
        azimuth_zero_aligned = torch.fmod(azimuth + 2.0 * torch.pi + self.zero_pix_offset_azimuth, 2.0 * np.pi)
        v = (2.0 * torch.pi - azimuth_zero_aligned) / (2 * torch.pi / self.n_azimuth_beams)
        coords = torch.concat((v, u), dim=-1)
        coords -= self.top_left
        return coords

    def is_in_lidar_fov(self, spherical_angles):
        altitudes = spherical_angles[..., 1]
        return (max(self.first_altitude_angle,
                    self.last_altitude_angle) > altitudes) & (altitudes > min(self.first_altitude_angle,
                                                                              self.last_altitude_angle))

    @autocast
    def world2image(self, p3d: torch.Tensor):
        '''Transform 3D points into 2D pixel coordinates.'''
        ranges = torch.linalg.vector_norm(p3d, dim=-1)
        has_range = ranges > self.eps
        p3d_sensor_frame = self.to_sensor_frame(p3d)
        p3d_beam_frame = self.to_beam_frame(p3d_sensor_frame)
        spherical_angles = self.to_spherical_angles(p3d_beam_frame)
        vu = self.calculate_vu(spherical_angles)
        valid = has_range & self.is_in_lidar_fov(spherical_angles) & self.is_in_image(vu)
        return vu, valid

    @autocast
    def world2imagebk(self, p3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Transform 3D points into 2D pixel coordinates.'''
        ranges = torch.linalg.vector_norm(p3d, dim=-1)
        has_range = ranges > self.eps

        spherical_points_tensor, encoder = transform_to_emitter_frame(p3d)

        spherical_ranges = torch.linalg.vector_norm(spherical_points_tensor, dim=-1)
        unit_vectors = spherical_points_tensor / spherical_ranges.reshape(*spherical_points_tensor.shape[:-1], 1)
        altitudes = torch.asin(unit_vectors[..., 2]).unsqueeze(-1)
        is_in_lidar = (max(self.first_altitude_angle,
                           self.last_altitude_angle) > altitudes) & (altitudes > min(self.first_altitude_angle,
                                                                                     self.last_altitude_angle))
        # TODO: Where is the centre of the pixel? mid-pixel or top-left corner?
        u = self.n_altitude_beams * (altitudes - self.first_altitude_angle) / \
            (self.last_altitude_angle - self.first_altitude_angle)

        v = (2.0 * torch.pi - encoder) / (2 * torch.pi / self.n_azimuth_beams)
        coords = torch.concat((v, u), dim=-1)
        coords -= self.top_left
        valid = has_range & is_in_lidar.squeeze(-1) & self.is_in_image(coords)
        return coords, valid

    def J_world2image(self, p3d: torch.Tensor):
        p2d_dist, valid = self.project(p3d)
        J = (self.J_denormalize()
             @ self.J_undistort(p2d_dist)
             @ self.J_project(p3d))
        return J, valid