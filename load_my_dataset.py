import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import pandas as pd
from glob import glob
from scipy.spatial.transform import Rotation


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_my_data(basedir, half_res=False, testskip=1):
    print("Load CameraInfo")
    file_camera_info = open(f"{basedir}/camera_info.txt")
    line = file_camera_info.readline()
    fx, cx, fy, cy = list(map(float, line.split(' ')))

    print(f"Load Pose")
    df_pose = pd.read_csv(f"{basedir}/pose.tsv", sep="\t")
    n = len(df_pose)
    pose_xyz = df_pose[['x', 'y', 'z']].values
    pose_quat = df_pose[['qx', 'qy', 'qz', 'qw']].values
    rotation_mat = Rotation.from_quat(pose_quat).as_matrix()
    pose_xyz -= pose_xyz[0]
    identity = np.eye(4, 4, dtype=np.float32)
    poses = np.stack([identity] * n)
    poses[:, 0:3, 0:3] = rotation_mat
    poses[:, 0:3, 3:4] = np.expand_dims(pose_xyz, -1)

    print("Load Images")
    imgs = []
    image_path_list = sorted(glob(f"{basedir}/images/*.png"))
    for image_path in image_path_list:
        imgs.append(imageio.imread(image_path))
    imgs = (np.array(imgs) / 255.).astype(np.float32)

    assert len(imgs) == len(df_pose)

    i_split = [np.arange(0, n),
               np.arange(0, n // 10),
               np.arange(0, n // 10)]

    H, W = imgs[0].shape[:2]
    focal = (fx + fy) / 2

    render_poses = poses[0:n // 10]
    if half_res:
        RESIZE_FACTOR = 3
        H = H // RESIZE_FACTOR
        W = W // RESIZE_FACTOR
        focal = focal / RESIZE_FACTOR

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split
