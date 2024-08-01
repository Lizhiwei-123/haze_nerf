# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF++ (https://github.com/Kai-46/nerfplusplus)
# Copyright (c) 2020 the NeRF++ authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from subprocess import check_output
from typing import *

import imageio
import numpy as np


def _minify(basedir, factors=[], resolutions=[]): #缩小数据集尺寸
    subdirs = ["images", "enhancewater"]  # 子目录列表

    for subdir in subdirs:
        needtoload = False  # 是否需要处理图像
        subdir_path = os.path.join(basedir, subdir)

        # 检查因子缩小
        for r in factors:
            if r != 0:
                imgdir = os.path.join(subdir_path, "{}_{}".format(subdir, r))
            else:
                imgdir = os.path.join(subdir_path, subdir)
            if not os.path.exists(imgdir):
                needtoload = True

        # 检查分辨率缩小
        for r in resolutions:
            imgdir = os.path.join(subdir_path, "{}_{}x{}".format(subdir, r[1], r[0]))
            if not os.path.exists(imgdir):
                needtoload = True

        if not needtoload:
            continue  # 如果不需要处理图像，则跳过

        imgdir_orig = subdir_path
        imgs = [os.path.join(imgdir_orig, f) for f in sorted(os.listdir(imgdir_orig))]
        imgs = [
            f
            for f in imgs
            if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
        ]

        wd = os.getcwd()

        for r in factors + resolutions:
            if isinstance(r, int):
                name = "{}_{}".format(subdir, r)
                resizearg = "{}%".format(100.0 / r)
            else:
                name = "{}_{}x{}".format(subdir, r[1], r[0])
                resizearg = "{}x{}".format(r[1], r[0])

            imgdir = os.path.join(subdir_path, name)
            if os.path.exists(imgdir):
                continue

            print("Minifying", r, subdir_path)

            os.makedirs(imgdir)
            print(imgdir_orig, imgdir)
            check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)

            ext = imgs[0].split(".")[-1]
            args = " ".join(
                ["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(ext)]
            )
            print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)

            if ext != "png":
                check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
                print("Removed duplicates")
            print("Done")


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    print(basedir)
    # 用于加载图像数据集及其相关的位姿和边界信息，并根据给定的缩放因子或目标宽高调整图像的尺寸
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0_files = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    img1_files = [
        os.path.join(basedir, "enhancewater", f)
        for f in sorted(os.listdir(os.path.join(basedir, "enhancewater")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    if len(img0_files) == 0 or len(img1_files) == 0:
        print("No images found in one of the directories.")
        return

    sh = imageio.imread(img0_files[0]).shape

    sfx = ""

    if factor is not None:
        sfx = "_{}".format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir0 = os.path.join(basedir, "images" + sfx)
    imgdir1 = os.path.join(basedir, "enhancewater" + sfx)
    if not os.path.exists(imgdir0) or not os.path.exists(imgdir1):
        print("One of the image directories does not exist, returning")
        return

    imgfiles0 = [
        os.path.join(imgdir0, f)
        for f in sorted(os.listdir(imgdir0))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    imgfiles1 = [
        os.path.join(imgdir1, f)
        for f in sorted(os.listdir(imgdir1))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    if poses.shape[-1] != len(imgfiles0) or poses.shape[-1] != len(imgfiles1):
        print(
            "Mismatch between imgs and poses!!!! img0: {}, img1: {}, poses: {}".format(
                len(imgfiles0), len(imgfiles1), poses.shape[-1]
            )
        )
        return

    sh = imageio.imread(imgfiles0[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        # if f.endswith("png"):
        #     return imageio.imread(f, ignoregamma=True)
        # else:
            return imageio.imread(f)

    imgs0 = [imread(f)[..., :3] / 255.0 for f in imgfiles0]
    imgs1 = [imread(f)[..., :3] / 255.0 for f in imgfiles1]

    imgs0 = np.stack(imgs0, -1)
    imgs1 = np.stack(imgs1, -1)

    return poses, bds, imgs0,imgs1


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos): #生成一个视图矩阵，用于表示相机的视角和位置
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w): #一组点从世界坐标系转换到相机坐标系
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses): #用于计算一组相机位姿的平均值，返回一个新的相机位姿矩阵

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

#用于生成一系列相机位姿，使相机沿着螺旋路径移动
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):

    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):

    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def transform_pose_llff(poses):
    ret = np.zeros_like(poses)
    ret[:] = poses[:]
    ret[:, 0, 1:3] *= -1
    ret[:, 1:, 3] *= -1
    ret[:, 1:3, 0] *= -1
    return ret


def load_llff_data(
    datadir: str,
    scene_name: str,
    factor: int,
    ndc_coord: bool,
    recenter: bool,
    bd_factor: float,
    spherify: bool,
    llffhold: int,
    path_zflat: bool,
    near: Optional[float],
    far: Optional[float],
):

    basedir = os.path.join(datadir, scene_name)
    poses, bds, imgs0,imgs1 = _load_data(basedir, factor=factor)
    # factor=8 downsamples original imgs by 8x

    # 调整旋转矩阵的顺序，并将可变维度移动到轴0，然后转换数据类型为 float32
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs0 = np.moveaxis(imgs0, -1, 0).astype(np.float32)
    imgs1 = np.moveaxis(imgs1, -1, 0).astype(np.float32)
    images0 = imgs0
    images1 = imgs1
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # 如果提供了 bd_factor，则按比例缩放位姿和边界
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        #计算平均位姿 c2w 和相机上方向 up
        c2w = poses_avg(poses)

        up = normalize(poses[:, :3, 1].sum(0))

        # 计算数据集的合理焦深
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # 计算螺旋路径的半径和其他参数
        shrink_factor = 0.8
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2

        if path_zflat:
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2
        #调用 render_path_spiral 函数生成渲染位姿
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    c2w = poses_avg(poses)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)

    #将图像和位姿数据类型转换为 float32
    images0 = images0.astype(np.float32)
    images1 = images1.astype(np.float32)
    poses = poses.astype(np.float32)

    # 对位姿和渲染位姿进行坐标变换，并提取外参矩阵
    poses = transform_pose_llff(poses)
    render_poses = np.stack(render_poses)[:, :3, :4]
    render_poses = transform_pose_llff(render_poses)
    extrinsics = poses[:, :3, :4]

    if not isinstance(i_test, list):
        i_test = [i_test]

    #提取图像尺寸和相机内参矩阵
    num_frame = len(poses)
    hwf = poses[0, :3, -1]
    h, w, focal = hwf
    h, w = int(h), int(w)
    hwf = [h, w, focal]
    intrinsics = np.array(
        [
            [[focal, 0.0, 0.5 * w], [0.0, focal, 0.5 * h], [0.0, 0.0, 1.0]]
            for _ in range(num_frame)
        ]
    )

    if llffhold > 0:
        i_test = np.arange(num_frame)[::llffhold]

    i_val = i_test
    is_train = lambda i: i not in i_test and i not in i_val
    i_train = np.array([i for i in np.arange(num_frame) if is_train(i)])

    if near is None and far is None:
        near = np.ndarray.min(bds) * 0.9 if not ndc_coord else 0.0
        far = np.ndarray.max(bds) * 1.0 if not ndc_coord else 1.0

    image_sizes = np.array([[h, w] for i in range(num_frame)])

    i_all = np.arange(num_frame)
    i_split = (i_train, i_val, i_test, i_all)

    if ndc_coord:
        ndc_coeffs = (2 * intrinsics[0, 0, 0] / w, 2 * intrinsics[0, 1, 1] / h)
    else:
        ndc_coeffs = (-1.0, -1.0)

    return (
        images0,
        images1,
        intrinsics,
        extrinsics,
        image_sizes,
        near,
        far,
        ndc_coeffs,
        i_split,
        render_poses,
    )
