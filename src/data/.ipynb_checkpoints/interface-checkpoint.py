# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import *

import gin
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.ray_utils import batchified_get_rays
from src.data.sampler import (
    DDPSequnetialSampler,
    MultipleImageDDPSampler,
    MultipleImageDynamicDDPSampler,
    SingleImageDDPSampler,
)


@gin.configurable()
class LitData(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        batch_size: int = 4096,
        chunk: int = 1024 * 32,
        num_workers: int = 16,
        ndc_coord: bool = False,
        batch_sampler: str = "all_images_wo_replace",
        eval_test_only: bool = True,
        epoch_size: int = 50000,
        use_pixel_centers: bool = True,
        white_bkgd: bool = False,
        precrop: bool = False,
        precrop_steps: int = 0,
        scene_center: List[float] = [0.0, 0.0, 0.0],
        scene_radius: List[float] = [1.0, 1.0, 1.0],
        use_sphere_bound: bool = True,
        load_radii: bool = False,
        needs_train_info: bool = False,
        use_near_clip: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitData, self).__init__()
        if not hasattr(self, "multlosses"):
            self.multlosses = None
        if not hasattr(self, "normals"):
            self.normals = None

    def setup(self, stage):

        # Get GPU numbers here
        self.num_devices = self.trainer.num_devices
        
        # def check_and_correct_indices(indices, name):
        #     max_size = len(self.extrinsics)
        #     if max(indices) >= max_size:
        #         print(f"Warning: Index in {name} is out of bounds. Adjusting to max available index.")
        #         return [idx for idx in indices if idx < max_size]
        #     return indices

        # self.i_train = check_and_correct_indices(self.i_train, "i_train")
        # self.i_val = check_and_correct_indices(self.i_val, "i_val")
        # self.i_test = check_and_correct_indices(self.i_test, "i_test")
        
        if stage == "fit" or self.needs_train_info:
            self.train_dset, _ = self.split_each(
            self.images0, self.images1, self.normals, None, self.i_train, dummy=False
        )
            self.train_image_sizes = self.image_sizes[self.i_train]

        if stage == "fit" or stage is None:
            self.val_dset, self.val_dummy = self.split_each(
            self.images0, self.images1, self.normals, None, self.i_val, dummy=True
        )
            self.val_image_sizes = self.image_sizes[self.i_val]

        if stage == "test" or stage is None:
            test_idx = self.i_test if self.eval_test_only else self.i_all
            self.test_dset, self.test_dummy = self.split_each(
            self.images0, self.images1, self.normals, None, test_idx, dummy=True
        )
            self.test_image_sizes = self.image_sizes[self.i_test]
            self.all_image_sizes = self.image_sizes[self.i_all]

        if stage == "predict" or stage is None:
            render_poses = np.stack(self.render_poses)[..., :4]
            self.predict_dset, self.pred_dummy = self.split_each(
                None, None, render_poses, np.arange(len(render_poses))
            )

        if self.use_near_clip:
            self.inward_nearfar_heuristic(self.extrinsics[self.i_train][:, :3, 3])

    # DVGO
    def inward_nearfar_heuristic(self, cam_o, ratio=0.05):
        dist = np.linalg.norm(cam_o[:, None] - cam_o, axis=-1)
        self.far = dist.max()  # could be too small to exist the scene bbox
        # it is only used to determined scene bbox
        # lib/dvgo use 1e9 as far
        self.near_clip = self.far * ratio
        self.near = 0

    def split_each(
        self,
        _images0,
        _images1,
        _normals,
        render_poses,
        idx,
        dummy=True,
    ):
        images0 = None
        images1 = None
        normals = None
        radii = None
        multloss = None

        if _images0 is not None and _images1 is not None:
            extrinsics_idx = self.extrinsics[idx]
            intrinsics_idx = self.intrinsics[idx]
            image_sizes_idx = self.image_sizes[idx]
        else:
            extrinsics_idx = render_poses
            N_render = len(render_poses)
            intrinsics_idx = np.stack([self.intrinsics[0] for _ in range(N_render)])
            image_sizes_idx = np.stack([self.image_sizes[0] for _ in range(N_render)])

        _rays_o, _rays_d, _viewdirs, _radii, _multloss = batchified_get_rays(
            intrinsics_idx,
            extrinsics_idx,
            image_sizes_idx,
            self.use_pixel_centers,
            self.load_radii,
            self.ndc_coord,
            self.ndc_coeffs,
            self.multlosses[idx] if self.multlosses is not None else None,
        )

        device_count = self.num_devices
        n_dset = len(_rays_o)
        dummy_num = (
            (device_count - n_dset % device_count) % device_count if dummy else 0
        )

        rays_o = np.zeros((n_dset + dummy_num, 3), dtype=np.float32)
        rays_d = np.zeros((n_dset + dummy_num, 3), dtype=np.float32)
        viewdirs = np.zeros((n_dset + dummy_num, 3), dtype=np.float32)

        rays_o[:n_dset] = _rays_o
        rays_d[:n_dset] = _rays_d
        viewdirs[:n_dset] = _viewdirs
        
        if dummy_num > 0:
            rays_o[n_dset:] = _rays_o[:dummy_num]
            rays_d[n_dset:] = _rays_d[:dummy_num]
            viewdirs[n_dset:] = _viewdirs[:dummy_num]

        viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=1, keepdims=True)

        if _images0 is not None and _images1 is not None:
            if len(idx) > 0:
                images0_idx = np.concatenate([_images0[i].reshape(-1, 3) for i in idx])
                images1_idx = np.concatenate([_images1[i].reshape(-1, 3) for i in idx])
                images0 = np.zeros((n_dset + dummy_num, 3))
                images1 = np.zeros((n_dset + dummy_num, 3))
                images0[:n_dset] = images0_idx
                images1[:n_dset] = images1_idx
                if dummy_num > 0:
                    images0[n_dset:] = images0[:dummy_num]
                    images1[n_dset:] = images1[:dummy_num]

        if _normals is not None:
            if len(idx) > 0:
                normals_idx = np.concatenate([_normals[i].reshape(-1, 4) for i in idx])
                normals = np.zeros((n_dset + dummy_num, 4))
                normals[:n_dset] = normals_idx
                if dummy_num > 0:
                    normals[n_dset:] = normals[:dummy_num]

        if _radii is not None:
            radii = np.zeros((n_dset + dummy_num, 1), dtype=np.float32)
            radii[:n_dset] = _radii
            if dummy_num > 0:
                radii[n_dset:] = _radii[:dummy_num]

        if _multloss is not None:
            multloss = np.zeros((n_dset + dummy_num, 1), dtype=np.float32)
            multloss[:n_dset] = _multloss
            if dummy_num > 0:
                multloss[n_dset:] = _multloss[:dummy_num]

        rays_info = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "viewdirs": viewdirs,
            "images0": images0,
            "images1": images1,
            "radii": radii,
            "multloss": multloss,
            "normals": normals,
        }

        return RaySet(rays_info), dummy_num

    # Only for DVGO
    def update_masked_sampler(self, mask):

        self.train_dset.rays_o = self.train_dset.rays_o[mask]
        self.train_dset.rays_d = self.train_dset.rays_d[mask]
        self.train_dset.viewdirs = self.train_dset.viewdirs[mask]
        self.train_dataloader().sampler.total_len = len(self.train_dset.rays_o)

    def train_dataloader(self):

        if self.batch_sampler == "single_image":
            sampler = SingleImageDDPSampler(
                batch_size=self.batch_size,
                num_replicas=None,
                rank=None,
                N_img=len(self.i_train),
                N_pixels=self.image_sizes[self.i_train],
                epoch_size=self.epoch_size,
                tpu=False,
                precrop=self.precrop,
                precrop_steps=self.precrop_steps,
            )
        elif self.batch_sampler == "all_images":
            sampler = MultipleImageDDPSampler(
                batch_size=self.batch_size,
                num_replicas=None,
                rank=None,
                total_len=len(self.train_dset),
                epoch_size=self.epoch_size,
                tpu=False,
            )

        elif self.batch_sampler == "dynamic_all_images":
            sampler = MultipleImageDynamicDDPSampler(
                batch_size=self.batch_size,
                num_replicas=None,
                rank=None,
                total_len=len(self.train_dset),
                N_img=len(self.i_train),
                N_pixels=self.image_sizes[self.i_train],
                epoch_size=self.epoch_size,
                tpu=False,
            )

        else:
            raise NameError(f"Unknown batch sampler {self.batch_sampler}")

        return DataLoader(
            dataset=self.train_dset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

    def val_dataloader(self):

        sampler = DDPSequnetialSampler(
            batch_size=self.chunk,
            num_replicas=None,
            rank=None,
            N_total=len(self.val_dset),
            tpu=False,
        )

        return DataLoader(
            dataset=self.val_dset,
            batch_size=self.chunk,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):

        sampler = DDPSequnetialSampler(
            batch_size=self.chunk,
            num_replicas=None,
            rank=None,
            N_total=len(self.test_dset),
            tpu=False,
        )

        return DataLoader(
            dataset=self.test_dset,
            batch_size=self.chunk,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):

        sampler = DDPSequnetialSampler(
            batch_size=self.chunk,
            num_replicas=None,
            rank=None,
            N_total=len(self.predict_dset),
            tpu=False,
        )

        return DataLoader(
            dataset=self.predict_dset,
            batch_size=self.chunk,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )


class RaySet(Dataset):
    def __init__(self, rays_info):

        # Image
        self.images0 = rays_info["images0"]
        self.images1 = rays_info["images1"]

        # Ray offset and direction
        self.rays_o = rays_info["rays_o"]
        self.rays_d = rays_info["rays_d"]
        self.viewdirs = rays_info["viewdirs"]

        # Ray radii (for MipNeRF)
        self.radii = rays_info["radii"]

        # MultLoss (for MipNeRF)
        self.multloss = rays_info["multloss"]

        # Normals (for RefNeRF)
        self.normals = rays_info["normals"]

        self.N = len(self.rays_d)

    def __getitem__(self, index):
        ret = {
            "rays_o": torch.from_numpy(self.rays_o[index]),
            "rays_d": torch.from_numpy(self.rays_d[index]),
            "viewdirs": torch.from_numpy(self.viewdirs[index]),
            "target0": torch.zeros_like(torch.from_numpy(self.rays_o[index])),
            "target1": torch.zeros_like(torch.from_numpy(self.rays_o[index])),
            "radii": torch.zeros((self.rays_o[index].shape[0], 1)),
            "multloss": torch.zeros((self.rays_o[index].shape[0], 1)),
            "normals": torch.zeros_like(torch.from_numpy(self.rays_o[index])),
        }

        if self.images0 is not None:
            ret["target0"] = torch.from_numpy(self.images0[index])
            # print(index)

        if self.images1 is not None:
            ret["target1"] = torch.from_numpy(self.images1[index])

        if self.radii is not None:
            ret["radii"] = torch.from_numpy(self.radii[index])

        if self.multloss is not None:
            ret["multloss"] = torch.from_numpy(self.multloss[index])

        if self.normals is not None:
            ret["normals"] = torch.from_numpy(self.normals[index])

        return ret

    def __len__(self):
        return self.N
