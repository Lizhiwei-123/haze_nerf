# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from random import random
from typing import *

import math
import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import src.model.nerf.helper as helper # 导入自定义的辅助函数
from utils.store_image import store_image,store_video # 导入存储图像的工具函数
from src.model.interface import LitModel # 导入LitModel接口
from torchvision.models import vgg16

@gin.configurable()
class NeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        #如果索引 idx 是 skip_layer 的倍数且大于0，则在该层的输入中添加原始输入，以实现跳跃连接。
        for idx in range(netdepth - 1):  #
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer_1 = nn.Linear(netwidth_condition, num_rgb_channels)
        self.rgb_layer_2 = nn.Linear(netwidth_condition, num_rgb_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer_1.weight)
        init.xavier_uniform_(self.rgb_layer_2.weight)

    def forward(self, x, condition):

        #获取 x 的形状信息，并将 x 重塑为二维张量，同时保留原始输入 inputs
        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb_1 = self.rgb_layer_1(x).reshape(-1, num_samples, self.num_rgb_channels)
        raw_rgb_2 = self.rgb_layer_2(x).reshape(-1, num_samples, self.num_rgb_channels)
        return raw_rgb_1,raw_rgb_2, raw_density

# class LossNetwork(nn.Module):
#     def __init__(self, vgg_model):
#         super(LossNetwork, self).__init__()
#         self.vgg_layers = vgg_model
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '15': "relu3_3"
#         }

#     def output_features(self, x):
#         output = {}
#         for name, module in self.vgg_layers._modules.items():
#             x = module(x)
#             if name in self.layer_name_mapping:
#                 output[self.layer_name_mapping[name]] = x
#         return list(output.values())

#     def forward(self, pred_im, gt):
#         loss = []
#         pred_im_features = self.output_features(pred_im)
#         gt_features = self.output_features(gt)
#         for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
#             loss.append(F.mse_loss(pred_im_feature, gt_feature))

#         return sum(loss)/len(loss)
        
#         # Loss & Optimizer Setting & Metric
#         vgg_model = vgg16(pretrained=True).features[:16]
#         vgg_model = vgg_model.cuda()

#         for param in vgg_model.parameters():
#             param.requires_grad = False

#         loss_network = LossNetwork(vgg_model)
#         loss_network.eval()    

@gin.configurable()
class NeRF(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,
        noise_std: float = 0.0,
        lindisp: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRF, self).__init__()

        self.rgb_activation = nn.Sigmoid()
        self.sigma_activation = nn.ReLU()
        self.coarse_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        self.fine_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        self.A = nn.Parameter(torch.FloatTensor([0.2, 0.2, 0.2]), requires_grad=True)
        
        self.B = nn.Parameter(torch.FloatTensor([1e-5]), requires_grad=True)
        
        
    def forward(self, rays, randomized, white_bkgd, near, far):
        print(f"A: {self.A}, B: {self.B}")
        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                mlp = self.coarse_mlp

            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples = helper.sample_pdf(
                    bins=t_mids,
                    weights=weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                )
                mlp = self.fine_mlp

            samples_enc = helper.pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
            )
            viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)

            raw_rgb_1,raw_rgb_2, raw_sigma = mlp(samples_enc, viewdirs_enc)

            if self.noise_std > 0 and randomized:
                raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std

            rgb_1 = self.rgb_activation(raw_rgb_1)
            rgb_2 = self.rgb_activation(raw_rgb_2)
            sigma = self.sigma_activation(raw_sigma)

            comp_rgb_1, acc, weights,depth = helper.volumetric_rendering(
                rgb_1,
                sigma,
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
                
                
            )
            t = torch.exp(-self.B * depth)
            t_expanded = t.unsqueeze(-1)
            comp_rgb_aug = (comp_rgb_1 - self.A * (1 -  t_expanded)) / t_expanded
            # print(comp_rgb_aug)
            
            comp_rgb_2, acc, weights,_ = helper.volumetric_rendering(
                rgb_2,
                sigma,
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )
            ret.append((comp_rgb_1,comp_rgb_2,comp_rgb_aug ,acc,depth))
           
        
        return ret


@gin.configurable()
class LitNeRF(LitModel):
    def __init__(
        self,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
        # B_init: float = 1e-5
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitNeRF, self).__init__()
        self.model = NeRF()
        # self.B = nn.Parameter(torch.FloatTensor([B_init]), requires_grad=True)
        
        
        
        # vgg_model = vgg16(pretrained=True).features
        # self.loss_network = LossNetwork(vgg_model)
        
    def setup(self, stage: Optional[str] = None) -> None:
        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.white_bkgd = self.trainer.datamodule.white_bkgd

    def training_step(self, batch, batch_idx):

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        rgb_coarse_water = rendered_results[0][0]
        rgb_fine_water = rendered_results[1][0]
        rgb_coarse_enhance = rendered_results[0][1]
        rgb_fine_enhance = rendered_results[1][1]
        rgb_coarse_aug = rendered_results[0][2]
        rgb_fine_aug = rendered_results[1][2]
        depth_coarse = rendered_results[0][-1]
        depth_fine = rendered_results[1][-1]
        target0 = batch["target0"]
        target1 = batch["target1"]
        
        # epsilon = 1e-10
        # A_expanded = self.A
        # A_expanded = self.A.view(1, 1, 3).expand(rgb_coarse_water.size(0), 1, 3)
        # t_coarse_expanded = t_coarse.unsqueeze(-1)
        # t_fine_expanded = t_fine.unsqueeze(-1)
        # J_coarse = (rgb_coarse_water - A_expanded * (1 - t_coarse_expanded)) / t_coarse_expanded
        # J_fine =  (rgb_fine_water - A_expanded * (1 - t_fine_expanded)) / t_fine_expanded
        
        
        
        # print("J_fine shape before reshaping:", J_fine.shape)
        # print("rgb_fine_enhance shape before reshaping:", rgb_fine_enhance.shape)
        # print("rgb_fine_enhance shape before reshaping:", rgb_coarse_enhance.shape)
        # print("rgb_fine_enhance shape before reshaping:", rgb_coarse_water.shape)
        # print("rgb_fine_enhance shape before reshaping:", rgb_fine_water.shape)
        
#         batch_size = 1  # 假设 batch_size 是 1
#         num_pixels = J_fine.size(0) * J_fine.size(1) // 3  # 假设有3个通道
#         channels = J_fine.size(1)  # 假设有3个通道（RGB）
        
        # img_size = int(math.sqrt(num_pixels))
        # if img_size * img_size  != num_pixels:
        #     raise ValueError(f"Cannot reshape J_fine to image dimensions: {img_size}x{img_size}")
        # print(f"Batch size: {batch_size}, Number of pixels: {num_pixels}, Channels: {channels}")
        
        # J_fine = J_fine.view(batch_size, img_size, img_size, channels).permute(0, 3, 1, 2)
        # rgb_fine_enhance = rgb_fine_enhance.view(batch_size, img_size, img_size, channels).permute(0, 3, 1, 2)
        
        # print("J_fine shape after reshaping for VGG:", J_fine_vgg.shape)
        # print("rgb_fine_enhance shape after reshaping for VGG:", rgb_fine_enhance_vgg.shape)
        
        
        
        loss_coarse_water = helper.img2mse(rgb_coarse_water, target0)
        loss_fine_water = helper.img2mse(rgb_fine_water, target0)
        loss_coarse_enhance = helper.img2mse(rgb_coarse_enhance, target1)
        loss_fine_enhance = helper.img2mse(rgb_fine_enhance, target1)
        
        loss_coarse_aug = helper.img2mse(rgb_coarse_aug,rgb_coarse_enhance)
        loss_fine_aug = helper.img2mse(rgb_fine_aug,rgb_fine_enhance)
        
        loss = loss_coarse_water + loss_fine_water + loss_coarse_enhance + loss_fine_enhance + 0.1 * loss_coarse_aug + 0.1 * loss_fine_aug
        

        psnr_coarse_water = helper.mse2psnr(loss_coarse_water)
        psnr_fine_water = helper.mse2psnr(loss_fine_water)
        psnr_coarse_enhance = helper.mse2psnr(loss_coarse_enhance)
        psnr_fine_enhance = helper.mse2psnr(loss_fine_enhance)

        self.log("train/psnr_coarse_water", psnr_coarse_water, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr_fine_water", psnr_fine_water, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr_coarse_enhance", psnr_coarse_enhance, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr_fine_enhance", psnr_fine_enhance, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)

        return {"loss": loss}

    def render_rays(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far
        )
        rgb_fine_water = rendered_results[1][0]
        rgb_fine_enhance = rendered_results[1][1]
        rgb_fine_aug = rendered_results[1][2]
        target0 = batch["target0"]
        target1 = batch["target1"]
        ret["target0"] = target0
        ret["target1"] = target1
        ret["rgb_fine_water"] = rgb_fine_water
        ret["rgb_fine_enhance"] = rgb_fine_enhance
        ret["rgb_fine_aug"] = rgb_fine_aug
        
        return {
            "rgb_fine_water": rgb_fine_water,
            "rgb_fine_enhance": rgb_fine_enhance,
            "rgb_fine_aug" : rgb_fine_aug,
            "target0": target0,
            "target1": target1
        }

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = gin.query_parameter("run.max_steps")

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def validation_epoch_end(self, outputs):
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs_water = self.alter_gather_cat(outputs, "rgb_fine_water", val_image_sizes)
        rgbs_enhance = self.alter_gather_cat(outputs, "rgb_fine_enhance", val_image_sizes)
        rgbs_aug = self.alter_gather_cat(outputs, "rgb_fine_aug", val_image_sizes)
        targets_water = self.alter_gather_cat(outputs, "target0", val_image_sizes)
        targets_enhance = self.alter_gather_cat(outputs, "target1", val_image_sizes)
       
        psnr_mean_water = self.psnr_each(rgbs_water, targets_water).mean()
        psnr_mean_enhance = self.psnr_each(rgbs_enhance, targets_enhance).mean()
        ssim_mean_water = self.ssim_each(rgbs_water, targets_water).mean()
        ssim_mean_enhance = self.ssim_each(rgbs_enhance, targets_enhance).mean()
        lpips_mean_water = self.lpips_each(rgbs_water, targets_water).mean()
        lpips_mean_enhance = self.lpips_each(rgbs_enhance, targets_enhance).mean()
        
        psnr_mean = (psnr_mean_water + psnr_mean_enhance) / 2
        
        self.log("val/psnr_water", psnr_mean_water.item(), on_epoch=True, sync_dist=True)
        self.log("val/psnr_enhance", psnr_mean_enhance.item(), on_epoch=True, sync_dist=True)
        self.log("val/ssim_water", ssim_mean_water.item(), on_epoch=True, sync_dist=True)
        self.log("val/ssim_enhance", ssim_mean_enhance.item(), on_epoch=True, sync_dist=True)
        self.log("val/lpips_water", lpips_mean_water.item(), on_epoch=True, sync_dist=True)
        self.log("val/lpips_enhance", lpips_mean_enhance.item(), on_epoch=True, sync_dist=True)
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        # return psnr_mean_water, psnr_mean_enhance, ssim_mean_water, ssim_mean_enhance, lpips_mean_water, lpips_mean_enhance
    
    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )
        rgbs_water = self.alter_gather_cat(outputs, "rgb_fine_water", all_image_sizes)
        rgbs_enhance = self.alter_gather_cat(outputs, "rgb_fine_enhance", all_image_sizes)
        rgbs_aug = self.alter_gather_cat(outputs, "rgb_fine_aug", all_image_sizes)
        targets_water = self.alter_gather_cat(outputs, "target0", all_image_sizes)
        targets_enhance = self.alter_gather_cat(outputs, "target1", all_image_sizes)
        psnr_water = self.psnr(rgbs_water, targets_water, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        psnr_enhance = self.psnr(rgbs_enhance, targets_enhance, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim_water = self.ssim(rgbs_water, targets_water, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim_enhance = self.ssim(rgbs_enhance, targets_enhance, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips_water = self.lpips(rgbs_water, targets_water, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips_enhance = self.lpips(rgbs_enhance, targets_enhance, dmodule.i_train, dmodule.i_val, dmodule.i_test)

        self.log("test/psnr_water", psnr_water["test"], on_epoch=True)
        self.log("test/psnr_enhance", psnr_enhance["test"], on_epoch=True)
        self.log("test/ssim_water", ssim_water["test"], on_epoch=True)
        self.log("test/ssim_enhance", ssim_enhance["test"], on_epoch=True)
        self.log("test/lpips_water", lpips_water["test"], on_epoch=True)
        self.log("test/lpips_enhance", lpips_enhance["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            water_dir = os.path.join(self.logdir, "render_model/water")
            enhance_dir = os.path.join(self.logdir, "render_model/enhance")
            aug_dir = os.path.join(self.logdir, "render_model/aug")
            store_image(water_dir, rgbs_water)
            store_image(enhance_dir, rgbs_enhance)
            store_image(aug_dir , rgbs_aug)
            water_dir = os.path.join(self.logdir, "render_video/water")
            enhance_dir = os.path.join(self.logdir, "render_video/enhance")
            store_video(water_dir, rgbs_water)
            store_video(enhance_dir, rgbs_enhance)

            result_path = os.path.join(self.logdir, "results.json")
            self.write_stats(result_path, psnr_water, ssim_water, lpips_water)
            self.write_stats(result_path, psnr_enhance, ssim_enhance, lpips_enhance)

        return psnr_water, ssim_water, lpips_water, psnr_enhance, ssim_enhance, lpips_enhance
