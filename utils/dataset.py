import math
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from glob import glob
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import yaml

# Visualization for Debugging
from matplotlib import pyplot as plt
from pandas import read_csv
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from utils.augmentations import *
from utils.torch_utils import *
from utils.track_utils import PoseInterpolator, retrieve_track_tuples
from utils.utils import *

SUPPORTED_REPRESENTATIONS = [
    "time_surfaces_v2_5",
    "voxel_grids_5",
    "event_stacks_5",
    "event_stacks_normed_5",
]
MAX_ROTATION_ANGLE = 15
MAX_SCALE_CHANGE_PERCENTAGE = 20
MAX_PERSPECTIVE_THETA = 0.01
MAX_TRANSLATION = 3

torch.multiprocessing.set_sharing_strategy("file_system")


class InputModality(Enum):
    frame = 0
    event = 1


# Data Classes for Baseline Training
@dataclass
class TrackDataConfig:
    frame_paths: list
    event_paths: list
    patch_size: int
    representation: str
    track_name: str
    augment: bool


def recurrent_collate(batch_dataloaders):
    return batch_dataloaders

class TrackletAugmentor:

    def __init__(self, H, W, kernel_size=7, sigma=1.5):
        self.H = H
        self.W = W

        # ===== Gaussian kernel =====
        half_k = kernel_size // 2
        coords = torch.arange(kernel_size) - half_k
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')

        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        self.kernel = kernel / kernel.max()   # [7,7]
        self.kernel_size = kernel_size
        self.half_k = half_k

    def draw_gaussian(self, heatmap, y, x):
        """在 heatmap 上画一个高斯"""
        H, W = self.H, self.W
        k = self.kernel.to(heatmap.device)

        y = int(y)
        x = int(x)

        y0 = max(0, y - self.half_k)
        y1 = min(H, y + self.half_k + 1)
        x0 = max(0, x - self.half_k)
        x1 = min(W, x + self.half_k + 1)

        ky0 = self.half_k - (y - y0)
        ky1 = self.half_k + (y1 - y)
        kx0 = self.half_k - (x - x0)
        kx1 = self.half_k + (x1 - x)

        heatmap[y0:y1, x0:x1] = torch.maximum(
            heatmap[y0:y1, x0:x1],
            k[ky0:ky1, kx0:kx1]
        )

    def augment_and_generate(
            self,
            centers,  # [B,N,2]
            lambda_jt=0.05,
            lambda_fn=0.4,
            lambda_fp=0.2,
            max_fp=3
    ):
        """
        输出: heatmap [B,1,H,W]
        """

        B, N, _ = centers.shape
        device = centers.device

        heatmap = torch.zeros(B, self.H, self.W, device=device)

        for b in range(B):

            pts = centers[b].clone()

            # =========================
            # 过滤无效点
            # =========================
            valid = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
            pts = pts[valid]

            if pts.numel() == 0:
                continue

            # =========================
            # 1. jitter（始终执行）
            # =========================
            dy = torch.randn(len(pts), device=device) * (lambda_jt * self.H)
            dx = torch.randn(len(pts), device=device) * (lambda_jt * self.W)

            pts[:, 0] = (pts[:, 0] + dy).round().clamp(0, self.H - 1)
            pts[:, 1] = (pts[:, 1] + dx).round().clamp(0, self.W - 1)

            # =========================
            # 2. false negative（概率删除）
            # =========================
            keep_mask = torch.rand(len(pts), device=device) > lambda_fn
            pts_keep = pts[keep_mask]

            # =========================
            # 3. 画真实目标
            # =========================
            for p in pts_keep:
                y, x = int(p[0]), int(p[1])
                self.draw_gaussian(heatmap[b], y, x)

            # =========================
            # 4. spurious peak（邻域伪目标）
            # =========================
            for p in pts_keep:

                if torch.rand(1).item() < lambda_fp:

                    # 每个真实点附近生成1~max_fp个假点
                    num_fp = torch.randint(1, max_fp + 1, (1,)).item()

                    for _ in range(num_fp):
                        # 在邻域加扰动（关键改动）
                        dy_fp = torch.randn(1).item() * (lambda_jt * self.H)
                        dx_fp = torch.randn(1).item() * (lambda_jt * self.W)

                        fy = int((p[0] + dy_fp))
                        fx = int((p[1] + dx_fp))

                        # 边界裁剪
                        fy = max(0, min(self.H - 1, fy))
                        fx = max(0, min(self.W - 1, fx))

                        self.draw_gaussian(heatmap[b], fy, fx)

        return heatmap.unsqueeze(1)

class TrackData:
    """
    Dataloader for a single feature track. Returns input patches and displacement labels relative to
    the current feature location. Current feature location is either updated manually via accumulate_y_hat()
    or automatically via the ground-truth displacement.
    """

    def __init__(self, track_tuple, config):
        """
        Dataset for a single feature track
        :param track_tuple: (Path to track.gt.txt, track_id)
        :param config:
        """
        self.config = config

        # Track augmentation (disabled atm)
        if False:
            # if config.augment:
            self.flipped_lr = random.choice([True, False])
            self.flipped_ud = random.choice([True, False])
            # self.rotation_angle = round(random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE))
            self.rotation_angle = 0
        else:
            self.flipped_lr, self.flipped_ud, self.rotation_angle = False, False, 0
        self.last_aug_angle, self.last_aug_scale = 0.0, 1.0

        # Get input paths
        self.frame_paths = config.frame_paths
        self.event_paths = config.event_paths

        # TODO: Do this in a non-hacky way
        if "0.0100" in self.event_paths[0]:
            self.index_multiplier = 1
        elif "0.0200" in self.event_paths[0]:
            self.index_multiplier = 2
        else:
            print("Unsupported dt for feature track")
            raise NotImplementedError


        self.track_path = track_tuple
        raw_data = np.genfromtxt(self.track_path)
        track_ids = raw_data[:, 0].astype(int)
        coords = raw_data[:, 2:]
        unique_ids = np.unique(track_ids)
        self.num_tracks = len(unique_ids)

        if len(raw_data) % self.num_tracks == 0:
            # 可以直接 reshape（最快）
            self.seq_len = len(raw_data) // self.num_tracks
            coords = coords.reshape(self.num_tracks, self.seq_len, 2)

        else:
            # ===== 不等长 track，需要 padding =====
            track_list = []
            max_len = 0

            for tid in unique_ids:
                track_coords = coords[track_ids == tid]
                track_list.append(track_coords)
                max_len = max(max_len, len(track_coords))

            padded_tracks = []

            for track_coords in track_list:
                T = len(track_coords)

                if T < max_len:
                    last_coord = track_coords[-1]
                    pad = np.tile(last_coord, (max_len - T, 1))
                    track_coords = np.concatenate([track_coords, pad], axis=0)

                padded_tracks.append(track_coords)

            coords = np.stack(padded_tracks, axis=0)

            self.seq_len = max_len
        # 转成 [T, N_tracks, 2]
        self.track_data = np.transpose(coords, (1, 0, 2))

        # 随机初始值
        if "test" in self.event_paths[0]:
            self.start_idx = int(0)

        elif "train_fune_2+3" in self.event_paths[0]:
            self.start_idx = random.randint(0, self.track_data.shape[0] - 45)

        elif "train" in self.event_paths[0]:
            self.start_idx = random.randint(0, self.track_data.shape[0] - 7)

        # 对齐 track
        self.track_data = self.track_data[self.start_idx:]
        self.seq_len = self.track_data.shape[0]

        # 对齐 frame_paths
        self.frame_paths = self.frame_paths[self.start_idx:]

        # 对齐 event_paths
        self.event_paths = self.event_paths[self.start_idx:]

        # 重新读取 ref_input
        ref_input = read_input(self.frame_paths[0], "grayscale")
        '''ref_input = augment_input(
            ref_input, self.flipped_lr, self.flipped_ud, self.rotation_angle)'''

        '''self.track_data = augment_track(
            self.track_data,
            self.flipped_lr,
            self.flipped_ud,
            self.rotation_angle,
            (ref_input.shape[1], ref_input.shape[0]),
        )'''

        self.u_center = self.track_data[0]
        self.u_center_gt = self.track_data[0]
        self.u_center_init = self.track_data[0]

        #self.x_ref = get_patch_voxel(ref_input, self.u_center, config.patch_size)
        if len(ref_input.shape) == 2:
            self.x_ref = np.array(ref_input).astype(np.float32)
            self.x_ref = np.expand_dims(self.x_ref, axis=2)
        else:
            self.x_ref = np.array(ref_input).astype(np.float32)

        self.x_ref = np.transpose(self.x_ref, (2, 0, 1))
        self.x_ref = torch.from_numpy(self.x_ref)

        # Pathing for input data
        self.seq_name = Path(self.track_path).parents[1].stem

        # Operational
        self.time_idx = 0
        self.auto_update_center = False

        # Representation-specific Settings
        if "grayscale" in config.representation:
            self.channels_in_per_patch = 1
        else:
            self.channels_in_per_patch = int(config.representation[-1])
            # in v2, we have separate temporal bins for each event polarity
            if "v2" in config.representation:
                self.channels_in_per_patch *= 2

    def reset(self):
        self.time_idx = 0
        self.u_center = self.u_center_init

    def accumulate_y_hat(self, y_hat):
        """
        Accumulate predicted flows if using predictions instead of gt patches
        :param y_hat: 2-element Tensor
        """
        # Disregard confidence
        #y_hat = y_hat[:2]

        # Unaugment the predicted label
        if self.config.augment:
            # y_hat = unaugment_perspective(y_hat.detach().cpu(), self.last_aug_perspective[0], self.last_aug_perspective[1])
            y_hat = unaugment_rotation(y_hat.detach().cpu(), self.last_aug_angle)
            y_hat = unaugment_scale(y_hat, self.last_aug_scale)

            # Translation augmentation
            y_hat += (2 * torch.rand_like(y_hat) - 1) * MAX_TRANSLATION

        self.u_center += y_hat.detach().cpu().numpy()#.reshape((2,))

    def get_next(self):
        # Increment time
        #self.time_idx += 2
        self.time_idx +=2

        # Round feature location to accommodate get_patch_voxel
        self.u_center = np.rint(self.u_center)

        # Update gt location
        #self.u_center_gt = self.track_data[self.time_idx * self.index_multiplier, :]
        self.u_center_gt = self.track_data[self.time_idx]

        # Update total flow
        y = (self.u_center_gt - self.u_center).astype(np.float32)
        y = torch.from_numpy(y)
        #self.u_center_gt = torch.from_numpy(self.u_center_gt)

        # # Update xref (Uncomment if combining frames with events)
        # if self.time_idx % 5 == 0:
        #     frame_idx = self.time_idx // 5
        #     ref_input = read_input(self.frame_paths[frame_idx], 'grayscale')
        #     self.x_ref = get_patch_voxel2(ref_input, self.u_center, self.config.patch_size)

        # Get patch inputs for event representation
        input_1 = read_input(
            self.event_paths[self.time_idx], self.config.representation
        )
        input_1 = augment_input(
            input_1, self.flipped_lr, self.flipped_ud, self.rotation_angle
        )
        #x = get_patch_voxel(input_1, self.u_center, self.config.patch_size)
        if len(input_1.shape) == 2:
            x = np.array(input_1).astype(np.float32)
            x = np.expand_dims(x, axis=2)
        else:
            x = np.array(input_1).astype(np.float32)
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)
        x = torch.cat([x, self.x_ref], dim=0)

        # Augmentation
        if self.config.augment:
            # Sample rotation and scale
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_scaling,
            ) = augment_scale(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_scale_percentage=MAX_SCALE_CHANGE_PERCENTAGE,
            )
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_angle,
            ) = augment_rotation(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_rotation_deg=MAX_ROTATION_ANGLE,
            )
            # x[0:self.channels_in_per_patch, :, :], y, self.last_aug_perspective = augment_perspective(x[0:self.channels_in_per_patch, :, :], y,
            #                                                                                           theta=MAX_PERSPECTIVE_THETA)

        # Update center location for next patch
        if self.auto_update_center:
            #self.u_center = self.u_center + y.numpy().reshape((2,))
            self.u_center = self.u_center + y.numpy()

        # Minor Processing Steps
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        #u_center = torch.from_numpy(self.u_center)
        #u_center = torch.unsqueeze(u_center, 0)\
        u_center_gt = torch.from_numpy(self.u_center_gt)
        u_center_gt = torch.unsqueeze(u_center_gt, 0)

        return x, y, u_center_gt

    def get_prev(self):
        # =========================
        # Decrement time
        # =========================
        self.time_idx -= 2
        if self.time_idx < 0:
            self.time_idx = 0

        # Round feature location
        self.u_center = np.rint(self.u_center)

        # =========================
        # Get GT at current time
        # =========================
        self.u_center_gt = self.track_data[self.time_idx]

        # =========================
        # Compute reverse flow
        # =========================
        # 注意：方向反过来
        y = (self.u_center_gt - self.u_center).astype(np.float32)
        y = torch.from_numpy(y)

        # =========================
        # Load input
        # =========================
        input_1 = read_input(
            self.event_paths[self.time_idx], self.config.representation
        )
        input_1 = augment_input(
            input_1, self.flipped_lr, self.flipped_ud, self.rotation_angle
        )

        if len(input_1.shape) == 2:
            x = np.array(input_1).astype(np.float32)
            x = np.expand_dims(x, axis=2)
        else:
            x = np.array(input_1).astype(np.float32)

        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)
        x = torch.cat([x, self.x_ref], dim=0)

        # =========================
        # Augmentation (保持一致)
        # =========================
        if self.config.augment:
            (
                x[0: self.channels_in_per_patch, :, :],
                y,
                self.last_aug_scaling,
            ) = augment_scale(
                x[0: self.channels_in_per_patch, :, :],
                y,
                max_scale_percentage=MAX_SCALE_CHANGE_PERCENTAGE,
            )
            (
                x[0: self.channels_in_per_patch, :, :],
                y,
                self.last_aug_angle,
            ) = augment_rotation(
                x[0: self.channels_in_per_patch, :, :],
                y,
                max_rotation_deg=MAX_ROTATION_ANGLE,
            )

        # =========================
        # Update center (reverse)
        # =========================
        if self.auto_update_center:
            self.u_center = self.u_center + y.numpy()

        # =========================
        # Format output
        # =========================
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)

        u_center_gt = torch.from_numpy(self.u_center_gt)
        u_center_gt = torch.unsqueeze(u_center_gt, 0)

        return x, y, u_center_gt
'''
class TrackData:
    """
    Dataloader for multiple feature tracks.

    Returns:
        x:           [1, C, H, W]
        y:           [1, N_tracks, 2]
        u_center_gt: [1, N_tracks, 2]

    track_data:
        [T, N_tracks, 2]
    """

    def __init__(self, track_tuple, config):
        """
        :param track_tuple: Path to track.gt.txt
        :param config:
        """
        self.config = config

        # Track augmentation
        # 当前仍保持原始逻辑：这里 if False，因此不会执行随机 flip / rotation
        if False:
            self.flipped_lr = random.choice([True, False])
            self.flipped_ud = random.choice([True, False])
            self.rotation_angle = 0
        else:
            self.flipped_lr, self.flipped_ud, self.rotation_angle = False, False, 0

        self.last_aug_angle, self.last_aug_scale = 0.0, 1.0

        # Get input paths
        self.frame_paths = config.frame_paths
        self.event_paths = config.event_paths

        # TODO: Do this in a non-hacky way
        if "0.0100" in self.event_paths[0]:
            self.index_multiplier = 1
        elif "0.0200" in self.event_paths[0]:
            self.index_multiplier = 2
        else:
            print("Unsupported dt for feature track")
            raise NotImplementedError

        # =========================================================
        # Load multi-track GT
        # raw_data format:
        #     track_id time x y
        # =========================================================
        self.track_path = track_tuple

        raw_data = np.genfromtxt(self.track_path)

        if raw_data.ndim == 1:
            raw_data = raw_data[None, :]

        track_ids = raw_data[:, 0].astype(int)
        coords = raw_data[:, 2:]
        unique_ids = np.unique(track_ids)
        self.num_tracks = len(unique_ids)

        if len(raw_data) % self.num_tracks == 0:
            # 等长 track，可以直接 reshape
            self.seq_len = len(raw_data) // self.num_tracks
            coords = coords.reshape(self.num_tracks, self.seq_len, 2)
        else:
            # 不等长 track，需要 padding 到相同长度
            track_list = []
            max_len = 0

            for tid in unique_ids:
                track_coords = coords[track_ids == tid]
                track_list.append(track_coords)
                max_len = max(max_len, len(track_coords))

            padded_tracks = []

            for track_coords in track_list:
                T = len(track_coords)

                if T < max_len:
                    last_coord = track_coords[-1]
                    pad = np.tile(last_coord, (max_len - T, 1))
                    track_coords = np.concatenate([track_coords, pad], axis=0)

                padded_tracks.append(track_coords)

            coords = np.stack(padded_tracks, axis=0)
            self.seq_len = max_len

        # [N_tracks, T, 2] -> [T, N_tracks, 2]
        self.track_data = np.transpose(coords, (1, 0, 2))

        # =========================================================
        # Random start index
        # =========================================================
        if "test" in self.event_paths[0]:
            self.start_idx = 0
        elif "train" in self.event_paths[0]:
            self.start_idx = random.randint(50, self.track_data.shape[0] - 50)
        else:
            self.start_idx = 0

        # Align track
        self.track_data = self.track_data[self.start_idx:]
        self.seq_len = self.track_data.shape[0]

        # Align frame paths
        self.frame_paths = self.frame_paths[self.start_idx:]

        # Align event paths
        self.event_paths = self.event_paths[self.start_idx:]

        # =========================================================
        # Reference frame
        # =========================================================
        ref_input = read_input(self.frame_paths[0], "grayscale")
        ref_input = augment_input(
            ref_input,
            self.flipped_lr,
            self.flipped_ud,
            self.rotation_angle,
        )

        self.track_data = augment_track(
            self.track_data,
            self.flipped_lr,
            self.flipped_ud,
            self.rotation_angle,
            (ref_input.shape[1], ref_input.shape[0]),
        )

        self.u_center = self.track_data[0].copy()
        self.u_center_gt = self.track_data[0].copy()
        self.u_center_init = self.track_data[0].copy()

        # 第二个文件中使用整幅 ref_input，而不是 get_patch_voxel
        if len(ref_input.shape) == 2:
            self.x_ref = np.array(ref_input).astype(np.float32)
            self.x_ref = np.expand_dims(self.x_ref, axis=2)
        else:
            self.x_ref = np.array(ref_input).astype(np.float32)

        self.x_ref = np.transpose(self.x_ref, (2, 0, 1))
        self.x_ref = torch.from_numpy(self.x_ref)

        # Pathing for input data
        self.seq_name = Path(self.track_path).parents[1].stem

        # Operational
        self.time_idx = 0
        self.auto_update_center = False

        # Representation-specific Settings
        if "grayscale" in config.representation:
            self.channels_in_per_patch = 1
        else:
            self.channels_in_per_patch = int(config.representation[-1])
            # in v2, we have separate temporal bins for each event polarity
            if "v2" in config.representation:
                self.channels_in_per_patch *= 2

    def reset(self):
        self.time_idx = 0
        self.u_center = self.u_center_init.copy()
        self.u_center_gt = self.u_center_init.copy()

    def accumulate_y_hat(self, y_hat):
        """
        Accumulate predicted flows if using predictions instead of gt.

        y_hat:
            [N_tracks, 2]
        """
        if self.config.augment:
            y_hat = unaugment_rotation(
                y_hat.detach().cpu(),
                self.last_aug_angle,
            )

            y_hat = unaugment_scale(
                y_hat,
                self.last_aug_scale,
            )

            # Translation augmentation
            y_hat += (2 * torch.rand_like(y_hat) - 1) * MAX_TRANSLATION

        self.u_center += y_hat.detach().cpu().numpy()

    def get_next(self):
        # Increment time
        self.time_idx += 2

        # 防止索引越界
        if self.time_idx >= len(self.event_paths):
            self.time_idx = len(self.event_paths) - 1

        if self.time_idx >= self.track_data.shape[0]:
            self.time_idx = self.track_data.shape[0] - 1

        # Round feature location
        self.u_center = np.rint(self.u_center)

        # Update gt location
        self.u_center_gt = self.track_data[self.time_idx].copy()

        # Update total flow
        y = (self.u_center_gt - self.u_center).astype(np.float32)
        y = torch.from_numpy(y)

        # =========================================================
        # Read event representation
        # =========================================================
        input_1 = read_input(
            self.event_paths[self.time_idx],
            self.config.representation,
        )

        input_1 = augment_input(
            input_1,
            self.flipped_lr,
            self.flipped_ud,
            self.rotation_angle,
        )

        if len(input_1.shape) == 2:
            x = np.array(input_1).astype(np.float32)
            x = np.expand_dims(x, axis=2)
        else:
            x = np.array(input_1).astype(np.float32)

        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)

        # Concatenate event input and reference frame
        x = torch.cat([x, self.x_ref], dim=0)

        # =========================================================
        # Event augmentation
        # 和第一份代码一致：对事件通道和 y 同步做 scale / rotation
        # =========================================================
        if self.config.augment:
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                aug_scale,
            ) = augment_scale(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_scale_percentage=MAX_SCALE_CHANGE_PERCENTAGE,
            )
            self.last_aug_scale = aug_scale[0]
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_angle,
            ) = augment_rotation(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_rotation_deg=MAX_ROTATION_ANGLE,
            )

        # =========================================================
        # 关键修改：
        # y 如果经过 scale / rotation 增强，
        # 则 u_center_gt 也必须根据增强后的 y 同步更新。
        #
        # 不额外增加 u_center_before_aug 变量，
        # 直接沿用当前 self.u_center。
        # =========================================================
        y_np = y.detach().cpu().numpy().astype(np.float32)
        self.u_center_gt = self.u_center.astype(np.float32) + y_np

        # Update center location for next patch
        if self.auto_update_center:
            self.u_center = self.u_center_gt.copy()

        # Minor Processing Steps
        x = torch.unsqueeze(x, 0)  # [1, C, H, W]
        y = torch.unsqueeze(y, 0)  # [1, N_tracks, 2]

        u_center_gt = torch.from_numpy(self.u_center_gt)
        u_center_gt = torch.unsqueeze(u_center_gt, 0)  # [1, N_tracks, 2]

        return x, y, u_center_gt
'''


class TrackDataset(Dataset):
    """
    Dataloader for a collection of feature tracks. __getitem__ returns an instance of TrackData.
    """

    def __init__(
        self,
        track_tuples,
        get_frame_paths_fn,
        get_event_paths_fn,
        augment=False,
        patch_size=31,
        track_name="shitomasi_custom",
        representation="time_surfaces_v2_5",
    ):
        super(TrackDataset, self).__init__()
        self.track_tuples = track_tuples
        self.get_frame_paths_fn = get_frame_paths_fn
        self.get_event_paths_fn = get_event_paths_fn
        self.patch_size = patch_size
        self.track_name = track_name
        self.representation = representation
        self.augment = augment
        print(f"Initialized recurrent dataset with {len(self.track_tuples)} tracks.")

    def __len__(self):
        return len(self.track_tuples)

    '''def __getitem__(self, idx_track):
        track_tuple = self.track_tuples[idx_track]
        data_config = TrackDataConfig(
            self.get_frame_paths_fn(track_tuple[0]),
            self.get_event_paths_fn(track_tuple[0], self.representation),
            self.patch_size,
            self.representation,
            self.track_name,
            self.augment,
        )
        return TrackData(track_tuple, data_config)'''

    def __getitem__(self, idx_sample):
        track_path = self.track_tuples[idx_sample]

        data_config = TrackDataConfig(
            self.get_frame_paths_fn(track_path),
            self.get_event_paths_fn(track_path, self.representation),
            self.patch_size,
            self.representation,
            self.track_name,
            self.augment,
        )

        return TrackData(track_path, data_config)

class MFDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        extra_dir,
        dt=0.0100,
        batch_size=16,
        num_workers=4,
        patch_size=31,
        augment=False,
        n_train=20000,
        n_val=2000,
        track_name="shitomasi_custom",
        representation="time_surfaces_v2_1",
        mixed_dt=False,
        **kwargs,
    ):
        super(MFDataModule, self).__init__()

        random.seed(1234)

        self.num_workers = num_workers
        self.n_train = n_train
        self.n_val = n_val
        self._has_prepared_data = True

        self.data_dir = Path(data_dir)
        self.extra_dir = Path(extra_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.mixed_dt = mixed_dt
        self.dt = dt
        self.representation = representation
        self.patch_size = patch_size
        self.track_name = track_name

        self.dataset_train, self.dataset_val = None, None

        self.split_track_tuples = {}
        self.split_max_samples = {"train": n_train, "train_fune_2+3": n_train, "test": n_val}
        '''for split_name in ["train", "test"]:
            cache_path = (
                self.extra_dir / split_name / ".cache" / f"{track_name}.paths.pkl"
            )
            if cache_path.exists():
                with open(str(cache_path), "rb") as cache_f:
                    track_tuples = pickle.load(cache_f)
            else:
                track_tuples = retrieve_track_tuples(
                    self.extra_dir / split_name, track_name
                )
                with open(str(cache_path), "wb") as cache_f:
                    pickle.dump(track_tuples, cache_f)

            # Shuffle and trim
            n_tracks = len(track_tuples)
            track_tuples_array = np.asarray(track_tuples)
            track_tuples_array = track_tuples_array[: (n_tracks // 64) * 64, :]
            track_tuples_array = track_tuples_array.reshape([(n_tracks // 64), 64, 2])
            rand_perm = np.random.permutation((n_tracks // 64))
            track_tuples_array = track_tuples_array[rand_perm, :, :].reshape(
                (n_tracks // 64) * 64, 2
            )
            track_tuples_array[:, 1] = track_tuples_array[:, 1].astype(int)
            track_tuples = []
            for i in range(track_tuples_array.shape[0]):
                track_tuples.append(
                    [track_tuples_array[i, 0], int(track_tuples_array[i, 1])]
                )

            if self.split_max_samples[split_name] < len(track_tuples):
                track_tuples = track_tuples[: self.split_max_samples[split_name]]
            self.split_track_tuples[split_name] = track_tuples'''

        for split_name in ["train", "train_fune_2+3", "test"]:
            cache_path = (
                    self.extra_dir / split_name / ".cache" / f"{track_name}.paths.pkl"
            )

            if cache_path.exists():
                with open(str(cache_path), "rb") as cache_f:
                    track_tuples = pickle.load(cache_f)
            else:
                track_tuples = retrieve_track_tuples(
                    self.extra_dir / split_name, track_name
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(cache_path), "wb") as cache_f:
                    pickle.dump(track_tuples, cache_f)

            # 只保留 gt_path
            gt_paths = [t[0] for t in track_tuples]

            # 去重（因为同一个 gt.txt 会出现很多次）
            gt_paths = list(sorted(set(gt_paths)))

            # 打乱（按样本打乱，而不是按质心分组）
            np.random.shuffle(gt_paths)

            # 限制样本数量
            if self.split_max_samples[split_name] < len(gt_paths):
                gt_paths = gt_paths[: self.split_max_samples[split_name]]

            # 保存
            self.split_track_tuples[split_name] = gt_paths


    @staticmethod
    def get_frame_paths(track_path):
        images_dir = Path(
            os.path.split(track_path)[0]
            .replace("_extra", "")
            .replace("tracks", "images")
        )
        '''return sorted(
            [
                frame_p
                for frame_p in glob(str(images_dir / "*.png"))
                if 400000
                <= int(os.path.split(frame_p)[1].replace(".png", ""))
                <= 900000
            ]
        )'''
        frame_paths = []

        for frame_p in glob(str(images_dir / "*.png")):

            filename = os.path.split(frame_p)[1]

            try:

                # 支持 frame_00000076.png
                frame_id = int(
                    filename.replace(".png", "").replace("frame_", "")
                )

                # 如果你不需要限制范围，可以删除下面两行
                if 0 <= frame_id <= 9000000:
                    frame_paths.append(frame_p)

            except Exception as e:

                print("Invalid frame filename:", filename)
                continue
        return sorted(frame_paths)

    @staticmethod
    def get_event_paths(track_path, rep, dt):
        event_files = sorted(
            glob(
                str(
                    Path(os.path.split(track_path)[0].replace("tracks", "events"))
                    / f"{random.choice([0.0100, 0.0200]):.4f}"
                    / rep
                    / "*.h5"
                )
            )
        )
        return [
            event_p
            for event_p in event_files
            if 0000000 <= int(os.path.split(event_p)[1].replace(".h5", "")) <= 9000000
        ]

    @staticmethod
    def get_event_paths_mixed_dt(track_path, rep, dt):
        event_files = sorted(
            glob(
                str(
                    Path(os.path.split(track_path)[0].replace("tracks", "events"))
                    / f"{dt:.4f}"
                    / rep
                    / "*.h5"
                )
            )
        )
        return [
            event_p
            for event_p in event_files
            if 0000000 <= int(os.path.split(event_p)[1].replace(".h5", "")) <= 9000000
        ]

    def setup(self, stage=None):
        # Create train and val splits
        self.dataset_train = TrackDataset(
            self.split_track_tuples["train_fune_2+3"],
            MFDataModule.get_frame_paths,
            partial(MFDataModule.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(MFDataModule.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name=self.track_name,
            representation=self.representation,
            augment=self.augment,
        )
        self.dataset_val = TrackDataset(
            self.split_track_tuples["test"],
            MFDataModule.get_frame_paths,
            partial(MFDataModule.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(MFDataModule.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name="shitomasi_custom",
            representation=self.representation,
            augment=False,
        )

    def train_dataloader(self):
        subseq_sampler = SubSequenceRandomSampler(
            list(range(self.dataset_train.__len__()))
        )

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
            sampler=subseq_sampler,
        )

    def val_dataloader(self):
        subseq_sampler = SubSequenceRandomSampler(
            list(range(self.dataset_val.__len__()))
        )

        return DataLoader(
            self.dataset_val,
            batch_size=1,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=recurrent_collate,
            pin_memory=True,
            sampler=subseq_sampler,
        )


@dataclass
class CornerConfig:
    maxCorners: int
    qualityLevel: float
    minDistance: int
    k: float
    useHarrisDetector: bool
    blockSize: int


# Data Classes for Inference
class EvalDatasetType(Enum):
    EC = 0
    EDS = 1


class SequenceDataset(ABC):
    """
    Data class without ground-truth labels
    """

    def __init__(self):
        self.u_centers, self.u_centers_init = None, None
        self.heatmap, self.heatmap_init = None, None
        self.n_tracks = None
        self.event_first, self.frame_first = None, None
        self.t_now, self.t_init = None, None
        self.n_events, self.n_frames = None, None
        self.patch_size = None
        self.has_poses = False
        self.device = "cpu"
        self.x_ref = torch.zeros(1)
        self.height = 260#260 256 260 180 480
        self.width = 346#344 256 346 240 640

    def initialize(self, max_keypoints=30):
        self.initialize_keypoints(max_keypoints)
        self.initialize_reference_patches()

    def override_keypoints(self, keypoints):
        self.u_centers = keypoints
        self.u_centers = torch.from_numpy(self.u_centers.astype(np.float32))
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        if self.n_tracks == 0:
            raise ValueError("There are no corners in the initial frame")

        self.initialize_reference_patches()

    def override_heatmap(self, keypoints, height, width):

        keypoints = torch.from_numpy(keypoints.astype(np.float32))
        N, _ = keypoints.shape
        self.height = height
        self.width = width
        H, W = self.height, self.width

        heatmap = torch.zeros(1, H, W, device=self.device)

        # ===== 生成 7×7 Gaussian 核 =====
        kernel_size = 7
        sigma = 1.5
        half_k = kernel_size // 2

        coords = torch.arange(kernel_size, device=self.device) - half_k
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')

        gaussian_kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
        # (7,7)

        # ===== 获取目标中心坐标 =====
        x = keypoints[:, 0].round().long()  # [B,N]
        y = keypoints[:, 1].round().long()  # [B,N]

        # ===== 构造局部网格 =====
        grid_y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) + yy  # [B,N,7,7]
        grid_x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) + xx  # [B,N,7,7]

        # ===== 边界 mask =====
        valid = (
                (grid_x >= 0) & (grid_x < W) &
                (grid_y >= 0) & (grid_y < H)
        )

        # ===== batch index =====
        batch_idx = torch.arange(1, device=self.device).view(1, 1, 1, 1).expand(1, N, 7, 7)

        # ===== kernel expand =====
        kernel_expand = gaussian_kernel.view(1, 1, 7, 7).expand(1, N, 7, 7)

        # ===== 写入 heatmap =====
        heatmap[
            batch_idx[valid],
            grid_y[valid],
            grid_x[valid]
        ] = kernel_expand[valid]
        heatmap = heatmap.unsqueeze(1)

        self.heatmap = heatmap
        self.heatmap_init = heatmap.clone()


    '''def override_heatmap(self, keypoints):
        heatmap = np.zeros((1, self.height, self.width), dtype=np.float32)
        heatmap[0, keypoints[:, 1], keypoints[:, 0]] = 1.0
        self.heatmap = heatmap
        self.heatmap = torch.from_numpy(self.heatmap.astype(np.float32))
        self.heatmap_init = self.heatmap.clone()'''


    def initialize_keypoints(self, max_keypoints):
        self.u_centers = cv2.goodFeaturesToTrack(
            self.frame_first,
            max_keypoints,
            qualityLevel=self.corner_config.qualityLevel,
            minDistance=self.corner_config.minDistance,
            k=self.corner_config.k,
            useHarrisDetector=self.corner_config.useHarrisDetector,
            blockSize=self.corner_config.blockSize,
        ).reshape((-1, 2))
        self.u_centers = torch.from_numpy(self.u_centers.astype(np.float32))
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        if self.n_tracks == 0:
            raise ValueError("There are no corners in the initial frame")

    def move_centers(self):
        self.u_centers = self.u_centers.to(self.device)
        self.u_centers_init = self.u_centers_init.to(self.device)
        self.x_ref = self.x_ref.to(self.device)

    def move_heatmap(self):
        self.heatmap = self.heatmap.to(self.device)
        self.heatmap_init = self.heatmap_init.to(self.device)

    def accumulate_y_hat(self, y_hat):
        if y_hat.device != self.device:
            self.device = y_hat.device
            self.move_centers()

        self.u_centers += y_hat.detach()

    def update_heatmap(self, heatmap):
        if heatmap.device != self.device:
            self.device = heatmap.device
            self.move_heatmap()

        self.heatmap = heatmap.detach()

    def frames(self):
        """
        :return: generator over frames
        """
        pass

    def events(self):
        """
        :return: generator over event representations
        """
        pass

    #
    # def get_track_data(self):
    #     track_data = []
    #     for i in range(self.u_centers.shape[0]):
    #         track_data.append([i, self.t_now, self.u_centers[i, 0], self.u_centers[i, 1]])
    #     return track_data

    def get_patches(self, f):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param f:
        :return:
        """
        if f.device != self.device:
            self.device = f.device
            self.move_centers()

        # 0.5 offset is needed due to coordinate system of grid_sample
        return extract_glimpse(
            f.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size, self.patch_size),
            self.u_centers.detach() + 0.5,
            mode="nearest",
        )

    def get_patches_new(self, arr_h5, padding=4):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param arr_h5: h5 file for the input event representation
        :return: (n_tracks, c, p, p) tensor
        """
        # Extract expanded patches from the h5 files
        u_centers_np = self.u_centers.detach().cpu().numpy()
        x_patches = []
        for i in range(self.n_tracks):
            u_center = u_centers_np[i, :]
            u_center_rounded = np.rint(u_center)
            u_center_offset = (
                u_center - u_center_rounded + ((self.patch_size + padding) // 2.0)
            )
            x_patch_expanded = get_patch_voxel(
                arr_h5, u_center_rounded.reshape((-1,)), self.patch_size + padding
            ).unsqueeze(0)
            x_patch = extract_glimpse(
                x_patch_expanded,
                (self.patch_size, self.patch_size),
                torch.from_numpy(u_center_offset).view((1, 2)) + 0.5,
                mode="nearest",
            )
            x_patches.append(x_patch)
        return torch.cat(x_patches, dim=0)

    @abstractmethod
    def initialize_reference_patches(self):
        pass

    def get_next(self):
        """
        Abstract method for getting input patches and epipolar lines
        :return: input patches (n_corners, C, patch_size, patch_size) and epipolar lines (n_corners, 3)
        """
        pass

    def get_frame(self, image_idx):
        pass


class Multiflow(SequenceDataset):
    def __init__(
        self,
        sequence_idx,
        data_dir,
        extra_dir,
        patch_size,
        representation,
        track_name,
        include_prev,
        **kwargs,
    ):
        super().__init__()
        self.sequence_idx = sequence_idx
        self.data_dir = Path(data_dir)
        self.extra_dir = Path(extra_dir)
        self.patch_size = patch_size
        self.representation = representation
        self.track_name = track_name
        self.include_prev = include_prev

        self.sequence_dir = list((self.extra_dir / "test").iterdir())[self.sequence_idx]
        self.data_sequence_dir = Path(str(self.sequence_dir).replace("_extra", ""))
        self.n_images = 11

        # Get first and last images
        self.t_now, self.t_init = 0.4, 0.4
        self.time_idx = 0
        self.dt = 0.01
        self.img_first = cv2.imread(
            str(self.data_sequence_dir / "images" / f"0400000.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        self.img_last = cv2.imread(
            str(self.data_sequence_dir / "images" / f"0900000.png"),
            cv2.IMREAD_GRAYSCALE,
        )

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return 50

    def initialize_reference_patches(self):
        # TODO: accommodate grayscale
        # Store reference patches
        self.x_ref = []
        ref_input_path = (
            self.sequence_dir / "events" / f"{self.representation}" / "0400000.h5"
        )
        if "time_surface" in self.representation:
            self.channels_in_per_patch = 1
            ref_input = h5py.File(str(ref_input_path), "r")["time_surface"]
            self.h5_key = "time_surface"

            if self.include_prev:
                self.cropping_fn = get_patch_pairs
            else:
                self.cropping_fn = get_patch

            for i in range(self.n_tracks):
                x = get_patch(ref_input, self.u_centers[i, :], self.patch_size)
                self.x_ref.append(x.unsqueeze(0))
            self.x_ref = torch.cat(self.x_ref, dim=0)

        elif "voxel" in self.representation:
            self.channels_in_per_patch = int(self.representation[-1])
            ref_input = h5py.File(str(ref_input_path), "r")["voxel_grid"]
            self.h5_key = "voxel_grid"

            if self.include_prev:
                self.cropping_fn = get_patch_voxel_pairs
            else:
                self.cropping_fn = get_patch_voxel

            for i in range(self.n_tracks):
                x = get_patch_voxel(ref_input, self.u_centers[i, :], self.patch_size)
                self.x_ref.append(x.unsqueeze(0))
            self.x_ref = torch.cat(self.x_ref, dim=0)

    def get_next(self):
        # Round patch locations
        self.u_centers = np.rint(self.u_centers)

        # Get patch inputs
        if self.include_prev:
            input_0_path = (
                self.sequence_dir
                / "events"
                / f"{self.representation}"
                / f"0{400000 + self.time_idx * 10000:.0f}.h5"
            )
            input_1_path = (
                self.sequence_dir
                / "events"
                / f"{self.representation}"
                / f"0{410000 + self.time_idx * 10000:.0f}.h5"
            )
            input_0 = h5py.File(input_0_path, "r")[self.h5_key]
            input_1 = h5py.File(input_1_path, "r")[self.h5_key]

            x = []
            for i in range(self.n_tracks):
                x.append(
                    self.cropping_fn(
                        input_1, input_0, self.u_centers[i, :], self.patch_size
                    ).unsqueeze(0)
                )
        else:
            input_1_path = (
                self.sequence_dir
                / "events"
                / f"{self.representation}"
                / f"0{410000 + self.time_idx * 10000:.0f}.h5"
            )
            input_1 = h5py.File(input_1_path, "r")[self.h5_key]

            x = []
            for i in range(self.n_tracks):
                x.append(
                    self.cropping_fn(
                        input_1, self.u_centers[i, :], self.patch_size
                    ).unsqueeze(0)
                )
        x = torch.cat(x, dim=0)
        x = torch.cat([x, self.x_ref], dim=1)

        # Update time info
        self.time_idx += 1
        self.t_now = 0.4 + self.dt * self.time_idx

        return x

    def get_frame(self, image_idx):
        img_path = (
            self.data_sequence_dir / "images" / f"0{400000 + image_idx * 50000:.0f}.png"
        )
        t = 0.4 + 0.05 * image_idx
        return t, cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)


class EDSSubseq(SequenceDataset):
    # ToDo: Add to config file
    pose_r = 3
    pose_mode = False

    def __init__(
        self,
        root_dir,
        sequence_name,
        n_frames,
        patch_size,
        representation,
        dt,
        corner_config,
        include_prev=False,
        fused=False,
        grayscale_ref=True,
        use_colmap_poses=True,
        global_mode=False,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation
        self.include_prev = include_prev
        self.dt, self.dt_us = dt, dt * 1e6
        self.grayscale_ref = grayscale_ref
        self.use_colmap_poses = use_colmap_poses
        self.global_mode = global_mode
        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config

        # Determine number of frames
        self.frame_dir = self.root_dir / sequence_name / "images_corrected"
        max_frames = len(list(self.frame_dir.iterdir())) - 1
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames

        # Check that event representations have been generated for this dt
        if not self.pose_mode:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / "events"
                / f"{dt:.4f}"
                / f"{self.representation}"
            )
        else:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / "events"
                / f"pose_{self.pose_r:.0f}"
                / f"{self.representation}"
            )
        if not self.dir_representation.exists():
            print(
                f"{self.representation} has not yet been generated for a dt of {self.dt}"
            )
            exit()

        # Read timestamps
        self.frame_ts_arr = np.genfromtxt(
            str(self.sequence_dir / "images_timestamps.txt")
        )

        # Read poses and camera matrix
        if self.use_colmap_poses:
            pose_data_path = self.sequence_dir / "colmap" / "stamped_groundtruth.txt"
        else:
            pose_data_path = self.sequence_dir / "stamped_groundtruth.txt"
        self.pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        with open(str(self.root_dir / "calib.yaml"), "r") as fh:
            intrinsics = yaml.load(fh, Loader=yaml.SafeLoader)["cam0"]["intrinsics"]
            self.camera_matrix = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ]
            )
            self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        # Tensor Manipulation
        self.channels_in_per_patch = int(self.representation[-1])
        if "v2" in self.representation:
            self.channels_in_per_patch *= 2

        if self.include_prev:
            self.cropping_fn = get_patch_voxel_pairs
        else:
            self.cropping_fn = get_patch_voxel

        # Timing and Indices
        self.current_idx = 0
        self.t_init = self.frame_ts_arr[0] * 1e-6
        self.t_end = self.frame_ts_arr[-1] * 1e-6
        self.t_now = self.t_init

        # Pose interpolator for epipolar geometry
        self.pose_interpolator = PoseInterpolator(self.pose_data)
        self.T_last_W = self.pose_interpolator.interpolate(self.t_now)

        # Get counts
        self.n_events = int(np.ceil((self.t_end - self.t_init) / self.dt))

        # Get first imgs
        self.frame_first = cv2.imread(
            str(self.frame_dir / ("frame_" + f"{0}".zfill(10) + ".png")),
            cv2.IMREAD_GRAYSCALE,
        )
        # self.event_first = array_to_tensor(read_input(str(self.dir_representation / '0000000.h5'), self.representation))
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return

    def reset(self):
        self.t_now = self.t_init
        self.current_idx = 0
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        if "grayscale" in self.representation or self.grayscale_ref:
            ref_input = (
                torch.from_numpy(self.frame_first.astype(np.float32) / 255)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            ref_input = self.event_first.unsqueeze(0)
        self.x_ref = self.get_patches(ref_input)

        # for i in range(self.n_tracks):
        #     x = get_patch_voxel(ref_input, self.u_centers[i, :], self.patch_size)
        #     self.x_ref.append(x[:self.channels_in_per_patch, :, :].unsqueeze(0))
        # self.x_ref = torch.cat(self.x_ref, dim=0)

    def globals(self):
        for i in range(1, self.n_events):
            self.t_now += self.dt
            x = array_to_tensor(
                read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            )

            yield self.t_now, x.unsqueeze(0)

    def get_current_event(self):
        # Get patch inputs and set current time
        if not self.pose_mode:
            self.t_now += self.dt
            input_1 = read_input(
                self.dir_representation
                / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
                self.representation,
            )
        else:
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[self.current_idx])[
                        1
                    ].replace(".h5", "")
                )
                * 1e-6
            )
            input_1 = read_input(
                self.event_representation_paths[self.current_idx], self.representation
            )

            if self.current_idx > 0 and self.current_idx % self.pose_r == 0:
                ref_input = cv2.imread(
                    str(
                        self.frame_dir
                        / (
                            "frame_"
                            + f"{self.current_idx // self.pose_r}".zfill(10)
                            + ".png"
                        )
                    ),
                    cv2.IMREAD_GRAYSCALE,
                )
                ref_input = (
                    torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.x_ref.device)
                )
                self.x_ref = extract_glimpse(
                    ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                    (self.patch_size, self.patch_size),
                    self.u_centers.detach() + 0.5,
                    mode="bilinear",
                )

        input_1 = np.array(input_1)
        input_1 = np.transpose(input_1, (2, 0, 1))
        input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

        x = extract_glimpse(
            input_1.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size, self.patch_size),
            self.u_centers.detach() + 0.5,
        )
        x = torch.cat([x, self.x_ref], dim=1)

        return self.t_now, x

    def full_representation_events(self):
        self.event_representation_paths = sorted(
            glob(str(self.dir_representation / "*.h5")),
            key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
        )
        self.n_events = len(self.event_representation_paths)
        current_idx = 0

        for i in range(current_idx, self.n_events):
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[current_idx])[
                        1
                    ].replace(".h5", "")
                )
                * 1e-6
            )
            events_repr = read_input(
                self.event_representation_paths[current_idx], self.representation
            )

            events_repr = np.array(events_repr)
            events_repr = np.transpose(events_repr, (2, 0, 1))

            current_idx += 1

            yield self.t_now, events_repr

    def events(self):
        if self.pose_mode:
            self.event_representation_paths = sorted(
                glob(str(self.dir_representation / "*.h5")),
                key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
            )
            self.n_events = len(self.event_representation_paths)
            self.current_idx = 0
        else:
            self.current_idx = 1

        for self.current_idx in range(self.current_idx, self.n_events):
            # Get patch inputs and set current time
            if not self.pose_mode:
                self.t_now += self.dt
                input_1 = read_input(
                    self.dir_representation
                    / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            else:
                self.t_now = (
                    float(
                        os.path.split(
                            self.event_representation_paths[self.current_idx]
                        )[1].replace(".h5", "")
                    )
                    * 1e-6
                )
                input_1 = read_input(
                    self.event_representation_paths[self.current_idx],
                    self.representation,
                )

                if self.current_idx > 0 and self.current_idx % self.pose_r == 0:
                    ref_input = cv2.imread(
                        str(
                            self.frame_dir
                            / (
                                "frame_"
                                + f"{self.current_idx // self.pose_r}".zfill(10)
                                + ".png"
                            )
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    ref_input = (
                        torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.x_ref.device)
                    )
                    self.x_ref = extract_glimpse(
                        ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                        (self.patch_size, self.patch_size),
                        self.u_centers.detach() + 0.5,
                        mode="bilinear",
                    )

            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            x = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size, self.patch_size),
                self.u_centers.detach() + 0.5,
            )
            x = torch.cat([x, self.x_ref], dim=1)

            yield self.t_now, x

    def frames(self):
        for i in range(1, self.n_frames):
            # Update time info
            self.t_now = self.frame_ts_arr[i] * 1e-6

            frame = cv2.imread(
                str(
                    self.sequence_dir
                    / "images_corrected"
                    / ("frame_" + f"{i}".zfill(10) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            yield self.t_now, frame

    def get_next(self):
        """Strictly for pose supervision"""

        # Update time info
        self.t_now += self.dt

        self.current_idx += 1
        # DEBUG: Use grayscale frame timestamps
        # self.t_now = self.frame_ts_arr[self.current_idx]*1e-6

        # Get patch inputs
        input_1 = read_input(
            self.dir_representation
            / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
            self.representation,
        )
        x = array_to_tensor(input_1)
        x_patches = self.get_patches(x)

        # Get epipolar lines
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
        T_now_last = T_now_W @ np.linalg.inv(self.T_last_W)
        T_last_now = np.linalg.inv(T_now_last)
        self.T_last_W = T_now_W
        F = (
            self.camera_matrix_inv.T
            @ skew(T_last_now[:3, 3])
            @ T_last_now[:3, :3]
            @ self.camera_matrix_inv
        )
        u_centers = self.u_centers.detach().cpu().numpy()
        u_centers_homo = np.concatenate(
            [u_centers, np.ones((u_centers.shape[0], 1))], axis=1
        )
        l_epi = torch.from_numpy(u_centers_homo @ F)

        return x_patches, l_epi


class ECSubseq(SequenceDataset):
    # ToDO: Add to config file
    pose_r = 4
    pose_mode = False

    def __init__(
        self,
        root_dir,
        sequence_name,
        n_frames,
        patch_size,
        representation,
        dt,
        corner_config,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation
        self.dt, self.dt_us = dt, dt * 1e6
        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config

        # Determine number of frames
        self.frame_dir = self.sequence_dir / "images_corrected"
        max_frames = len(list(self.frame_dir.iterdir()))
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames

        # Check that event representations have been generated for this dt
        if not self.pose_mode:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / "events"
                / f"{dt:.4f}"
                / f"{self.representation}"
            )
        else:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / "events"
                / f"pose_{self.pose_r:.0f}"
                / f"{self.representation}"
            )

        if not self.dir_representation.exists():
            print(
                f"{self.representation} has not yet been generated for a dt of {self.dt}"
            )
            exit()

        # Read timestamps
        self.frame_ts_arr = np.genfromtxt(str(self.sequence_dir / "images.txt"))

        # Read poses and camera matrix
        if (self.sequence_dir / "colmap").exists():
            pose_data_path = self.sequence_dir / "colmap" / "stamped_groundtruth.txt"
            self.pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        else:
            self.pose_data = np.genfromtxt(str(self.sequence_dir / "groundtruth.txt"))
        intrinsics = np.genfromtxt(str(self.sequence_dir / "calib.txt"))
        self.camera_matrix = np.array(
            [
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1],
            ]
        )

        # Tensor Manipulation
        self.channels_in_per_patch = int(self.representation[-1])
        if "v2" in self.representation:
            self.channels_in_per_patch *= 2
        self.cropping_fn = get_patch_voxel

        # Timing and Indices
        self.t_init = self.frame_ts_arr[0]
        self.t_end = self.frame_ts_arr[-1]
        self.t_now = self.t_init

        # Get counts
        self.n_events = int(np.ceil((self.t_end - self.t_init) / self.dt))

        # Get first imgs
        self.frame_first = cv2.imread(
            str(self.frame_dir / ("frame_" + f"{0}".zfill(8) + ".png")),
            cv2.IMREAD_GRAYSCALE,
        )
        # self.event_first = array_to_tensor(read_input(str(self.dir_representation / '0000000.h5'), self.representation))
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return

    def reset(self):
        self.t_now = self.t_init
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        ref_input = (
            torch.from_numpy(self.frame_first.astype(np.float32) / 255)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.x_ref = self.get_patches(ref_input)

    def globals(self):
        for i in range(1, self.n_events):
            self.t_now += self.dt
            x = array_to_tensor(
                read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            )

            yield self.t_now, x.unsqueeze(0)

    def events(self):
        if self.pose_mode:
            self.event_representation_paths = sorted(
                glob(str(self.dir_representation / "*.h5")),
                key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
            )
            self.n_events = len(self.event_representation_paths)
            i_start = 0
        else:
            i_start = 1

        for i in range(i_start, self.n_events):
            # Get patch inputs and set current time
            if not self.pose_mode:
                self.t_now += self.dt
                input_1 = read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            else:
                self.t_now = (
                    float(
                        os.path.split(self.event_representation_paths[i])[1].replace(
                            ".h5", ""
                        )
                    )
                    * 1e-6
                )
                input_1 = read_input(
                    self.event_representation_paths[i], self.representation
                )

                if i > 0 and i % self.pose_r == 0:
                    # print(self.frame_dir / ('frame_' + f'{i // self.pose_r}'.zfill(8) + '.png'))
                    ref_input = cv2.imread(
                        str(
                            self.frame_dir
                            / ("frame_" + f"{i//self.pose_r}".zfill(8) + ".png")
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    ref_input = (
                        torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.x_ref.device)
                    )
                    self.x_ref = extract_glimpse(
                        ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                        (self.patch_size, self.patch_size),
                        self.u_centers.detach() + 0.5,
                        mode="bilinear",
                    )

            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            x = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size, self.patch_size),
                self.u_centers.detach() + 0.5,
            )
            x = torch.cat([x, self.x_ref], dim=1)

            yield self.t_now, x

    def full_representation_events(self):
        self.event_representation_paths = sorted(
            glob(str(self.dir_representation / "*.h5")),
            key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
        )
        self.n_events = len(self.event_representation_paths)
        i_start = 0

        for i in range(i_start, self.n_events):
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[i])[1].replace(
                        ".h5", ""
                    )
                )
                * 1e-6
            )
            events_repr = read_input(
                self.event_representation_paths[i], self.representation
            )

            events_repr = np.array(events_repr)
            events_repr = np.transpose(events_repr, (2, 0, 1))

            yield self.t_now, events_repr

    def frames(self):
        for i in range(self.n_frames):
            # Update time info
            self.t_now = self.frame_ts_arr[i]

            frame = cv2.imread(
                str(
                    self.sequence_dir
                    / "images_corrected"
                    / ("frame_" + f"{i}".zfill(8) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            yield self.t_now, frame

    def get_next(self):
        pass

class ECSubseq_nogt(SequenceDataset):
    # ToDO: Add to config file
    pose_r = 4
    pose_mode = False

    def __init__(
        self,
        root_dir,
        sequence_name,
        n_frames,
        patch_size,
        representation,
        dt,
        corner_config,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation
        self.dt, self.dt_us = dt, dt * 1e6
        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config

        # Determine number of frames
        self.frame_dir = self.sequence_dir / "images"
        max_frames = len(list(self.frame_dir.iterdir()))
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames

        # Check that event representations have been generated for this dt
        if not self.pose_mode:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / "events"
                / f"{dt:.4f}"
                / f"{self.representation}"
            )
        else:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / "events"
                / f"pose_{self.pose_r:.0f}"
                / f"{self.representation}"
            )

        if not self.dir_representation.exists():
            print(
                f"{self.representation} has not yet been generated for a dt of {self.dt}"
            )
            exit()

        # Read timestamps
        self.frame_ts_arr = np.genfromtxt(str(self.sequence_dir / "images.txt"))

        # Tensor Manipulation
        self.channels_in_per_patch = int(self.representation[-1])
        if "v2" in self.representation:
            self.channels_in_per_patch *= 2
        self.cropping_fn = get_patch_voxel

        # Timing and Indices
        self.t_init = self.frame_ts_arr[0]
        self.t_end = self.frame_ts_arr[-1]
        self.t_now = self.t_init

        # Get counts
        self.n_events = int(np.ceil((self.t_end - self.t_init) / self.dt))

        # Get first image
        self.frame_first = cv2.imread(
            str(self.frame_dir / ("frame_" + f"{0}".zfill(8) + ".png")),
            cv2.IMREAD_GRAYSCALE,
        )
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return self.n_events

    def reset(self):
        self.t_now = self.t_init
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        ref_input = (
            torch.from_numpy(self.frame_first.astype(np.float32) / 255)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.x_ref = ref_input #self.get_patches(ref_input)

    def globals(self):
        for i in range(1, self.n_events):
            self.t_now += self.dt
            x = array_to_tensor(
                read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            )
            yield self.t_now, x.unsqueeze(0)

    def events(self):
        if self.pose_mode:
            self.event_representation_paths = sorted(
                glob(str(self.dir_representation / "*.h5")),
                key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
            )
            self.n_events = len(self.event_representation_paths)
            i_start = 0
        else:
            i_start = 1

        for i in range(i_start, self.n_events):
            # Get patch inputs and set current time
            if not self.pose_mode:
                self.t_now += self.dt
                input_1 = read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            else:
                self.t_now = (
                    float(
                        os.path.split(self.event_representation_paths[i])[1].replace(
                            ".h5", ""
                        )
                    )
                    * 1e-6
                )
                input_1 = read_input(
                    self.event_representation_paths[i], self.representation
                )

                if i > 0 and i % self.pose_r == 0:
                    ref_input = cv2.imread(
                        str(
                            self.frame_dir
                            / ("frame_" + f"{i//self.pose_r}".zfill(8) + ".png")
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    ref_input = (
                        torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.x_ref.device)
                    )

                    '''self.x_ref = extract_glimpse(
                        ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                        (self.patch_size, self.patch_size),
                        self.u_centers.detach() + 0.5,
                        mode="bilinear",
                    )'''
                    self.x_ref = self.x_ref[:,:,:self.height,:self.width]
            # u_centers 是不断更新的
            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            '''x = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size, self.patch_size),
                self.u_centers.detach() + 0.5,
            )'''
            x = torch.cat([input_1, self.x_ref], dim=1)
            x = x[:, :, :self.height, :self.width]

            yield self.t_now, x

    def full_representation_events(self):
        self.event_representation_paths = sorted(
            glob(str(self.dir_representation / "*.h5")),
            key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
        )
        self.n_events = len(self.event_representation_paths)
        i_start = 0

        for i in range(i_start, self.n_events):
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[i])[1].replace(
                        ".h5", ""
                    )
                )
                * 1e-6
            )
            events_repr = read_input(
                self.event_representation_paths[i], self.representation
            )

            events_repr = np.array(events_repr)
            events_repr = np.transpose(events_repr, (2, 0, 1))

            yield self.t_now, events_repr

    def frames(self):
        for i in range(self.n_frames):
            # Update time info
            self.t_now = self.frame_ts_arr[i]
            frame = cv2.imread(
                str(
                    self.sequence_dir
                    / "images"
                    / ("frame_" + f"{i}".zfill(8) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            frame = frame[...,:self.height,:self.width]
            yield self.t_now, frame

    def get_next(self):
        pass

class EDSSubseq_nogt(SequenceDataset):
    # ToDo: Add to config file
    pose_r = 3
    pose_mode = False

    def __init__(
        self,
        root_dir,
        sequence_name,
        n_frames,
        patch_size,
        representation,
        dt,
        corner_config,
        include_prev=False,
        fused=False,
        grayscale_ref=True,
        global_mode=False,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation
        self.include_prev = include_prev
        self.dt, self.dt_us = dt, dt * 1e6
        self.grayscale_ref = grayscale_ref
        self.global_mode = global_mode
        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config

        # Determine number of frames
        self.frame_dir = self.root_dir / sequence_name / "images"
        max_frames = len(list(self.frame_dir.iterdir())) - 1
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames

        # Check that event representations have been generated for this dt
        self.dir_representation = (
            self.root_dir
            / sequence_name
            / "events"
            / f"{dt:.4f}"
            / f"{self.representation}"
        )
        if not self.dir_representation.exists():
            print(f"{self.representation} has not yet been generated for a dt of {self.dt}")
            exit()

        # Read timestamps
        self.frame_ts_arr = np.genfromtxt(
            str(self.sequence_dir / "images_timestamps.txt")
        )

        # Read camera matrix
        with open(str(self.root_dir / "calib.yaml"), "r") as fh:
            intrinsics = yaml.load(fh, Loader=yaml.SafeLoader)["cam0"]["intrinsics"]
            self.camera_matrix = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ]
            )
            self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        # Tensor Manipulation
        self.channels_in_per_patch = int(self.representation[-1])
        if "v2" in self.representation:
            self.channels_in_per_patch *= 2

        if self.include_prev:
            self.cropping_fn = get_patch_voxel_pairs
        else:
            self.cropping_fn = get_patch_voxel

        # Timing and Indices
        self.current_idx = 0
        self.t_init = self.frame_ts_arr[0] * 1e-6
        self.t_end = self.frame_ts_arr[-1] * 1e-6
        self.t_now = self.t_init

        # Get counts
        self.n_events = int(np.ceil((self.t_end - self.t_init) / self.dt))

        # Get first img
        self.frame_first = cv2.imread(
            str(self.frame_dir / ("frame_" + f"{0}".zfill(10) + ".png")),
            cv2.IMREAD_GRAYSCALE,
        ) # EC: zfill(8) EDS:10
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return

    def reset(self):
        self.t_now = self.t_init
        self.current_idx = 0
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        if "grayscale" in self.representation or self.grayscale_ref:
            ref_input = (
                torch.from_numpy(self.frame_first.astype(np.float32) / 255)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            ref_input = self.event_first.unsqueeze(0)
        self.x_ref = ref_input

    def globals(self):
        for i in range(1, self.n_events):
            self.t_now += self.dt
            x = array_to_tensor(
                read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            )
            yield self.t_now, x.unsqueeze(0)

    def get_current_event(self):
        self.t_now += self.dt
        input_1 = read_input(
            self.dir_representation
            / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
            self.representation,
        )

        input_1 = np.array(input_1)
        input_1 = np.transpose(input_1, (2, 0, 1))
        input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

        x = torch.cat([input_1, self.x_ref], dim=1)

        return self.t_now, x

    def full_representation_events(self):
        self.event_representation_paths = sorted(
            glob(str(self.dir_representation / "*.h5")),
            key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
        )
        self.n_events = len(self.event_representation_paths)
        current_idx = 0

        for i in range(current_idx, self.n_events):
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[current_idx])[1].replace(".h5", "")
                )
                * 1e-6
            )
            events_repr = read_input(
                self.event_representation_paths[current_idx], self.representation
            )

            events_repr = np.array(events_repr)
            events_repr = np.transpose(events_repr, (2, 0, 1))

            current_idx += 1
            yield self.t_now, events_repr

    def events(self):
        self.current_idx = 1

        for self.current_idx in range(self.current_idx, self.n_events):
            self.t_now += self.dt
            input_1 = read_input(
                self.dir_representation
                / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
                self.representation,
            )

            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            x = torch.cat([input_1, self.x_ref], dim=1)

            yield self.t_now, x

    def frames(self):
        for i in range(1, self.n_frames):
            self.t_now = self.frame_ts_arr[i] * 1e-6

            frame = cv2.imread(
                str(
                    self.sequence_dir
                    / "images"
                    / ("frame_" + f"{i}".zfill(10) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            yield self.t_now, frame

    def get_next(self):
        pass


# Dataclasses for Pose Training
class PoseDataset(Dataset):
    """
    Dataset encapsulating multiple sequences/subsequences.
    The events for each sequence/subsequence are loaded into memory and used to instantiate segments.
    Segments start at an image index, so that keypoints can be extracted.
    """

    def __init__(
        self,
        dataset_type,
        root_dir,
        n_event_representations_per_frame,
        max_segments,
        representation,
        patch_size,
        n_frames_skip,
    ):
        if dataset_type == EvalDatasetType.EDS:
            self.segment_dataset_class = EDSPoseSegmentDataset
        elif dataset_type == EvalDatasetType.EC:
            self.segment_dataset_class = ECPoseSegmentDataset
        else:
            raise NotImplementedError

        self.root_dir = Path(root_dir)
        self.idx2sequence = []
        self.n_event_representations_per_frame = n_event_representations_per_frame
        self.n_frames_skip = n_frames_skip
        self.max_segments = max_segments
        self.patch_size = patch_size
        self.representation = representation

        for sequence_dir in self.root_dir.iterdir():
            # ToDo: Change back for EDS Pose training
            if "." in str(sequence_dir) or str(sequence_dir.stem) in [
                "rocket_earth_dark",
                "peanuts_dark",
                "ziggy_and_fuzz",
            ]:
                # if '.' in str(sequence_dir):
                continue
            sequence_name = sequence_dir.stem

            # Don't consider frames after the last pose timestamp
            pose_interpolator = self.segment_dataset_class.get_pose_interpolator(
                sequence_dir
            )
            pose_ts_min = np.min(pose_interpolator.pose_data[:, 0])
            pose_ts_max = np.max(pose_interpolator.pose_data[:, 0]) - 4 * 0.010

            frame_ts_arr = self.segment_dataset_class.get_frame_timestamps(sequence_dir)
            inrange_mask = np.logical_and(
                frame_ts_arr > pose_ts_min, frame_ts_arr < pose_ts_max
            )
            frame_indices = np.nonzero(inrange_mask)[0]
            frame_paths = self.segment_dataset_class.get_frame_paths(sequence_dir)

            cached_mappings_path = sequence_dir / "valid_mappings.pkl"
            if cached_mappings_path.exists():
                with open(cached_mappings_path, "rb") as cached_mappings_f:
                    new_mappings = pickle.load(cached_mappings_f)
            else:
                frame_indices_skipped = list(
                    range(
                        np.min(frame_indices), np.max(frame_indices) - 4, n_frames_skip
                    )
                )
                new_mappings = []
                for i in tqdm(
                    frame_indices_skipped, desc="Checking corners in starting frames..."
                ):
                    img = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
                    kp = cv2.goodFeaturesToTrack(
                        img, 2, 0.3, 15, blockSize=11, useHarrisDetector=False, k=0.15
                    )
                    if not isinstance(kp, type(None)):
                        new_mappings.append((sequence_name, i))

                with open(cached_mappings_path, "wb") as cached_mappings_f:
                    pickle.dump(new_mappings, cached_mappings_f)

            random.shuffle(new_mappings)

            self.idx2sequence += new_mappings

        random.shuffle(self.idx2sequence)
        if len(self.idx2sequence) > self.max_segments:
            self.idx2sequence = self.idx2sequence[: self.max_segments]

    def __len__(self):
        return len(list(self.idx2sequence))

    def __getitem__(self, idx_segment):
        sequence, idx_start = self.idx2sequence[idx_segment]
        return self.segment_dataset_class(
            self.root_dir / sequence,
            idx_start,
            self.patch_size,
            self.representation,
            self.n_event_representations_per_frame,
        )


class PoseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_type,
        root_dir,
        n_event_representations_per_frame,
        n_train,
        n_val,
        batch_size,
        num_workers,
        patch_size,
        representation,
        n_frames_skip,
        **kwargs,
    ):
        super(PoseDataModule, self).__init__()
        assert dataset_type in [
            "EDS",
            "EC",
            "FPV",
        ], "Dataset type must be one of EDS, EC, or FPV"
        if dataset_type == "EDS":
            self.dataset_type = EvalDatasetType.EDS
        elif dataset_type == "EC":
            self.dataset_type = EvalDatasetType.EC
        else:
            raise NotImplementedError("Dataset type not supported for pose training")

        self.root_dir = root_dir
        self.n_event_representations_per_frame = n_event_representations_per_frame
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.representation = representation
        self.n_frames_skip = n_frames_skip
        self.n_train = n_train
        self.n_val = n_val
        self.dataset_train, self.dataset_val = None, None

    def setup(self, stage=None):
        self.dataset_train = PoseDataset(
            self.dataset_type,
            self.root_dir,
            self.n_event_representations_per_frame,
            self.n_train,
            self.representation,
            self.patch_size,
            self.n_frames_skip,
        )
        # Change this later
        self.dataset_val = PoseDataset(
            self.dataset_type,
            self.root_dir,
            self.n_event_representations_per_frame,
            self.n_val,
            self.representation,
            self.patch_size,
            self.n_frames_skip,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
        )


class SequenceDatasetV2(ABC):
    """Abstract class for real data.
    Defines loaders for timestamps, pose data, and input paths.
    Defines generators for frames, events, and even_events"""

    def __init__(self):
        self.idx_start = 0
        pass

    # TODO: fix these
    def initialize_pose_and_calib(self):
        # Loading
        (
            self.camera_matrix,
            self.camera_matrix_inv,
            self.distortion_coeffs,
        ) = self.get_calibration(self.sequence_dir)
        self.frame_ts_arr = self.get_frame_timestamps(self.sequence_dir)
        self.pose_interpolator = self.get_pose_interpolator(self.sequence_dir)

    def initialize_time_and_input_paths(self, pose_mode, dt_or_r):
        if pose_mode:
            self.event_representation_paths = self.get_even_event_paths(
                self.sequence_dir, self.representation, dt_or_r
            )
        else:
            self.event_representation_paths = self.get_event_paths(
                self.sequence_dir, self.representation, dt_or_r
            )
        self.frame_paths = self.get_frame_paths(self.sequence_dir)

        # Initial time and pose
        self.t_init = self.frame_ts_arr[self.idx_start]
        self.t_now = self.t_init
        self.T_first_W = self.pose_interpolator.interpolate(self.t_now)

    def initialize_centers_and_ref_patches(self, corner_config):
        # Initial keypoints and patches
        self.frame_first = cv2.imread(
            self.frame_paths[self.idx_start], cv2.IMREAD_GRAYSCALE
        )
        self.u_centers = self.get_keypoints(self.frame_first, corner_config)
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        # Store reference patches
        ref_input = (
            torch.from_numpy(self.frame_first.astype(np.float32) / 255)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.x_ref = self.get_patches(ref_input)

    def accumulate_y_hat(self, y_hat):
        if y_hat.device != self.device:
            self.device = y_hat.device
            self.move_centers()

        self.u_centers += y_hat.detach()

    def configure_patches_iterator(self, corner_config, pose_mode, dt_or_r):
        # Initialize indices and timestamps
        self.initialize_time_and_input_paths(pose_mode, dt_or_r)
        self.initialize_centers_and_ref_patches(corner_config)
        self.n_events = len(self.event_representation_paths)
        self.n_frames = len(self.frame_paths)
        self.pose_mode = pose_mode
        self.dt_or_r = dt_or_r

    def get_patches_iterator(self):
        # Initialize reference patches
        for i in range(1, self.n_events):
            # Update time info
            if not self.pose_mode:
                self.t_now += self.dt_or_r
            else:
                self.t_now = (
                    float(
                        os.path.split(self.event_representation_paths[i])[1].replace(
                            ".h5", ""
                        )
                    )
                    * 1e-6
                )

            # Get patch inputs
            input_1 = read_input(
                self.event_representation_paths[i], self.representation
            )

            x = self.get_patches_new(input_1)
            if x.device != self.x_ref.device:
                self.x_ref = self.x_ref.to(x.device)

            x = torch.cat([x, self.x_ref], dim=1)

            yield self.t_now, x

    @staticmethod
    @abstractmethod
    def get_frame_timestamps(sequence_dir):
        """
        :return: (-1,) array of timestamps in seconds
        """
        pass

    @staticmethod
    @abstractmethod
    def get_pose_interpolator(sequence_dir):
        """
        :return: PoseInterpolator object instantiated from sequence's pose data
        """
        pass

    @staticmethod
    @abstractmethod
    def get_calibration(sequence_dir):
        """
        :return: dict with keys 'camera_matrix', 'camera_matrix_inv', and 'distortion_coeffs'
        """
        pass

    @staticmethod
    @abstractmethod
    def get_frame_paths(sequence_dir):
        """
        :return: sorted list of frame paths
        """
        pass

    @staticmethod
    @abstractmethod
    def get_frames_iterator(sequence_dir):
        pass

    @staticmethod
    @abstractmethod
    def get_events_iterator(sequence_dir, dt):
        pass

    @staticmethod
    @abstractmethod
    def get_events(sequence_dir):
        pass

    @staticmethod
    def get_event_paths(sequence_dir, representation, dt):
        """
        :return: sorted list of event paths for a given representation and time-delta
        """
        return sorted(
            glob(
                str(
                    sequence_dir / "events" / f"{dt:.4f}" / f"{representation}" / "*.h5"
                )
            )
        )

    @staticmethod
    def get_even_event_paths(sequence_dir, representation, r):
        """
        :return: sorted list of event paths for a given representation and time-delta
        """
        return sorted(glob(str(sequence_dir / "events" / f"pose_{r:.0f}" / "*.h5")))

    @staticmethod
    def get_keypoints(frame_start, corner_config: CornerConfig):
        """
        :param frame_start:
        :param max_keypoints:
        :return: (N, 2) torch float32 tensor of initial keypoint locations
        """
        keypoints = cv2.goodFeaturesToTrack(
            frame_start,
            maxCorners=15,
            qualityLevel=corner_config.qualityLevel,
            minDistance=corner_config.minDistance,
            k=corner_config.k,
            useHarrisDetector=corner_config.useHarrisDetector,
            blockSize=corner_config.blockSize,
        ).reshape((-1, 2))

        if keypoints.shape[0] == 0:
            print("No corners in frame")
            exit()

        elif keypoints.shape[0] > corner_config.maxCorners:
            indices = list(range(keypoints.shape[0]))
            sampled_indices = random.sample(indices, corner_config.maxCorners)
            keypoints = keypoints[sampled_indices, :]

        return torch.from_numpy(keypoints.astype(np.float32))

    def move_centers(self):
        self.u_centers = self.u_centers.to(self.device)
        self.u_centers_init = self.u_centers_init.to(self.device)
        self.x_ref = self.x_ref.to(self.device)

    def get_patches(self, f):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param f:
        :return:
        """
        # # OLD
        # # Round feature locations
        # self.u_centers = np.rint(self.u_centers)
        #
        # # Get patch crops
        # x_patches = []
        # for i_track in range(self.n_tracks):
        #     x_patches.append(get_patch_tensor(f, self.u_centers[i_track, :], self.patch_size))
        # x_patches = torch.cat(x_patches, dim=0)
        #
        # return x_patches

        if f.device != self.device:
            self.device = f.device
            self.move_centers()

        return extract_glimpse(
            f.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size, self.patch_size),
            self.u_centers.detach() + 0.5,
            mode="nearest",
        )

    def get_patches_new(self, arr_h5, padding=4):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param arr_h5: h5 file for the input event representation
        :return: (n_tracks, c, p, p) tensor
        """

        # Extract expanded patches from the h5 files
        u_centers_np = self.u_centers.detach().cpu().numpy()
        x_patches = []
        for i in range(self.n_tracks):
            u_center = u_centers_np[i, :]
            u_center_rounded = np.rint(u_center)
            u_center_offset = (
                u_center - u_center_rounded + ((self.patch_size + padding) // 2.0)
            )
            x_patch_expanded = get_patch_voxel(
                arr_h5, u_center_rounded.reshape((-1,)), self.patch_size + padding
            ).unsqueeze(0)
            x_patch = extract_glimpse(
                x_patch_expanded,
                (self.patch_size, self.patch_size),
                torch.from_numpy(u_center_offset).view((1, 2)) + 0.5,
                mode="nearest",
            )
            x_patches.append(x_patch)
        return torch.cat(x_patches, dim=0)


class EDSSubseqDatasetV2(SequenceDatasetV2):
    resolution = (640, 480)

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_events(sequence_dir):
        """
        :param sequence_dir:
        :return: event dict with keys t, x, y, p. Values are numpy arrays.
        """
        with h5py.File(str(sequence_dir / "events_corrected.h5")) as h5f:
            return {
                "x": np.array(h5f["x"]),
                "y": np.array(h5f["y"]),
                "t": np.array(h5f["t"]),
                "p": np.array(h5f["p"]),
            }

    @staticmethod
    def get_frames_iterator(sequence_dir):
        """
        :param sequence_dir: Path object
        :param dt: floating, seconds
        :return: Iterator over events between the frame timestamps
        """
        frame_paths = EDSSubseqDatasetV2.get_frame_paths(sequence_dir)
        frame_ts_arr = EDSSubseqDatasetV2.get_frame_timestamps(sequence_dir)
        assert len(frame_paths) == len(frame_ts_arr)

        for frame_idx in range(len(frame_paths)):
            yield frame_ts_arr[frame_idx], cv2.imread(
                frame_paths[frame_idx], cv2.IMREAD_GRAYSCALE
            )

    @staticmethod
    def get_events_iterator(sequence_dir, dt):
        """
        :param sequence_dir: Path object
        :param dt: floating, seconds
        :return: Iterator over events between the frame timestamps
        """
        events = EDSSubseqDatasetV2.get_events(sequence_dir)
        frame_ts_arr = EDSSubseqDatasetV2.get_frame_timestamps(sequence_dir)
        dt_elapsed = 0

        for t1 in np.arange(frame_ts_arr[0], frame_ts_arr[-1], dt):
            t1 = t1 * 1e6
            t0 = t1 - dt * 1e6
            idx0 = np.searchsorted(events["t"], t0, side="left")
            idx1 = np.searchsorted(events["t"], t1, side="right")

            yield dt_elapsed, {
                "x": events["x"][idx0:idx1],
                "y": events["y"][idx0:idx1],
                "p": events["p"][idx0:idx1],
                "t": events["t"][idx0:idx1],
            }

            dt_elapsed += dt

    @staticmethod
    def get_even_events_iterator(sequence_dir, r):
        """
        Return an iterator that (roughly) evenly splits events between frames into temporal bins.
        :param sequence_dir:
        :param r: number of temporal bins between frames
        :return:
        """
        events = EDSSubseqDatasetV2.get_events(sequence_dir)
        frame_ts_arr = EDSSubseqDatasetV2.get_frame_timestamps(sequence_dir)

        for i in range(len(frame_ts_arr) - 1):
            dt_us = (frame_ts_arr[i + 1] - frame_ts_arr[i]) * 1e6 // r

            t0 = frame_ts_arr[i] * 1e6
            for j in range(r):
                if j == r - 1:
                    t1 = frame_ts_arr[i + 1] * 1e6
                else:
                    t1 = t0 + dt_us

                idx0 = np.searchsorted(events["t"], t0, side="left")
                idx1 = np.searchsorted(events["t"], t1, side="right")
                yield t1 * 1e-6, {
                    "x": events["x"][idx0:idx1],
                    "y": events["y"][idx0:idx1],
                    "p": events["p"][idx0:idx1],
                    "t": events["t"][idx0:idx1],
                }
                t0 = t1

    @staticmethod
    def get_frame_paths(sequence_dir):
        return sorted(glob(str(sequence_dir / "images_corrected" / "*.png")))

    @staticmethod
    def get_frame_timestamps(sequence_dir):
        return np.genfromtxt(str(sequence_dir / "images_timestamps.txt")) * 1e-6

    @staticmethod
    def get_pose_interpolator(sequence_dir):
        colmap_pose_path = sequence_dir / "colmap" / "stamped_groundtruth.txt"
        if colmap_pose_path.exists():
            pose_data_path = sequence_dir / "colmap" / "stamped_groundtruth.txt"
            pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        else:
            pose_data = np.genfromtxt(
                str(sequence_dir / "stamped_groundtruth.txt"), skip_header=1
            )
        return PoseInterpolator(pose_data)

    @staticmethod
    def get_calibration(sequence_dir):
        with open(str(sequence_dir / ".." / "calib.yaml"), "r") as fh:
            data = yaml.load(fh, Loader=yaml.SafeLoader)["cam0"]
            camera_matrix = np.array(
                [
                    [data["intrinsics"][0], 0, data["intrinsics"][2]],
                    [0, data["intrinsics"][1], data["intrinsics"][3]],
                    [0, 0, 1],
                ]
            )
            camera_matrix_inv = np.linalg.inv(camera_matrix)
            distortion_coeffs = np.array(data["distortion_coeffs"]).reshape((-1,))
        return camera_matrix, camera_matrix_inv, distortion_coeffs


class EDSPoseSegmentDataset(EDSSubseqDatasetV2):
    def __init__(
        self, sequence_dir, idx_start, patch_size, representation, r=3, max_keypoints=2
    ):
        super().__init__()
        self.sequence_dir = sequence_dir
        self.idx_start = idx_start
        self.n_event_representations_per_frame = r
        self.patch_size = patch_size
        self.representation = representation
        self.device = torch.device("cpu")

        # Initial indices
        self.idx = self.idx_start
        self.event_representation_idx = (
            self.idx * self.n_event_representations_per_frame
        )
        self.initialize_pose_and_calib()
        self.initialize_time_and_input_paths(pose_mode=True, dt_or_r=r)

        # ToDO: Change back Pose Training
        max_keypoints = 4
        self.initialize_centers_and_ref_patches(
            CornerConfig(max_keypoints, 0.3, 15, 0.15, False, 11)
        )

    def get_next(self):
        self.event_representation_idx += 1
        self.t_now = (
            float(
                os.path.split(
                    self.event_representation_paths[self.event_representation_idx]
                )[1].replace(".h5", "")
            )
            * 1e-6
        )

        x_h5 = read_input(
            self.event_representation_paths[self.event_representation_idx],
            "time_surfaces_v2_5",
        )
        x_patches = self.get_patches_new(x_h5)
        x_patches = torch.cat([x_patches.to(self.x_ref.device), self.x_ref], dim=1)

        # # EPIPOLAR POSE SUPERVISION
        # # Get epipolar lines
        # T_now_W = self.pose_interpolator.interpolate(self.t_now)
        # T_first_now = np.linalg.inv(T_now_W @ np.linalg.inv(self.T_first_W))
        # F = self.camera_matrix_inv.T @ skew(T_first_now[:3, 3]) @ T_first_now[:3, :3] @ self.camera_matrix_inv
        # u_centers = self.u_centers.detach().cpu().numpy()
        # u_centers_homo = np.concatenate([u_centers, np.ones((u_centers.shape[0], 1))], axis=1)
        # l_epi = torch.from_numpy(u_centers_homo @ F)
        #
        # return x_patches, l_epi

        # REPROJECTION SUPERVISION
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
        T_now_first = T_now_W @ np.linalg.inv(self.T_first_W)
        projection_matrix = (self.camera_matrix @ T_now_first[:3, :]).astype(np.float32)
        projection_matrices = (
            torch.from_numpy(projection_matrix)
            .unsqueeze(0)
            .repeat(x_patches.size(0), 1, 1)
        )

        return x_patches, projection_matrices


class ECSubseqDatasetV2(SequenceDatasetV2):
    resolution = (240, 180)

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_events(sequence_dir):
        """
        :param sequence_dir:
        :return: event dict with keys t, x, y, p. Values are numpy arrays.
        """
        events = read_csv(
            str(sequence_dir / "events_corrected.txt"), delimiter=" "
        ).to_numpy()
        return {
            "x": events[:, 1],
            "y": events[:, 2],
            "t": events[:, 0],
            "p": events[:, 3],
        }

    @staticmethod
    def get_events_iterator(sequence_dir, dt):
        """
        :param sequence_dir: Path object
        :param dt: floating, seconds
        :return: Iterator over events between the frame timestamps
        """
        events = ECSubseqDatasetV2.get_events(sequence_dir)
        frame_ts_arr = ECSubseqDatasetV2.get_frame_timestamps(sequence_dir)
        dt_elapsed = 0

        for t1 in np.arange(frame_ts_arr[0], frame_ts_arr[-1], dt):
            t0 = t1 - dt
            idx0 = np.searchsorted(events["t"], t0, side="left")
            idx1 = np.searchsorted(events["t"], t1, side="right")

            yield dt_elapsed, {
                "x": events["x"][idx0:idx1],
                "y": events["y"][idx0:idx1],
                "p": events["p"][idx0:idx1],
                "t": events["t"][idx0:idx1],
            }

            dt_elapsed += dt

    @staticmethod
    def get_even_events_iterator(sequence_dir, r):
        """
        Return an iterator that (roughly) evenly splits events between frames into temporal bins.
        :param sequence_dir:
        :param r: number of temporal bins between frames
        :return:
        """
        events = ECSubseqDatasetV2.get_events(sequence_dir)
        frame_ts_arr = ECSubseqDatasetV2.get_frame_timestamps(sequence_dir)

        for i in range(len(frame_ts_arr) - 1):
            dt_us = (frame_ts_arr[i + 1] - frame_ts_arr[i]) * 1e6 // r

            t0 = frame_ts_arr[i] * 1e6
            for j in range(r):
                if j == r - 1:
                    t1 = frame_ts_arr[i + 1] * 1e6
                else:
                    t1 = t0 + dt_us

                idx0 = np.searchsorted(events["t"], t0 * 1e-6, side="left")
                idx1 = np.searchsorted(events["t"], (t1 + 1) * 1e-6, side="right")
                yield t1 * 1e-6, {
                    "x": events["x"][idx0:idx1],
                    "y": events["y"][idx0:idx1],
                    "p": events["p"][idx0:idx1],
                    "t": events["t"][idx0:idx1],
                }
                t0 = t1

    @staticmethod
    def get_frames_iterator(sequence_dir):
        """
        :param sequence_dir: Path object
        :param dt: floating, seconds
        :return: Iterator over events between the frame timestamps
        """
        frame_paths = ECSubseqDatasetV2.get_frame_paths(sequence_dir)
        frame_ts_arr = ECSubseqDatasetV2.get_frame_timestamps(sequence_dir)
        assert len(frame_paths) == len(frame_ts_arr)

        for frame_idx in range(len(frame_paths)):
            yield frame_ts_arr[frame_idx], cv2.imread(
                frame_paths[frame_idx], cv2.IMREAD_GRAYSCALE
            )

    @staticmethod
    def get_frame_paths(sequence_dir):
        return sorted(glob(str(sequence_dir / "images_corrected" / "*.png")))

    @staticmethod
    def get_frame_timestamps(sequence_dir):
        return np.genfromtxt(str(sequence_dir / "images.txt"), usecols=[0])

    @staticmethod
    def get_pose_interpolator(sequence_dir):
        if (sequence_dir / "colmap" / "stamped_groundtruth.txt").exists():
            pose_data_path = sequence_dir / "colmap" / "stamped_groundtruth.txt"
            pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        else:
            pose_data_path = sequence_dir / "groundtruth.txt"
            pose_data = np.genfromtxt(str(pose_data_path), skip_header=0, delimiter=" ")
        return PoseInterpolator(pose_data)

    @staticmethod
    def get_calibration(sequence_dir):
        intrinsics = np.genfromtxt(str(sequence_dir / "calib.txt"), delimiter=" ")
        camera_matrix = np.array(
            [
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1],
            ]
        )
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        distortion_coeffs = np.array(intrinsics[4:])
        return camera_matrix, camera_matrix_inv, distortion_coeffs


class ECPoseSegmentDataset(ECSubseqDatasetV2):
    def __init__(
        self, sequence_dir, idx_start, patch_size, representation, r=3, max_keypoints=2
    ):
        super().__init__()
        self.sequence_dir = sequence_dir
        self.idx_start = idx_start
        self.n_event_representations_per_frame = r
        self.patch_size = patch_size
        self.representation = representation
        self.device = torch.device("cpu")

        # ToDO: Change back Pose Training
        max_keypoints = 4

        # Initial indices
        self.idx = self.idx_start
        self.event_representation_idx = (
            self.idx * self.n_event_representations_per_frame
        )
        self.initialize_pose_and_calib()
        self.initialize_time_and_input_paths(pose_mode=True, dt_or_r=r)
        self.initialize_centers_and_ref_patches(
            CornerConfig(max_keypoints, 0.3, 15, 0.15, False, 11)
        )

    def get_next(self):
        self.event_representation_idx += 1
        self.t_now = (
            float(
                os.path.split(
                    self.event_representation_paths[self.event_representation_idx]
                )[1].replace(".h5", "")
            )
            * 1e-6
        )

        x_h5 = read_input(
            self.event_representation_paths[self.event_representation_idx],
            "time_surfaces_v2_5",
        )
        x_patches = self.get_patches_new(x_h5)
        x_patches = torch.cat([x_patches.to(self.x_ref.device), self.x_ref], dim=1)

        # # EPIPOLAR POSE SUPERVISION
        # # Get epipolar lines
        # T_now_W = self.pose_interpolator.interpolate(self.t_now)
        # T_first_now = np.linalg.inv(T_now_W @ np.linalg.inv(self.T_first_W))
        # F = self.camera_matrix_inv.T @ skew(T_first_now[:3, 3]) @ T_first_now[:3, :3] @ self.camera_matrix_inv
        # u_centers = self.u_centers.detach().cpu().numpy()
        # u_centers_homo = np.concatenate([u_centers, np.ones((u_centers.shape[0], 1))], axis=1)
        # l_epi = torch.from_numpy(u_centers_homo @ F)
        #
        # return x_patches, l_epi

        # REPROJECTION SUPERVISION
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
        T_now_first = T_now_W @ np.linalg.inv(self.T_first_W)
        projection_matrix = (self.camera_matrix @ T_now_first[:3, :]).astype(np.float32)
        projection_matrices = (
            torch.from_numpy(projection_matrix)
            .unsqueeze(0)
            .repeat(x_patches.size(0), 1, 1)
        )

        return x_patches, projection_matrices


Mapping_DatasetType2SegmentClass = {
    EvalDatasetType.EDS: EDSSubseqDatasetV2,
    EvalDatasetType.EC: ECSubseqDatasetV2,
}


class SubSequenceRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        # n_samples_per_seq = 8
        n_samples_per_seq = 32
        shifted_start = torch.randint(n_samples_per_seq, [1])
        shifted_indices = self.indices[shifted_start:] + self.indices[-shifted_start:]

        for i in torch.randperm(math.ceil(len(self.indices) / n_samples_per_seq)):
            i_idx = i * n_samples_per_seq

            for i_yield in range(i_idx, min(i_idx + n_samples_per_seq, self.__len__())):
                yield shifted_indices[i_yield]

    def __len__(self) -> int:
        return len(self.indices)
