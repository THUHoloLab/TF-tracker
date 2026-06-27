import cv2
import hydra
import matplotlib
import numpy as np
import torch.optim.lr_scheduler
from pytorch_lightning import LightningModule
from tensorflow_graphics.projects.points_to_3Dobjects.losses.focal_loss import FocalLoss

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.dataset import TrackletAugmentor

from utils.losses import *
import os

class Template(LightningModule):
    def __init__(
        self,
        representation="time_surfaces_1",
        max_unrolls=16,
        n_vis=8,
        patch_size=31,
        init_unrolls=10,
        pose_mode=False,
        debug=True,
        **kwargs,
    ):
        super(Template, self).__init__()
        self.save_hyperparameters()

        # High level model config
        self.representation = representation
        self.patch_size = patch_size
        self.debug = debug
        self.model_type = "non_global"
        self.pose_mode = pose_mode

        # Determine num channels from representation name
        if "grayscale" in representation:
            self.channels_in_per_patch = 1
        else:
            self.channels_in_per_patch = int(representation[-1])

            if "v2" in self.representation:
                self.channels_in_per_patch *= (
                    2  # V2 representations have separate channels for each polarity
                )

        # Loss Function
        self.loss = None
        self.loss_reproj = ReprojectionError(threshold=self.patch_size / 2)
        self.focalloss = FastFocalLoss()
        self.consistencyloss = L1Consistency()

        # Training variables
        self.unrolls = init_unrolls
        self.max_unrolls = max_unrolls
        self.n_vis = n_vis
        self.colormap = cm.get_cmap("inferno")
        self.graymap = cm.get_cmap("gray")

        # Validation variables
        self.epe_l2_hist = []
        self.l2 = L2Distance()

    def configure_optimizers(self):
        if not self.debug:
            opt = hydra.utils.instantiate(
                self.hparams.optimizer, params=self.parameters()
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        opt,
                        self.hparams.optimizer.lr,
                        total_steps=100000, # 1000000
                        pct_start=0.002,
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                    "name": "lr",
                },
            }
        else:
            return hydra.utils.instantiate(
                self.hparams.optimizer, params=self.parameters()
            )
    '''

                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt,
                        T_max=1000000,  # 总 step 数
                        eta_min=1e-6,  # 最小学习率
                    ),
    '''

    def forward(self, x, heatmap, n_tracks,attn_mask=None):
        return None

    def on_train_epoch_end(self, *args):
        return

    def generate_heatmap_init(self, n_tracks):

        '''heatmap = torch.zeros(1, self.H, self.W)

        x = u_center[:,0].round().long()
        y = u_center[:,1].round().long()
        heatmap[0, y, x] = 1.0'''

        H = self.patch_size
        W = self.patch_size
        device = self.device

        # 创建空 heatmap
        heatmap = torch.zeros((n_tracks, H, W), device=device)

        kernel_size = 7
        sigma = 1.5

        coords = torch.arange(kernel_size, device=device) - kernel_size // 2
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')

        gaussian_kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))

        # 归一化
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max()

        # ===== 放到 patch 中心 =====
        center_y = H // 2
        center_x = W // 2

        half_k = kernel_size // 2

        y1 = center_y - half_k
        y2 = center_y + half_k + 1
        x1 = center_x - half_k
        x2 = center_x + half_k + 1

        heatmap[:, y1:y2, x1:x2] = gaussian_kernel

        return heatmap

    def generate_heatmap_gt(self, u_centers):

        device = u_centers.device
        B, N, _ = u_centers.shape
        H, W = self.H, self.W

        heatmap = torch.zeros(B, H, W, device=device)

        # ===== 生成 7×7 Gaussian 核 =====
        kernel_size = 7
        sigma = 1.5
        half_k = kernel_size // 2

        coords = torch.arange(kernel_size, device=device) - half_k
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')

        gaussian_kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
        # (7,7)

        # ===== 获取目标中心坐标 =====
        x = u_centers[:, :, 0].round().long()  # [B,N]
        y = u_centers[:, :, 1].round().long()  # [B,N]

        # ===== 构造局部网格 =====
        grid_y = y.unsqueeze(-1).unsqueeze(-1) + yy  # [B,N,7,7]
        grid_x = x.unsqueeze(-1).unsqueeze(-1) + xx  # [B,N,7,7]

        # ===== 边界 mask =====
        valid = (
                (grid_x >= 0) & (grid_x < W) &
                (grid_y >= 0) & (grid_y < H)
        )

        # ===== batch index =====
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, 7, 7)

        # ===== kernel expand =====
        kernel_expand = gaussian_kernel.view(1, 1, 7, 7).expand(B, N, 7, 7)

        # ===== 写入 heatmap =====
        heatmap[
            batch_idx[valid],
            grid_y[valid],
            grid_x[valid]
        ] = kernel_expand[valid]
        heatmap = heatmap.unsqueeze(1)

        return heatmap

    def sample_y_hat(self, y_hat, u_centers):

        B, C, H, W = y_hat.shape
        _, N, _ = u_centers.shape

        device = y_hat.device

        # -----------------------------
        # 1 提取坐标
        # -----------------------------
        x = u_centers[..., 0].long().clamp(0, W - 1)
        y = u_centers[..., 1].long().clamp(0, H - 1)

        # -----------------------------
        # 2 batch索引
        # -----------------------------
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)

        # -----------------------------
        # 3 从y_hat采样
        # -----------------------------
        y_hat_sampled = y_hat[
                        batch_idx,  # B,N
                        :,  # channel=2
                        y,  # B,N
                        x  # B,N
                        ]

        return y_hat_sampled

    def training_step(self, batch_dataloaders, batch_nb):
        if self.pose_mode:
            # Freeze batchnorm running values for fine-tuning
            self.reference_encoder = self.reference_encoder.eval()
            self.target_encoder = self.target_encoder.eval()
            self.reference_redir = self.reference_redir.eval()
            self.target_redir = self.target_redir.eval()
            self.joint_encoder = self.joint_encoder.eval()
            self.predictor = self.predictor.eval()

        # Determine number of tracks in batch
        nb = len(batch_dataloaders)
        if self.pose_mode:
            nt = 0
            for bl in batch_dataloaders:
                nt += bl.n_tracks
        else:
            nt = nb

        if not self.pose_mode:
            x_ref = batch_dataloaders[0].x_ref
            _, self.H, self.W = x_ref.shape
            u_centers_init = []
            for bl in batch_dataloaders:
                bl.auto_update_center = False
                u_centers_init.append(torch.from_numpy(bl.u_center_init))
            u_centers_init = (
                torch.stack(u_centers_init, dim=0).to(self.device)
            )
        else:
            u_centers_init = []
            for bl in batch_dataloaders:
                u_centers_init.append(bl.u_centers)
            u_centers_init = (
                torch.cat(u_centers_init, dim=0).to(self.device).unsqueeze(1)
            )
            u_centers_hist = [u_centers_init]
            projection_matrices_hist = [
                torch.cat(
                    [
                        torch.from_numpy(
                            batch_dataloaders[0].camera_matrix.astype(np.float32)
                        ),
                        torch.zeros((3, 1), dtype=torch.float32),
                    ],
                    dim=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(u_centers_init.size(0), 1, 1, 1)
                .to(self.device)
            ]

        # Unroll network
        loss_total = torch.zeros(1, dtype=torch.float32, device=self.device)
        # loss_mask_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        self.reset(nt)

        if self.pose_mode:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            i_src = 0
            for bl_src in batch_dataloaders:
                n_src_tracks = bl_src.n_tracks
                attn_mask[
                i_src: i_src + n_src_tracks, i_src: i_src + n_src_tracks
                ] = 1
                i_src += n_src_tracks
        else:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            for i_src in range(nt):
                src_path = batch_dataloaders[i_src].track_path.split("/")[-3]
                for i_target in range(nt):
                    attn_mask[i_src, i_target] = (
                            src_path
                            == batch_dataloaders[i_target].track_path.split("/")[-3]
                    )
        attn_mask = (1 - attn_mask).bool()
        #scene_name = batch_dataloaders[0].event_paths[0]
        #scene_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(scene_name)))))
        '''heatmap_augmentor = TrackletAugmentor(H=self.H, W=self.H)
        heatmap_pre = self.generate_heatmap_gt(u_centers_init)
        heatmap_init = heatmap_pre.clone()
        #heatmap_pre_prev = heatmap_pre.clone()
        for i_unroll in range(self.unrolls):
            # Construct batched x and y for current timestep
            x, y, u_centers_gt = [], [], []
            for bl in batch_dataloaders:
                bl.auto_update_center = True
                x_j, y_j, u_centers_gt_j = bl.get_next()
                x.append(x_j)
                y.append(y_j)
                u_centers_gt.append(u_centers_gt_j)
            x = torch.cat(x, dim=0).to(self.device)
            y = torch.cat(y, dim=0).to(self.device)
            u_centers_gt = torch.cat(u_centers_gt, dim=0).to(self.device)
            #heatmap_pre = heatmap_pre.squeeze(1)
            # Inference
            y_hat, heatmap = self.forward(x, torch.cat((heatmap_pre, heatmap_init), dim=1), attn_mask)
            new_heatmap = heatmap_augmentor.augment_and_generate(u_centers_gt)  # heatmap.clone()
            #heatmap_pre_prev = heatmap_pre  # t-1 → t-2
            heatmap_pre = new_heatmap  # 当前 → t-1
            heatmap_gt = self.generate_heatmap_gt(u_centers_gt)

            x_sum = torch.sum(x[0, 1:10], dim=0)
            heatmap_np = heatmap_gt[0, 0].detach().cpu().numpy()+x_sum.detach().cpu().numpy()
            heatmap_np = (heatmap_np * 255).astype(np.uint8)
            filename = f"{scene_name}_{i_unroll}.png"
            save_path = os.path.join("/data/cyt2/TF_tracker/heatmap_gt/", filename)
            cv2.imwrite(save_path, heatmap_np)'''


        def _fetch_batch(batch_dataloaders, reverse=False):
            xs, ys, us = [], [], []
            for bl in batch_dataloaders:
                bl.auto_update_center = True
                if reverse:
                    x_i, y_i, u_i = bl.get_prev()
                else:
                    x_i, y_i, u_i = bl.get_next()
                xs.append(x_i)
                ys.append(y_i)
                us.append(u_i)

            x = torch.cat(xs, dim=0).to(self.device, non_blocking=True)
            y = torch.cat(ys, dim=0).to(self.device, non_blocking=True)
            u = torch.cat(us, dim=0).to(self.device, non_blocking=True)
            return x, y, u

        heatmap_init = self.generate_heatmap_gt(u_centers_init)
        _, n_tracks, _ = u_centers_init.shape
        heatmap_pre = heatmap_init
        heatmap_pre_pre = heatmap_init
        heatmap_augmentor = TrackletAugmentor(H=self.H, W=self.W)

        #forward_y_hat_list = []
        #forward_center_list = []

        # =========================================================
        # forward
        # =========================================================
        for i_unroll in range(self.unrolls):
            x, y, u_centers_gt = _fetch_batch(batch_dataloaders, reverse=False)

            flow_target = u_centers_gt - y

            y_hat, heatmap = self.forward(
                x,
                torch.cat((heatmap_pre_pre, heatmap_pre, heatmap_init), dim=1),
                i_unroll,
                attn_mask
            )

            heatmap_gt = self.generate_heatmap_gt(u_centers_gt)
            heatmap_pre_pre = heatmap_pre
            heatmap_pre = heatmap_augmentor.augment_and_generate(u_centers_gt)
            #heatmap_pre = heatmap_gt

            # 为 consistency 保留必要信息
            #forward_y_hat_list.append(y_hat)
            #forward_center_list.append(flow_target)

            if self.pose_mode:
                u_centers = torch.cat(
                    [bl.u_centers for bl in batch_dataloaders], dim=0
                ).to(self.device, non_blocking=True)

                u_centers_hist.append(
                    u_centers.unsqueeze(1).detach() + y_hat.unsqueeze(1)
                )
                projection_matrices_hist.append(y.unsqueeze(1))
            else:
                loss_total += 1.0 * self.loss(y, y_hat, flow_target)
                loss_total += 1.0 * self.focalloss(heatmap, heatmap_gt, u_centers_gt)

            # Pass predicted flow to dataloader
            y_hat_sampled = self.sample_y_hat(y_hat, u_centers_gt)

            if self.pose_mode:
                idx_acc = 0
                for j in range(nb):
                    n_tracks = batch_dataloaders[j].n_tracks
                    batch_dataloaders[j].accumulate_y_hat(
                        y_hat_sampled[idx_acc: idx_acc + n_tracks, :]
                    )
                    idx_acc += n_tracks
            else:
                for j in range(nb):
                    batch_dataloaders[j].accumulate_y_hat(y_hat_sampled[j, ...])

        # backward 初始化
        '''heatmap_init = heatmap_gt
        heatmap_pre = heatmap_init
        heatmap_pre_pre = heatmap_init

        backward_y_hat_list = []
        backward_center_list = []

        # =========================================================
        # backward
        # =========================================================
        for i_unroll in range(self.unrolls):
            x_rev, y_rev, u_centers_gt_rev = _fetch_batch(batch_dataloaders, reverse=True)

            flow_target_rev = u_centers_gt_rev - y_rev

            y_hat_rev, heatmap_rev = self.forward(
                x_rev,
                torch.cat((heatmap_pre_pre, heatmap_pre), dim=1),
                n_tracks,
                attn_mask
            )

            heatmap_gt_rev = self.generate_heatmap_gt(u_centers_gt_rev)
            heatmap_pre_pre = heatmap_pre
            heatmap_pre = heatmap_augmentor.augment_and_generate(u_centers_gt_rev)

            backward_y_hat_list.append(y_hat_rev)
            backward_center_list.append(flow_target_rev)

            if self.pose_mode:
                u_centers = torch.cat(
                    [bl.u_centers for bl in batch_dataloaders], dim=0
                ).to(self.device, non_blocking=True)

                u_centers_hist.append(
                    u_centers.unsqueeze(1).detach() + y_hat_rev.unsqueeze(1)
                )
                projection_matrices_hist.append(y_rev.unsqueeze(1))
            else:
                loss_total += 50.0 * self.loss(y_rev, y_hat_rev, flow_target_rev)
                loss_total += 100.0 * self.focalloss(
                    heatmap_rev, heatmap_gt_rev, u_centers_gt_rev
                )

        # =========================================================
        # forward-backward consistency
        # =========================================================
        loss_fb = 0.0
        for fwd, bwd, c_fwd, c_bwd in zip(
                forward_y_hat_list,
                reversed(backward_y_hat_list),
                forward_center_list,
                reversed(backward_center_list),
        ):
            loss_fb += 10.0 * self.consistencyloss(fwd, -bwd, c_fwd, c_bwd)

        loss_total += loss_fb'''

        # Average out losses (ignoring the masked out steps)
        if self.pose_mode:
            u_centers_hist = torch.cat(u_centers_hist, dim=1)
            projection_matrices_hist = torch.cat(projection_matrices_hist, dim=1)
            loss_total, loss_mask_total = self.loss_reproj.forward(
                projection_matrices_hist, u_centers_hist
            )

            loss_total = loss_total.sum(1)
            loss_mask_total = loss_mask_total.sum(1)

            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()

        self.log(
            "loss/train",
            loss_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=nb,
        )

        return loss_total

    def on_validation_epoch_start(self):
        # Reset distribution monitors
        self.epe_l2_hist = []
        self.track_error_hist = []
        self.feature_age_hist = []

    def validation_step(self, batch_dataloaders, batch_nb):
        # Determine number of tracks in batch
        nb = len(batch_dataloaders)
        if self.pose_mode:
            nt = 0
            for bl in batch_dataloaders:
                nt += bl.n_tracks
        else:
            nt = nb

        # Validation Metrics
        if not self.pose_mode:
            metrics = {
                "feature_age": torch.zeros(nb, dtype=torch.float32, device="cpu"),
                "tracking_error": [[] for _ in range(nb)],
            }
            x_ref = batch_dataloaders[0].x_ref
            _, self.H, self.W = x_ref.shape

            u_centers_init = []
            for bl in batch_dataloaders:
                bl.auto_update_center = False
                u_centers_init.append(torch.from_numpy(bl.u_center_init))
            u_centers_init = (
                torch.stack(u_centers_init, dim=0).to(self.device)
            )
        else:
            u_centers_init = []
            for bl in batch_dataloaders:
                u_centers_init.append(bl.u_centers)
            u_centers_init = (
                torch.cat(u_centers_init, dim=0).to(self.device).unsqueeze(1)
            )
            u_centers_hist = [u_centers_init]
            projection_matrices_hist = [
                torch.cat(
                    [
                        torch.from_numpy(
                            batch_dataloaders[0].camera_matrix.astype(np.float32)
                        ),
                        torch.zeros((3, 1), dtype=torch.float32),
                    ],
                    dim=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(u_centers_init.size(0), 1, 1, 1)
                .to(self.device)
            ]

        # Unroll network
        loss_total = torch.zeros(1, dtype=torch.float32, device=self.device)
        #loss_mask_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        self.reset(nt)

        if self.pose_mode:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            i_src = 0
            for bl_src in batch_dataloaders:
                n_src_tracks = bl_src.n_tracks
                attn_mask[
                    i_src : i_src + n_src_tracks, i_src : i_src + n_src_tracks
                ] = 1
                i_src += n_src_tracks
        else:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            for i_src in range(nt):
                src_path = batch_dataloaders[i_src].track_path.split("/")[-3]
                for i_target in range(nt):
                    attn_mask[i_src, i_target] = (
                        src_path
                        == batch_dataloaders[i_target].track_path.split("/")[-3]
                    )
        attn_mask = (1 - attn_mask).bool()

        def _fetch_batch(batch_dataloaders):
            xs, ys, us = [], [], []
            for bl in batch_dataloaders:
                bl.auto_update_center = True
                x_i, y_i, u_i = bl.get_next()
                xs.append(x_i)
                ys.append(y_i)
                us.append(u_i)

            x = torch.cat(xs, dim=0).to(self.device, non_blocking=True)
            y = torch.cat(ys, dim=0).to(self.device, non_blocking=True)
            u = torch.cat(us, dim=0).to(self.device, non_blocking=True)
            return x, y, u

        heatmap_init = self.generate_heatmap_gt(u_centers_init)
        _, n_tracks, _ = u_centers_init.shape
        heatmap_pre = heatmap_init
        heatmap_pre_pre = heatmap_init

        for i_unroll in range(self.unrolls):
            # -------------------------
            # 1) 取 batch
            # -------------------------
            x, y, u_centers_gt = _fetch_batch(batch_dataloaders)
            flow_target = u_centers_gt - y

            # -------------------------
            # 2) forward
            # -------------------------
            y_hat, heatmap = self.forward(
                x,
                torch.cat((heatmap_pre_pre, heatmap_pre, heatmap_init), dim=1),
                i_unroll,
                attn_mask
            )

            # heatmap 递推
            heatmap_pre_pre = heatmap_pre
            heatmap_pre = heatmap
            heatmap_gt = self.generate_heatmap_gt(u_centers_gt)

            # -------------------------
            # 3) loss
            # -------------------------
            if self.pose_mode:
                u_centers = torch.cat(
                    [bl.u_centers for bl in batch_dataloaders], dim=0
                ).to(self.device, non_blocking=True)

                # Reprojection Loss
                u_centers_hist.append(
                    u_centers.unsqueeze(1).detach() + y_hat.unsqueeze(1)
                )
                projection_matrices_hist.append(y.unsqueeze(1))
            else:
                loss_total += 1.0 * self.loss(y, y_hat, flow_target)
                loss_total += 1.0 * self.focalloss(heatmap, heatmap_gt, u_centers_gt)

            # -------------------------
            # 4) 把预测 flow 回写给 dataloader
            # -------------------------
            y_hat_sampled = self.sample_y_hat(y_hat, u_centers_gt)

            if self.pose_mode:
                idx_acc = 0
                for j in range(nb):
                    n_tracks = batch_dataloaders[j].n_tracks
                    batch_dataloaders[j].accumulate_y_hat(
                        y_hat_sampled[idx_acc: idx_acc + n_tracks, :]
                    )
                    idx_acc += n_tracks
            else:
                for j in range(nb):
                    batch_dataloaders[j].accumulate_y_hat(y_hat_sampled[j, ...])

            # -------------------------
            # 5) 可视化缓存（只在 first batch）
            # -------------------------
            '''
            if batch_nb == 0:
                x_hat_hist.append(
                    torch.max(x[:, :-1, :, :], dim=1, keepdim=True)[0].detach()
                )
                x_ref_hist.append(
                    x[:, -1:, :, :].detach()
                )

            # -------------------------
            # 6) metrics
            # -------------------------
            if not self.pose_mode:
                dist_list = []
                y_hat_total_list = []
                y_total_list = []

                for j in range(nb):
                    y_hat_total_j = (
                            batch_dataloaders[j].u_center - batch_dataloaders[j].u_center_init
                    )
                    y_total_j = (
                            batch_dataloaders[j].u_center_gt - batch_dataloaders[j].u_center_init
                    )
                    dist_j = np.linalg.norm(
                        batch_dataloaders[j].u_center_gt - batch_dataloaders[j].u_center
                    )

                    y_hat_total_list.append(y_hat_total_j)
                    y_total_list.append(y_total_j)
                    dist_list.append(dist_j)

                y_total = torch.from_numpy(np.asarray(y_total_list))
                y_hat_total = torch.from_numpy(np.asarray(y_hat_total_list))
                dist = torch.from_numpy(np.asarray(dist_list))

                live_track_idxs = torch.nonzero(dist < self.H, as_tuple=False)
                for idx in live_track_idxs:
                    metrics["feature_age"][idx] = (i_unroll + 1) * 0.01
                    if self.representation == "grayscale":
                        metrics["feature_age"] *= 5
                    metrics["tracking_error"][idx].append(dist[idx].item())

                if batch_nb == 0:
                    y_total_hist.append(y_total.unsqueeze(0))
                    y_hat_total_hist.append(y_hat_total.unsqueeze(0))

        # =========================================================
        # pose mode 的最终 loss
        # =========================================================
        if self.pose_mode:
            u_centers_hist = torch.cat(u_centers_hist, dim=1)
            projection_matrices_hist = torch.cat(projection_matrices_hist, dim=1)

            loss_total, loss_mask_total, u_centers_reproj = self.loss_reproj.forward(
                projection_matrices_hist, u_centers_hist, training=False
            )
            loss_hist = loss_total.clone()

            loss_total = loss_total.sum(1)
            loss_mask_total = loss_mask_total.sum(1)

            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()

        '''
        self.log(
            "loss/val",
            loss_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        '''
        # Log predicted patches for both training modes
        if batch_nb == 0:
            x_hat_hist = torch.cat(x_hat_hist, dim=0)
            x_ref_hist = torch.cat(x_ref_hist, dim=0)

        # Log GT patch visualizations and track metrics for GT supervision
        if not self.pose_mode:
            # Log scalars
            for j in range(nb):
                if len(metrics["tracking_error"][j]):
                    self.track_error_hist.append(np.mean(metrics["tracking_error"][j]))

            self.feature_age_hist += (
                metrics["feature_age"]
                .numpy()
                .reshape(
                    -1,
                )
                .tolist()
            )

            dist = self.l2(y_total, y_hat_total)
            self.epe_l2_hist += (
                dist.detach()
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
                .tolist()
            )

            # Visualize some predicted patch trajectories
            if batch_nb == 0:
                # Get gt patches
                for j in range(nb):
                    batch_dataloaders[j].reset()
                    batch_dataloaders[j].auto_update_center = True

                for i_unroll in range(self.unrolls + 1):
                    x = []
                    for j in range(nb):
                        x_j, _,_ = batch_dataloaders[j].get_next()
                        x.append(x_j)
                    x = torch.cat(x, dim=0).to(self.device)
                    x_hist.append(
                        torch.max(x[:, :-1, :, :], dim=1, keepdim=True)[0]
                        .detach()
                        .clone()
                    )

                # Concatenate along time axis
                y_total_hist = torch.cat(y_total_hist, dim=0)
                y_hat_total_hist = torch.cat(y_hat_total_hist, dim=0)
                x_hist = torch.cat(x_hist, dim=0).squeeze(1)

                with plt.style.context("ggplot"):
                    for i_vis in range(self.n_vis):
                        # Flow histories
                        fig = plt.figure()
                        ax = fig.add_subplot()

                        traj = y_total_hist[:,0,i_vis, :].cpu().numpy()
                        traj_hat = y_hat_total_hist[:,0,i_vis, :].cpu().numpy()

                        ax.plot(traj[:, 0], traj[:, 1], color="g")
                        ax.plot(traj_hat[:, 0], traj_hat[:, 1], color="b")

                        plt_lims = ax.get_xlim()
                        plt_lims += ax.get_ylim()
                        plt_lims = max([abs(x) for x in plt_lims])
                        ax.set_xlim([-plt_lims, plt_lims])
                        ax.set_ylim([-plt_lims, plt_lims])
                        ax.set_xticks(
                            np.linspace(np.floor(-plt_lims), np.ceil(plt_lims), 10)
                        )
                        ax.set_yticks(
                            np.linspace(np.floor(-plt_lims), np.ceil(plt_lims), 10)
                        )

                        ax.set_aspect("equal")
                        ax.set_title(f"Val Batch 0 - Patch {i_vis}")
                        self.logger.experiment.add_figure(
                            f"cumulative_flow/patch_{i_vis}", fig, self.global_step
                        )

                        # Patches
                        img_traj = torch.from_numpy(
                            self.colormap(x_hist[0].cpu().numpy())[..., :3]
                        )
                        self.logger.experiment.add_image(
                            f"time_surface_traj/patch_{i_vis}",
                            img_traj,
                            self.global_step,
                            dataformats="HWC",
                        )'''

        return loss_total

    def on_validation_epoch_end(self):
        if self.pose_mode is False:
            # L2 error cumsum
            with plt.style.context("ggplot"):
                fig = plt.figure()
                x, counts = np.unique(self.epe_l2_hist, return_counts=True)
                y = np.cumsum(counts) / np.sum(counts)
                ax = fig.add_subplot()
                ax.plot(x, y)
                ax.set_xlabel("EPE (px)")
                ax.set_ylabel("Proportion")
                self.logger.experiment.add_figure(
                    "l2_cumsum/val", fig, self.global_step
                )
                plt.close("all")
            '''self.epe_l2_hist = np.array(self.epe_l2_hist, dtype=np.float32)
            self.logger.experiment.add_histogram(
                "EPE_hist/val", self.epe_l2_hist, self.global_step
            )'''

            self.log("EPE_median/val", np.median(self.epe_l2_hist))
            self.log("TE_median/val", np.median(self.track_error_hist))
            self.log("TE_mean/val", np.mean(self.track_error_hist))
            self.log("EPE_mean/val", np.mean(self.epe_l2_hist))
            self.log("TE_std/val", np.std(self.track_error_hist))
            self.log("EPE_std/val", np.std(self.epe_l2_hist))
            self.log(f"FA_median/val", np.median(self.feature_age_hist))
            self.log(f"FA_mean/val", np.mean(self.feature_age_hist))
