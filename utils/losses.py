import torch
import torch.nn as nn


'''class L1Truncated(nn.Module):
    """
    L1 Loss, but zero if label is outside the patch
    """

    def __init__(self, patch_size=31):
        super(L1Truncated, self).__init__()
        self.patch_size = patch_size
        self.L1 = nn.L1Loss(reduction="none")

    def forward(self, y, y_hat):
        self.mask = (
            (torch.abs(y) <= self.patch_size / 2.0)
            .all(dim=1)
            .float()
            .detach()
            .requires_grad_(True)
        )
        loss = self.L1(y, y_hat).sum(1)
        loss *= self.mask
        return loss, self.mask'''

class L1Truncated(nn.Module):

    def __init__(self):
        super(L1Truncated, self).__init__()
        self.L1 = nn.L1Loss(reduction="none")

    def forward(self, y, y_hat, centers):

        B, N, _ = y.shape
        H, W = y_hat.shape[2:]

        # -------------------------
        # 1 生成mask
        # -------------------------
        mask = (
                (centers[..., 0] > 0) &
                (centers[..., 0] < (W-1)) &
                (centers[..., 1] > 0) &
                (centers[..., 1] < (H-1))
        ).float()

        # -------------------------
        # 2 从y_hat采样
        # -------------------------
        x_c = centers[..., 0].long().clamp(0, W - 1)
        y_c = centers[..., 1].long().clamp(0, H - 1)

        batch_idx = torch.arange(B, device=y_hat.device).view(B, 1).expand(B, N)

        y_hat_sampled = y_hat[
            batch_idx,
            :,
            y_c,
            x_c
        ]  # [B,N,2]

        # -------------------------
        # 3 L1 loss
        # -------------------------
        loss = self.L1(y, y_hat_sampled).sum(dim=2)  # [B,N]

        # -------------------------
        # 4 mask
        # -------------------------
        loss = loss * mask

        # -------------------------
        # 5 mean
        # -------------------------
        loss = loss.sum() / (mask.sum() + 1e-6)

        return loss

class L1Consistency(nn.Module):
    def __init__(self):
        super(L1Consistency, self).__init__()
        self.L1 = nn.L1Loss(reduction="none")

    def forward(self, y, y_rev, centers, centers_rev):
        """
        Args:
            y:        [B, 2, H, W] 或 [B, C, H, W]，正向位移场
            y_rev:    [B, 2, H, W] 或 [B, C, H, W]，反向位移场
            centers:      [B, N, 2]，从 y 中采样的位置
            centers_rev:  [B, N, 2]，从 y_rev 中采样的位置

        Returns:
            scalar loss
        """
        B, _, H, W = y.shape
        N = centers.shape[1]

        # -------------------------
        # 1) 分别生成两个mask
        # -------------------------
        mask_y = (
            (centers[..., 0] >= 0) &
            (centers[..., 0] <= (W - 1)) &
            (centers[..., 1] >= 0) &
            (centers[..., 1] <= (H - 1))
        ).float()

        mask_y_rev = (
            (centers_rev[..., 0] >= 0) &
            (centers_rev[..., 0] <= (W - 1)) &
            (centers_rev[..., 1] >= 0) &
            (centers_rev[..., 1] <= (H - 1))
        ).float()

        # 只有两个位置都有效才参与损失
        mask = mask_y * mask_y_rev  # [B, N]

        # -------------------------
        # 2) 从 y 中按 centers 采样
        # -------------------------
        x_c = centers[..., 0].long().clamp(0, W - 1)
        y_c = centers[..., 1].long().clamp(0, H - 1)

        batch_idx = torch.arange(B, device=y.device).view(B, 1).expand(B, N)

        y_sampled = y[
            batch_idx,
            :,
            y_c,
            x_c
        ]  # [B, N, C]

        # -------------------------
        # 3) 从 y_rev 中按 centers_rev 采样
        # -------------------------
        x_c_rev = centers_rev[..., 0].long().clamp(0, W - 1)
        y_c_rev = centers_rev[..., 1].long().clamp(0, H - 1)

        y_rev_sampled = y_rev[
            batch_idx,
            :,
            y_c_rev,
            x_c_rev
        ]  # [B, N, C]

        # -------------------------
        # 4) 计算采样点的偏差
        # -------------------------
        loss = self.L1(y_sampled, y_rev_sampled).sum(dim=2)  # [B, N]

        # -------------------------
        # 5) mask
        # -------------------------
        loss = loss * mask

        # -------------------------
        # 6) mean
        # -------------------------
        loss = loss.sum() / (mask.sum() + 1e-6)

        return loss

class ReprojectionError:
    def __init__(self, threshold=15):
        self.threshold = threshold

    def forward(self, projection_matrices, u_centers_hat, training=True):
        """
        :param projection_matrices: (B, T, 3, 4)
        :param u_centers_hat: (B, T, 2)
        :return: (N, T) re-projection errors, (N, T) masks
        """
        e_reproj, masks, u_centers_reproj = [], [], []

        for idx_track in range(u_centers_hat.size(0)):
            A_rows = []

            # Triangulate
            for idx_obs in range(u_centers_hat.size(1)):
                A_rows.append(
                    u_centers_hat[idx_track, idx_obs, 0]
                    * projection_matrices[idx_track, idx_obs, 2:3, :]
                    - projection_matrices[idx_track, idx_obs, 0:1, :]
                )
                A_rows.append(
                    u_centers_hat[idx_track, idx_obs, 1]
                    * projection_matrices[idx_track, idx_obs, 2:3, :]
                    - projection_matrices[idx_track, idx_obs, 1:2, :]
                )
            A = torch.cat(A_rows, dim=0)
            _, s, vh = torch.linalg.svd(A)
            X_init = vh[-1, :].view(4, 1)
            X_init = X_init / X_init[3, 0]

            # Re-project
            (
                e_reproj_track,
                mask_track,
                x_proj_track,
            ) = (
                [],
                [],
                [],
            )
            for idx_obs in range(u_centers_hat.size(1)):
                x_proj = torch.matmul(
                    projection_matrices[idx_track, idx_obs, :, :], X_init
                )
                x_proj = x_proj / x_proj[2, 0]
                x_proj_track.append(x_proj[:2, :].detach().view(1, 1, 2))
                err = torch.linalg.norm(
                    x_proj[:2, 0].view(1, 2).detach()
                    - u_centers_hat[idx_track, idx_obs, :].view(1, 2),
                    dim=1,
                )
                e_reproj_track.append(err.view(1, 1))
                mask_track.append((err < self.threshold).view(1, 1))
            e_reproj.append(torch.cat(e_reproj_track, dim=1))
            u_centers_reproj.append(torch.cat(x_proj_track, dim=1))

            mask_track = torch.cat(mask_track, dim=1)
            # if X_init[2, 0] < 0 or s[-1] > 20:
            # if s[-1] > 20:
            #     mask_track = torch.zeros_like(mask_track)
            masks.append(mask_track)

        e_reproj = torch.cat(e_reproj, dim=0)
        masks = torch.cat(masks, dim=0).detach()

        e_reproj *= masks

        if training:
            return e_reproj, masks
        else:
            u_centers_reproj = torch.cat(u_centers_reproj, dim=0)
            return e_reproj, masks, u_centers_reproj


class L2Distance(nn.Module):
    def __init__(self):
        super(L2Distance, self).__init__()

    def forward(self, y, y_hat):
        diff = y - y_hat
        diff = diff**2
        return torch.sqrt(torch.sum(diff, dim=list(range(1, len(y.size())))))

def _only_neg_loss(pred, gt):
  gt = torch.pow(1 - gt, 4)
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss
def _gather_feat(feat, ind):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  return feat
def _tranpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat

class FastFocalLoss(nn.Module):
    """
    Focal loss with centers defining positive pixels
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()
        self.only_neg_loss = _only_neg_loss

    def forward(self, out, target, centers):

        B, C, H, W = out.shape
        N = centers.shape[1]

        device = out.device

        # ----------------------------------
        # 1 生成正样本 mask
        # ----------------------------------

        pos_mask = torch.zeros(B, 1, H, W, device=device)

        x = centers[..., 0].long().clamp(0, W - 1)
        y = centers[..., 1].long().clamp(0, H - 1)

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)

        pos_mask[batch_idx, 0, y, x] = 1

        # ----------------------------------
        # 2 负样本 mask
        # ----------------------------------

        neg_mask = 1 - pos_mask

        # ----------------------------------
        # 3 pos loss
        # ----------------------------------

        pos_loss = torch.log(out.clamp(min=1e-6)) * torch.pow(1 - out, 2) * pos_mask

        pos_loss = pos_loss.sum()

        # ----------------------------------
        # 4 neg loss
        # ----------------------------------

        neg_loss = self.only_neg_loss(out, target) * neg_mask

        neg_loss = neg_loss.sum()

        # ----------------------------------
        # 5 normalization
        # ----------------------------------

        num_pos = pos_mask.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


