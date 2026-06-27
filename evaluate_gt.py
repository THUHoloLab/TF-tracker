"""
Predict tracks for a sequence with a network
Add Feature Age and Expected FA metrics
Use gt.txt time=0 points as initial keypoints
"""
import logging
import os
from pathlib import Path

import cv2
import hydra
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict
from prettytable import PrettyTable
from tqdm import tqdm

from utils.dataset import CornerConfig, ECSubseq_nogt, EDSSubseq, EvalDatasetType, EDSSubseq_nogt
from utils.timers import CudaTimer, cuda_timers
from utils.track_utils import TrackObserver, compute_tracking_errors, read_txt_results
from utils.visualization import generate_track_colors, render_pred_tracks

# Configure GPU order
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Logging
logger = logging.getLogger(__name__)
results_table = PrettyTable()
results_table.field_names = ["Inference Time", "Feature Age@5", "Expected FA"]

# Configure datasets
corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

EvalDatasetConfigDict = {
    EvalDatasetType.EC: {"dt": 0.010, "root_dir": "/data/cyt2/TF_tracker/datasets/test/"},
    EvalDatasetType.EDS: {"dt": 0.005, "root_dir": "/data/cyt2/TF_tracker/datasets/test_EDS/"},
}

EVAL_DATASETS = [
    ("shapes_translation_8_88", EvalDatasetType.EC),
]

# GT root directory:
# /data/cyt2/TF_tracker/datasets/test/shapes_6dof_485_565/tracks/gt.txt
GT_ROOT_DIR = Path("/data/cyt2/TF_tracker/datasets/test/")

ERROR_THRESHOLD_RANGE = np.arange(1, 32, 1)

img_H=180
img_W=240


def load_gt_start_corners(gt_path: Path):
    """
    gt.txt format:
        track_id  time  x  y

    Take rows whose original time is the minimum time in the file
    as the initial keypoints.
    Sort by track_id to keep a stable order.
    """
    gt_data = np.loadtxt(str(gt_path))
    if gt_data.ndim == 1:
        gt_data = gt_data[None, :]

    # Keep original timestamps, do not normalize
    t0 = np.min(gt_data[:, 1])

    # Take rows at the first timestamp in the source file
    #init_rows = gt_data[np.isclose(gt_data[:, 1], t0)]
    init_rows = gt_data[gt_data[:, 1] == t0]
    if init_rows.size == 0:
        raise ValueError(f"No initial-time rows found in GT file: {gt_path}")

    init_rows = init_rows[np.argsort(init_rows[:, 0])]
    gt_start_corners = init_rows[:, 2:4].astype(np.int32)
    return gt_start_corners


def compute_feature_age(track_data_pred, track_data_gt, error_threshold=5, asynchronous=False):
    """
    Feature Age@threshold:
    mean of non-zero fa_rel under a fixed error threshold.
    """
    fa_rel, _ = compute_tracking_errors(
        track_data_pred,
        track_data_gt,
        error_threshold=error_threshold,
        asynchronous=asynchronous,
    )

    #fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
    fa_rel_nz = fa_rel
    if len(fa_rel_nz) == 0:
        return 0.0

    return float(np.mean(fa_rel_nz))


def compute_expected_fa(track_data_pred, track_data_gt, asynchronous=False):
    """
    Expected FA:
    mean over thresholds 1..31 of:
        inlier_ratio(threshold) * mean(nonzero fa_rel(threshold))
    """
    inlier_ratio_arr, fa_rel_nz_arr = [], []

    for thresh in ERROR_THRESHOLD_RANGE:
        fa_rel, _ = compute_tracking_errors(
            track_data_pred,
            track_data_gt,
            error_threshold=thresh,
            asynchronous=asynchronous,
        )

        inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)

        '''if inlier_ratio > 0:
            fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
        else:
            fa_rel_nz = np.array([0.0])'''
        fa_rel_nz = fa_rel
        inlier_ratio_arr.append(inlier_ratio)
        fa_rel_nz_arr.append(np.mean(fa_rel_nz))

    expected_fa = np.mean(np.array(inlier_ratio_arr) * np.array(fa_rel_nz_arr))
    return float(expected_fa)


def evaluate(model, sequence_dataset, dt_track_vis, sequence_name, visualize, save_frames, gt_path):
    tracks_pred = TrackObserver(
        t_init=sequence_dataset.t_init, u_centers_init=sequence_dataset.u_centers
    )

    model.reset(sequence_dataset.n_tracks)
    event_generator = sequence_dataset.events()
    cuda_timer = CudaTimer(model.device, sequence_dataset.sequence_name)
    RESULTS_DIR = Path("/data/cyt2/TF_tracker/results/test/EC/")

    heatmap_init = sequence_dataset.heatmap.clone().to(model.device)
    heatmap_pre = heatmap_init.clone()
    heatmap_pre_pre = heatmap_init.clone()
    # 初始热图就是初始质心图，在质心位置为1，其余位置为0

    with torch.no_grad():
        # Predict network tracks
        for i_event, (t, x) in enumerate(
                tqdm(
                    event_generator,
                    total=sequence_dataset.n_events - 1,
                    desc="Predicting tracks with network...",
                )
        ):
            with cuda_timer:
                x = x.to(model.device)
                heatmap = torch.cat((heatmap_pre_pre, heatmap_pre, heatmap_init), dim=1)
                y_hat, heatmap = model.forward(x, heatmap, i_event)

                y_hat = y_hat.squeeze(0)  # [2, H, W]
                coords = sequence_dataset.u_centers.long()  # [N, 2]
                x = coords[:, 0]  # 列
                y = coords[:, 1]  # 行
                H, W = y_hat.shape[1], y_hat.shape[2]
                x = x.clamp(0, W - 1)
                y = y.clamp(0, H - 1)
                values = y_hat[:, y, x]  # [2, N]
                values = values.permute(1, 0)
                sequence_dataset.accumulate_y_hat(values)
                sequence_dataset.update_heatmap(heatmap)
                heatmap_pre_pre = heatmap_pre
                heatmap_pre = sequence_dataset.heatmap.to(model.device)
            tracks_pred.add_observation(t, sequence_dataset.u_centers.cpu().numpy())

        frames_dir = f"{sequence_name}_frames"
        if save_frames and not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        if visualize:
            gif_img_arr = []
            tracks_pred_interp = tracks_pred.get_interpolators()
            track_colors = generate_track_colors(sequence_dataset.n_tracks)
            for i, (t, img_now) in enumerate(
                tqdm(
                    sequence_dataset.frames(),
                    total=sequence_dataset.n_frames - 1,
                    desc="Rendering predicted tracks... ",
                )
            ):
                fig_arr = render_pred_tracks(
                    tracks_pred_interp, t, img_now, track_colors, dt_track=dt_track_vis
                )
                gif_img_arr.append(fig_arr)
                if save_frames:
                    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    imageio.imsave(frame_path, fig_arr)
            gif_path = RESULTS_DIR / f"{sequence_name}_tracks_pred.gif"
            imageio.mimsave(str(gif_path), gif_img_arr, fps=10, loop=0)

    # Save predicted tracks
    tracks_path = RESULTS_DIR / f"{sequence_name}_pred_tracks.txt"
    track_data_pred = np.asarray(tracks_pred.track_data, dtype=np.float64)
    track_data_pred = np.nan_to_num(track_data_pred, nan=-1.0)

    with open(tracks_path, "w") as f:
        np.savetxt(
            f,
            track_data_pred,
            fmt=["%i", "%.9f", "%i", "%i"],
            delimiter=" ",
        )

    # =========================
    # Metrics: Feature Age / Expected FA
    # =========================
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    track_data_gt = read_txt_results(str(gt_path))
    track_data_pred = read_txt_results(tracks_path)

    # Align timestamps if needed
    if track_data_pred[0, 1] != track_data_gt[0, 1]:
        track_data_pred = track_data_pred.copy()
        track_data_pred[:, 1] += -track_data_pred[0, 1] + track_data_gt[0, 1]

    feature_age = compute_feature_age(
        track_data_pred,
        track_data_gt,
        error_threshold=5,
        asynchronous=False,
    )
    expected_fa = compute_expected_fa(
        track_data_pred,
        track_data_gt,
        asynchronous=False,
    )

    metrics = {}
    metrics["latency"] = sum(cuda_timers[sequence_dataset.sequence_name])
    metrics["feature_age"] = feature_age
    metrics["expected_fa"] = expected_fa
    return metrics


@hydra.main(config_path="configs", config_name="eval_real_defaults")
def track(cfg):
    pl.seed_everything(1234)
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model.representation = cfg.representation
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Configure model
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    state_dict = torch.load(cfg.weights_path, map_location="cuda:0")["state_dict"]
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Run evaluation on each dataset
    for seq_name, seq_type in EVAL_DATASETS:
        if seq_type == EvalDatasetType.EC:
            dataset_class = ECSubseq_nogt
        elif seq_type == EvalDatasetType.EDS:
            dataset_class = EDSSubseq_nogt
        else:
            raise ValueError

        dataset = dataset_class(
            EvalDatasetConfigDict[seq_type]["root_dir"],
            seq_name,
            -1,
            cfg.patch_size,
            cfg.representation,
            EvalDatasetConfigDict[seq_type]["dt"],
            corner_config,
        )

        # Use GT time=0 points as initial keypoints
        gt_path = GT_ROOT_DIR / seq_name / "tracks" / f"{seq_name}.gt.txt"
        gt_start_corners = load_gt_start_corners(gt_path)
        dataset.override_keypoints(gt_start_corners)
        dataset.override_heatmap(gt_start_corners,height=img_H,width=img_W)

        metrics = evaluate(
            model,
            dataset,
            cfg.dt_track_vis,
            seq_name,
            cfg.visualize,
            cfg.save_frames,
            gt_path
        )

        logger.info(f"=== DATASET: {seq_name} ===")
        logger.info(f"Latency: {metrics['latency']} s")
        logger.info(f"Feature Age@5: {metrics['feature_age']}")
        logger.info(f"Expected FA: {metrics['expected_fa']}")

        results_table.add_row([
            metrics["latency"],
            metrics["feature_age"],
            metrics["expected_fa"],
        ])

    logger.info(f"\n{results_table.get_string()}")


if __name__ == "__main__":
    track()