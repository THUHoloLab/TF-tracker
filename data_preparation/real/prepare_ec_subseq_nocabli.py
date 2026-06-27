"""
Prepare data for a subset of an Event Camera Dataset sequence
- Create an output directory with images, event txt, and time surfaces
(Without calibration or groundtruth)
"""
import os
import shutil
from glob import glob
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from tqdm import tqdm

from utils.utils import blosc_opts


def prepare_data(root_dir, sequence_name, start_idx, end_idx):
    sequence_dir = Path(root_dir) / sequence_name
    if not sequence_dir.exists():
        print(f"Sequence directory does not exist for {sequence_name}")
        exit()

    # ---------- 创建输出文件夹 ----------
    subseq_dir = Path(root_dir) / f"{sequence_name}_{start_idx}_{end_idx}"
    subseq_dir.mkdir(exist_ok=True)

    # ---------- 图像处理 ----------
    images_dir = sequence_dir / "images"
    subseq_images_dir = subseq_dir / "images"
    subseq_images_dir.mkdir(exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        img_path = images_dir / f"frame_{str(i).zfill(8)}.png"
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping.")
            continue
        shutil.copy(
            str(img_path),
            str(subseq_images_dir / f"frame_{str(i - start_idx).zfill(8)}.png"),
        )

    # ---------- 读取图像尺寸 ----------
    first_img_path = sorted(list(images_dir.glob("*.png")))[0]
    IMG_H, IMG_W = cv2.imread(str(first_img_path), cv2.IMREAD_GRAYSCALE).shape

    # ---------- 读取图像时间戳 ----------
    image_timestamps = np.genfromtxt(sequence_dir / "images.txt", usecols=[0])
    image_timestamps = image_timestamps[start_idx : end_idx + 1]
    np.savetxt(str(subseq_dir / "images.txt"), image_timestamps)
    print(
        f"Image timestamps range: [{image_timestamps[0]}, {image_timestamps[-1]}]"
    )

    # ---------- 读取事件 ----------
    events = read_csv(
        str(sequence_dir / "events.txt"), header=None, delimiter=" "
    ).to_numpy()
    t_events = events[:, 0]
    print(f"Event timestamps range: [{t_events.min()}, {t_events.max()}]")

    # Generate debug frames
    debug_dir = subseq_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)
    n_frames_debug = len(image_timestamps)
    dt = 0.01
    for i in range(n_frames_debug):
        # Events
        t1 = image_timestamps[i]
        t0 = t1 - dt
        time_mask = np.logical_and(events[:, 0] >= t0, events[:, 0] < t1)
        events_slice = events[time_mask, :]

        on_mask = events_slice[:, 3] == 1
        off_mask = events_slice[:, 3] == 0
        events_slice_on = events_slice[on_mask, :]
        events_slice_off = events_slice[off_mask, :]

        # Image
        img = cv2.imread(
            str(images_dir / f"frame_{str(start_idx+i).zfill(8)}.png"), cv2.IMREAD_GRAYSCALE
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img, cmap="gray")
        ax.scatter(events_slice_on[:, 1], events_slice_on[:, 2], s=5, c="green")
        ax.scatter(events_slice_off[:, 1], events_slice_off[:, 2], s=5, c="red")
        plt.show()
        fig.savefig(str(debug_dir / f"frame_{str(i).zfill(8)}.png"))
        plt.close(fig)

    # ---------- 生成时间表 ----------
    for dt in [0.01, 0.02]:
        for n_bins in [1, 5]:
            dt_bin = dt / n_bins
            output_ts_dir = (
                subseq_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_{n_bins}"
            )
            output_ts_dir.mkdir(parents=True, exist_ok=True)

            debug_dir = subseq_dir / f"debug_events_{n_bins}"
            debug_dir.mkdir(exist_ok=True)

            print(f"Generating time surfaces for dt={dt}, n_bins={n_bins}...")
            for i, t1 in tqdm(
                enumerate(np.arange(image_timestamps[0], image_timestamps[-1] + dt, dt)),
                total=int((image_timestamps[-1] - image_timestamps[0]) / dt),
            ):
                output_ts_path = (
                    output_ts_dir / f"{str(int(i * (dt * 1e6))).zfill(7)}.h5"
                )
                if output_ts_path.exists():
                    continue

                time_surface = np.zeros((IMG_H, IMG_W, 2 * n_bins), dtype=np.float64)
                t0 = t1 - dt

                for i_bin in range(n_bins):
                    t0_bin = t0 + i_bin * dt_bin
                    t1_bin = t0_bin + dt_bin

                    time_mask = np.logical_and(
                        events[:, 0] >= t0_bin, events[:, 0] < t1_bin
                    )
                    events_slice = events[time_mask, :]

                    for e in events_slice:
                        y, x, p = int(e[2]), int(e[1]), int(e[3])
                        if 0 <= x < IMG_W and 0 <= y < IMG_H:
                            time_surface[y, x, 2 * i_bin + p] = e[0] - t0

                time_surface = np.divide(time_surface, dt)

                with h5py.File(output_ts_path, "w") as h5f_out:
                    h5f_out.create_dataset(
                        "time_surface",
                        data=time_surface,
                        shape=time_surface.shape,
                        dtype=np.float32,
                        **blosc_opts(complevel=1, shuffle="byte"),
                    )

                # 可视化时间表
                debug_event_frame = (np.any(time_surface > 0, axis=2).astype(np.uint8) * 255)
                #debug_event_frame = ((time_surface[:, :, 0] > 0) * 255).astype(np.uint8)
                cv2.imwrite(
                    str(debug_dir / f"{str(int(i * dt * 1e6)).zfill(7)}.png"),
                    debug_event_frame,
                )

    print(f"Done: subsequence saved to {subseq_dir}")


if __name__ == "__main__":
    root_dir = "/data/cyt2/deep_tracker/real_data/"  # 数据根目录
    sequence_name = "traffic_6"  # 序列名
    start_idx = 2  # 起始帧索引
    end_idx = 29  # 结束帧索引

    # ---------- 调用函数 ----------
    prepare_data(root_dir, sequence_name, start_idx, end_idx)

