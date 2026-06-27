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

def plot_event_stream_3d(root_dir, sequence_name, start_idx, end_idx, dt=0.01):
    """
    3D 可视化事件流 (x=time/frame, y=pixel_y, z=pixel_x)

    :param events: Nx4 array, columns=[t, x, y, polarity]
    :param image_timestamps: 图像时间戳列表
    :param images_dir: 图像文件夹路径
    :param start_idx: 起始帧编号
    :param end_idx: 结束帧编号
    :param show_image_first: 第一帧显示图像背景
    :param dt: 每个切片的时间窗口
    """
    sequence_dir = Path(root_dir) / sequence_name
    if not sequence_dir.exists():
        print(f"Sequence directory does not exist for {sequence_name}")
        exit()

    # ---------- 创建输出文件夹 ----------
    subseq_dir = Path(root_dir) / f"{sequence_name}_{start_idx}_{end_idx}"

    # ---------- 图像处理 ----------
    images_dir = sequence_dir / "images"

    # ---------- 读取图像尺寸 ----------
    first_img_path = sorted(list(images_dir.glob("*.png")))[start_idx]

    # ---------- 读取图像时间戳 ----------
    image_timestamps = np.genfromtxt(sequence_dir / "images.txt", usecols=[0])
    image_timestamps = image_timestamps[start_idx: end_idx + 1]
    print(
        f"Image timestamps range: [{image_timestamps[0]}, {image_timestamps[-1]}]"
    )

    # ---------- 读取事件 ----------
    events = read_csv(
        str(sequence_dir / "events.txt"), header=None, delimiter=" "
    ).to_numpy()
    t_events = events[:, 0]
    print(f"Event timestamps range: [{t_events.min()}, {t_events.max()}]")


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_zaxis()

    n_frames = len(image_timestamps)

    for i in range(n_frames):
        # 当前帧的时间范围
        t1 = image_timestamps[i]
        t0 = t1 - dt
        mask = np.logical_and(events[:, 0] >= t0, events[:, 0] < t1)
        events_slice = events[mask, :]   # shape: (N_i, 4)
        on_mask = events_slice[:, 3] == 1
        off_mask = events_slice[:, 3] == 0
        events_slice_on = events_slice[on_mask, :]
        events_slice_off = events_slice[off_mask, :]

        # 第一张需要叠加图像
        if i == 0:
            img = cv2.imread(str(first_img_path), cv2.IMREAD_GRAYSCALE)
            H, W = img.shape if img.ndim == 2 else img.shape[:2]

            # 生成（x=0）切片上的图像纹理
            zz, yy = np.meshgrid(np.arange(H), np.arange(W))
            yy = np.transpose(yy)
            zz = np.transpose(zz)
            ax.plot_surface(
                np.zeros_like(yy), yy, zz,
                rstride=1, cstride=1,
                facecolors=plt.cm.gray(img / 255.0),
                shade=False, alpha=0.8
            )

        # 绘制事件点
        time_slice = np.ones_like(events_slice_on[:, 1]) * i  # x = 帧编号
        ax.scatter(time_slice,events_slice_on[:, 1], events_slice_on[:, 2], s=5, c="green")
        time_slice = np.ones_like(events_slice_off[:, 1]) * i  # x = 帧编号
        ax.scatter(time_slice,events_slice_off[:, 1], events_slice_off[:, 2], s=5, c="red")


    '''ax.set_xlabel("Time / Frame Index")
    ax.set_ylabel("Pixel Y")
    ax.set_zlabel("Pixel X")

    ax.set_title("3D Event Stream Visualization")'''

    '''plt.tight_layout()
    plt.show()'''
    save_path = subseq_dir / "3d_event.png"
    plt.savefig(str(save_path), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    root_dir = "/data/cyt2/deep_tracker/real_data/"  # 数据根目录
    sequence_name = "traffic_6"  # 序列名
    start_idx = 2  # 起始帧索引
    end_idx = 29  # 结束帧索引

    # ---------- 调用函数 ----------
    plot_event_stream_3d(root_dir, sequence_name, start_idx, end_idx)
