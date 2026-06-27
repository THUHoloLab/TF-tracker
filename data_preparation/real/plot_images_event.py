import os
import re
import glob
import numpy as np
from PIL import Image
import imageio


def natural_key(path):
    """
    按文件名中的数字顺序排序，例如:
    1.png, 2.png, 10.png
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    parts = re.split(r'(\d+)', stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_images_from_folder(images_dir):
    """
    读取 images 文件夹中的图片，并按文件名序号排序
    """
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    image_paths = sorted(image_paths, key=natural_key)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"在 {images_dir} 中没有找到图片。")

    return image_paths


def load_frame_times(images_txt_path):
    """
    读取 images.txt，要求只有一列时间
    """
    times = np.loadtxt(images_txt_path)
    times = np.asarray(times).reshape(-1)
    return times


def load_events(events_path):
    """
    读取 events 文件，要求4列:
    t, x, y, p
    """
    events = np.loadtxt(events_path)
    if events.ndim == 1:
        events = events[None, :]
    if events.shape[1] != 4:
        raise ValueError("events 文件必须包含4列: [t, x, y, p]")
    return events


def render_event_image(events_slice, height, width):
    """
    将某个时间区间内的事件渲染成一张完整事件图

    正事件 p=1 -> 红色
    负事件 p=0 -> 绿色
    """
    event_img = np.zeros((height, width, 3), dtype=np.uint8)

    if len(events_slice) == 0:
        return event_img

    x = events_slice[:, 1].astype(np.int32)
    y = events_slice[:, 2].astype(np.int32)
    p = events_slice[:, 3].astype(np.int32)

    # 过滤越界点
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    p = p[valid]

    pos_count = np.zeros((height, width), dtype=np.int32)
    neg_count = np.zeros((height, width), dtype=np.int32)

    for xi, yi, pi in zip(x, y, p):
        if pi == 1:
            pos_count[yi, xi] += 1
        else:
            neg_count[yi, xi] += 1

    if pos_count.max() > 0:
        pos_vis = (pos_count / pos_count.max() * 255).astype(np.uint8)
    else:
        pos_vis = np.zeros_like(pos_count, dtype=np.uint8)

    if neg_count.max() > 0:
        neg_vis = (neg_count / neg_count.max() * 255).astype(np.uint8)
    else:
        neg_vis = np.zeros_like(neg_count, dtype=np.uint8)

    # 红色表示正事件
    event_img[:, :, 1] = pos_vis
    # 绿色表示负事件
    event_img[:, :, 1] = neg_vis

    return event_img


def merge_half_image_and_half_event(img_np, event_img):
    """
    输出宽度 = 原图宽度
    左半边：image 的左半边
    右半边：event_img 的右半边
    """
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    if event_img.ndim == 2:
        event_img = np.stack([event_img] * 3, axis=-1)

    h, w, _ = img_np.shape
    half_w = w // 2

    # 如果事件图尺寸不同，则调整到和原图一致
    if event_img.shape[:2] != img_np.shape[:2]:
        event_img = np.array(
            Image.fromarray(event_img).resize((w, h), Image.BILINEAR)
        )

    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 左半边：图像左半边
    frame[:, 0:half_w, :] = img_np[:, 0:half_w, :]

    # 右半边：事件图右半边
    frame[:, half_w:w, :] = event_img[:, half_w:w, :]

    return frame


def make_gif(images_dir, images_txt_path, events_path, save_gif_path, duration=0.2):
    """
    生成GIF:
    每一帧 = [左半边 image | 右半边 events]
    """
    image_paths = load_images_from_folder(images_dir)
    frame_times = load_frame_times(images_txt_path)
    events = load_events(events_path)

    if len(image_paths) != len(frame_times):
        raise ValueError(
            f"图片数量 ({len(image_paths)}) 与 images.txt 时间数量 ({len(frame_times)}) 不一致。"
        )

    gif_frames = []

    # 相邻两帧时间区间 [t_i, t_{i+1})
    for i in range(len(image_paths) - 1):
        t0 = frame_times[i]
        t1 = frame_times[i + 1]

        # 读取当前左侧图像
        img = Image.open(image_paths[i]).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # 累积当前时间区间内事件
        mask = (events[:, 0] >= t0) & (events[:, 0] < t1)
        events_slice = events[mask]

        # 渲染完整事件图
        event_img = render_event_image(events_slice, h, w)

        # 拼接：左半边图像 + 右半边事件图
        frame = merge_half_image_and_half_event(img_np, event_img)

        gif_frames.append(frame)

        print(
            f"Processed frame {i + 1}/{len(image_paths) - 1}, "
            f"time=[{t0}, {t1}), events={len(events_slice)}"
        )

    # 保存 GIF
    imageio.mimsave(save_gif_path, gif_frames, duration=duration, loop=0)
    print(f"\nGIF saved to: {save_gif_path}")


if __name__ == "__main__":
    # ==========================
    # 修改为你的实际路径
    # ==========================
    images_dir = "/data/cyt2/TF_tracker/datasets/ball/ball_7/images/"
    images_txt_path = "/data/cyt2/TF_tracker/datasets/ball/ball_7/images.txt"
    events_path = "/data/cyt2/TF_tracker/datasets/ball/ball_7/events.txt"
    save_gif_path = "/data/cyt2/TF_tracker/datasets/ball/ball_7/image_event.gif"

    # 每帧持续时间（秒）
    duration = 0.2

    make_gif(
        images_dir=images_dir,
        images_txt_path=images_txt_path,
        events_path=events_path,
        save_gif_path=save_gif_path,
        duration=duration,
    )