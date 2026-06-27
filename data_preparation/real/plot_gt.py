import os
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio


def load_gt(gt_path):
    """
    读取 gt.txt
    第一列：轨迹/质心索引号
    第二列：时刻
    第三列：x 坐标
    第四列：y 坐标
    """
    data = np.loadtxt(gt_path)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 4:
        raise ValueError("gt.txt 至少需要 4 列：index, time, x, y")

    track_id = data[:, 0].astype(np.int64)
    time = data[:, 1].astype(np.float64)
    x = data[:, 2].astype(np.float64)
    y = data[:, 3].astype(np.float64)

    return track_id, time, x, y


def get_color(track_id):
    """
    根据轨迹索引生成固定颜色。
    """
    colors = [
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 128, 255),    # blue
        (255, 128, 0),    # orange
        (255, 0, 255),    # magenta
        (0, 255, 255),    # cyan
        (255, 255, 0),    # yellow
        (128, 0, 255),    # purple
        (255, 128, 128),
        (128, 255, 128),
    ]
    return colors[int(track_id) % len(colors)]


def draw_one_frame(
    track_id,
    time,
    x,
    y,
    t_start,
    t_end,
    width,
    height,
    background_color=(0, 0, 0),
    line_width=3,
    point_radius=4,
):
    """
    绘制一个时间窗口 [t_start, t_end) 内的轨迹。
    """
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)

    unique_ids = np.unique(track_id)

    for tid in unique_ids:
        mask = (
            (track_id == tid) &
            (time >= t_start) &
            (time < t_end)
        )

        if not np.any(mask):
            continue

        points_time = time[mask]
        points_x = x[mask]
        points_y = y[mask]

        # 按时间排序
        order = np.argsort(points_time)
        points_x = points_x[order]
        points_y = points_y[order]

        # 过滤越界点
        valid = (
            (points_x >= 0) & (points_x < width) &
            (points_y >= 0) & (points_y < height)
        )

        points_x = points_x[valid]
        points_y = points_y[valid]

        if len(points_x) == 0:
            continue

        color = get_color(tid)

        points = [
            (int(round(px)), int(round(py)))
            for px, py in zip(points_x, points_y)
        ]

        # 如果该时间窗口内有多个点，画轨迹线
        if len(points) >= 2:
            draw.line(points, fill=color, width=line_width)

        # 画轨迹点
        for px, py in points:
            draw.ellipse(
                (
                    px - point_radius,
                    py - point_radius,
                    px + point_radius,
                    py + point_radius,
                ),
                fill=color,
            )

        # 当前时间窗口内最后一个点稍微画大一点
        px, py = points[-1]
        draw.ellipse(
            (
                px - point_radius * 2,
                py - point_radius * 2,
                px + point_radius * 2,
                py + point_radius * 2,
            ),
            outline=(255, 255, 255),
            width=2,
        )

    return np.array(img)


def make_trajectory_gif(
    gt_path,
    save_gif_path,
    width,
    height,
    window_time=0.2,
    step_time=0.2,
    gif_duration=0.2,
    background_color=(0, 0, 0),
    line_width=3,
    point_radius=4,
    loop=0,
):
    """
    根据 gt.txt 绘制轨迹 GIF。

    window_time:
        每帧显示的轨迹时间长度，例如 0.2 秒。

    step_time:
        相邻 GIF 帧之间的时间步长。
        如果 step_time = window_time，则是不重叠的时间窗口；
        如果 step_time < window_time，则是滑动窗口。

    gif_duration:
        GIF 中每一帧播放的时长，单位为秒。

    loop:
        0 表示无限循环。
    """
    track_id, time, x, y = load_gt(gt_path)

    t_min = np.min(time)
    t_max = np.max(time)

    frames = []

    frame_starts = np.arange(t_min, t_max, step_time)

    print(f"Time range: {t_min:.6f} ~ {t_max:.6f}")
    print(f"Total frames: {len(frame_starts)}")

    for i, t_start in enumerate(frame_starts):
        t_end = t_start + window_time

        frame = draw_one_frame(
            track_id=track_id,
            time=time,
            x=x,
            y=y,
            t_start=t_start,
            t_end=t_end,
            width=width,
            height=height,
            background_color=background_color,
            line_width=line_width,
            point_radius=point_radius,
        )

        frames.append(frame)

        print(
            f"[{i + 1}/{len(frame_starts)}] "
            f"t = [{t_start:.3f}, {t_end:.3f})"
        )

    os.makedirs(os.path.dirname(save_gif_path), exist_ok=True)

    imageio.mimsave(
        save_gif_path,
        frames,
        duration=gif_duration,
        loop=loop,
    )

    print(f"GIF saved to: {save_gif_path}")


if __name__ == "__main__":
    gt_path = "/data/cyt2/TF_tracker/datasets/traffic_data_subseq_deep/traffic_5_0_14/tracks/traffic_5_0_14.gt.txt"
    save_gif_path = "/data/cyt2/TF_tracker/datasets/traffic_data_subseq_deep/traffic_5_0_14/trajectory.gif"

    # 指定图像宽度和高度
    width = 346 #240
    height = 260 #180

    make_trajectory_gif(
        gt_path=gt_path,
        save_gif_path=save_gif_path,
        width=width,
        height=height,

        # 每帧显示 0.2 秒的轨迹
        window_time=0.2,

        # 每隔 0.2 秒生成一帧
        step_time=0.2,

        # GIF 播放时每帧显示 0.2 秒
        gif_duration=0.2,

        # 黑色背景
        background_color=(0, 0, 0),

        # 轨迹线宽
        line_width=3,

        # 点半径
        point_radius=4,

        # 0 表示 GIF 无限循环
        loop=0,
    )