import dv
import numpy as np
import cv2
from pathlib import Path

aedat_path ="/data/cyt2/deep_tracker/AEDAT4/dvSave-2025_11_18_16_09_51.aedat4"
out_dir = Path("/data/cyt2/deep_tracker/real_data/traffic_14/")
images_dir = out_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

events_list = []
img_timestamps = []

with dv.AedatFile(aedat_path) as f:

    # ---------- 事件 ----------
    for ev in f["events"]:
        # ev 已经是单个事件对象
        events_list.append((ev.timestamp * 1e-6, int(ev.x), int(ev.y), int(ev.polarity)))

    # ---------- 帧 ----------
    frame_idx = 0
    for frame in f["frames"]:
        img = frame.image  # frame 已经是 numpy array
        img_path = images_dir / f"frame_{frame_idx:08d}.png"
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        cv2.imwrite(str(img_path), img)
        img_timestamps.append(frame.timestamp * 1e-6 if hasattr(frame, "timestamp") else 0)
        frame_idx += 1

# ---------- 保存事件和帧时间戳 ----------
events_arr = np.array(events_list)
np.savetxt(out_dir / "events.txt", events_arr, fmt=["%.9f", "%d", "%d", "%d"])
np.savetxt(out_dir / "images.txt", np.array(img_timestamps), fmt="%.9f")

print(f"完成转换，事件数: {len(events_list)}, 图像帧数: {frame_idx}")
