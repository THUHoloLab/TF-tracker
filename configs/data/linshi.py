import numpy as np

data = np.load("/data/cyt2/ETAP/data_pipeline/sample/00000040/annotations.npy", allow_pickle=True)
'''t = np.array(data['t'])
t = t / 1e9
events = data.files
for name in data.files:
    print(data[name])'''
data = data.item()
target_points = data['target_points']
sz=target_points.shape

print(0)

'''
from pathlib import Path
import numpy as np

def resave_images_txt(root_dir, fmt="%.9f"):
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    processed = 0

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        images_txt = subdir / "images.txt"
        if not images_txt.exists():
            continue

        try:
            data = np.loadtxt(images_txt)
        except Exception as e:
            print(f"❌ Failed to read {images_txt}: {e}")
            continue

        # 覆盖保存
        np.savetxt(images_txt, data, fmt=fmt)
        processed += 1

        print(f"✅ Resaved: {images_txt}")

    print(f"\nDone. Total processed folders: {processed}")


if __name__ == "__main__":
    # ===== 修改为你的路径 =====
    root_dir = "/data/cyt2/TF_tracker/datasets/train/syn/"

    resave_images_txt(root_dir, fmt="%.9f")
'''
