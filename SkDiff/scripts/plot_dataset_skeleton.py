import sys
import os
import argparse
import random
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from utils.misc import load_config, seed_all
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset

def plot_3d_skeleton(joints_list, bones, titles, caption, output_path):
    """
    joints:   (N, 3) 关节坐标
    bones:    (M, 2) 骨骼连接关节索引
    """
    num_skeletons = len(joints_list)
    fig = plt.figure(figsize=(5*num_skeletons, 5), dpi=200)

    for i, joints in enumerate(joints_list):
        ax = fig.add_subplot(1, num_skeletons, i+1, projection='3d')

        # 绘制骨骼
        for bone in bones:
            p1, p2 = joints[bone[0]], joints[bone[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='gray', linewidth=1)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                   c='r', s=30, edgecolors='k', alpha=0.9)

        # 坐标轴范围（根据当前骨架）
        min_val = joints.min()
        max_val = joints.max()
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_zlim(min_val, max_val)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(titles[i] if titles else f"Skeleton {i+1}")

    plt.suptitle(caption)
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    split = "val"
    output_dir = Path(f"/root/autodl-fs-data3/gnndiff/vis/{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = EasyDict({
        "train_file": "/root/autodl-fs-data3/Articulation-XL2.0/train_meta.json",
        "val_file": "/root/autodl-fs-data3/Articulation-XL2.0/test_meta.json",
        "clip_emb_root": None
    })

    # === Data ===
    featurizer = FeaturizeGraph(use_rotate=False, use_perturb=False, perturb_prob=None)
    transform = Compose([
        featurizer,
    ])

    val_dataset = get_dataset(config, transform, f"{split}")
    val_dataset.random_view = False
    val_dataset.random_flip = False
    print(f"Val dataset size: {len(val_dataset)}")

    seed_all(42)
    input_entries = []
    indices = sorted(random.sample(range(len(val_dataset)), 100))

    for i, idx in enumerate(indices):
        data = val_dataset[idx]
        file_path = data.rig_path
        caption = data.caption
        category = data.category

        joints = data.node_pos.cpu().numpy()
        bond_index = data.bond_index.cpu().numpy()
        bones = []
        for j in range(bond_index.shape[1] // 2):
            bones.append((bond_index[0, j * 2], bond_index[0, j * 2 + 1]))

        uuid = os.path.basename(file_path).split('.')[0]
        output_path = output_dir / category / f"{uuid}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_entries.append([[joints], bones, ["None"], caption, output_path])
    
    featurizer.use_perturb = True
    for i, idx in enumerate(indices):
        data = val_dataset[idx]
        joints = data.node_pos.cpu().numpy()
        input_entries[i][0].append(joints)
        input_entries[i][2].append("Perturbed")

    featurizer.use_perturb = False
    featurizer.use_rotate = True
    for i, idx in enumerate(indices):
        data = val_dataset[idx]
        joints = data.node_pos.cpu().numpy()
        input_entries[i][0].append(joints)
        input_entries[i][2].append("Rotated")
    
    featurizer.use_perturb = True
    featurizer.use_rotate = True
    for i, idx in enumerate(indices):
        data = val_dataset[idx]
        joints = data.node_pos.cpu().numpy()
        input_entries[i][0].append(joints)
        input_entries[i][2].append("Perturbed + Rotated")
    
    for joints_list, bones, title, caption, output_path in input_entries:
        plot_3d_skeleton(joints_list, bones, title, caption, output_path)
        print(f"Saved skeleton plot to: {output_path}")
            