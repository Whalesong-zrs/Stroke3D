import os
import time
import random
import logging
import torch
import numpy as np
import yaml
from easydict import EasyDict
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from tqdm import tqdm 

# Load config
def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

# Set seed
def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

# Log dir
def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Logger
def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Torchify dict
def torchify_dict(info_dict: dict):
    output = {}
    for k, v in info_dict.items():
        if k == 'bond_index':
            output[k] = torch.from_numpy(v).to(torch.long)
        elif isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v).to(torch.float32)
        else:
            output[k] = v
    return output

def edge_index_to_hierarchy(edge_index):
    # edge_index shape: [2, E]
    edge_index = edge_index.cpu().numpy()
    hierarchy = set()

    for i in range(edge_index.shape[1]):
        parent = edge_index[0, i]
        child = edge_index[1, i]
        # 排序成 (min, max) 防止重复
        if parent != child:
            hierarchy.add((min(parent, child), max(parent, child)))

    return list(hierarchy)

# === 函数：生成 joints_2d 字典（用于画图） ===
def build_joint_dict(xy_tensor):
    # xy_tensor: [num_nodes, 2]
    xy = xy_tensor.cpu().numpy()
    joints_2d = {f"joint{i}": xy[i] for i in range(xy.shape[0])}
    return joints_2d

def visualize_skeleton_2d(joints, hierarchy, ax=None, proj_axes=(0, 1), title=None, color='b'):
    """
    joints: dict of {joint_name: [x, y, z]}
    hierarchy: list of (parent_idx, child_idx)
    proj_axes: tuple like (0,1) for xy, (1,2) for yz
    ax: optional matplotlib axis
    color: line color
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    for parent, child in hierarchy:
        name1, name2 = f"joint{parent}", f"joint{child}"
        if name1 in joints and name2 in joints:
            p1 = joints[name1]
            p2 = joints[name2]
            ax.plot([p1[proj_axes[0]], p2[proj_axes[0]]],
                    [p1[proj_axes[1]], p2[proj_axes[1]]],
                    color=color, linewidth=2)

    for name, coord in joints.items():
        ax.scatter(coord[proj_axes[0]], coord[proj_axes[1]], c='r', s=10)
        # ax.text(coord[proj_axes[0]], coord[proj_axes[1]], name, fontsize=6, color='black', ha='right')

    ax.axis('off')

def visualize_skeleton_3d(joints, hierarchy, ax=None, title=None, color='b'):
    """
    joints: dict of {joint_name: [x, y, z]}
    hierarchy: list of (parent_idx, child_idx)
    proj_axes: tuple like (0,1) for xy, (1,2) for yz
    ax: optional matplotlib axis
    color: line color
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    for parent, child in hierarchy:
        name1, name2 = f"joint{parent}", f"joint{child}"
        if name1 in joints and name2 in joints:
            p1 = joints[name1]
            p2 = joints[name2]
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, linewidth=2)

    coords = np.array(list(joints.values()))
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='r', s=10)

    min_val = coords.min()
    max_val = coords.max()
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if title:
        ax.set_title(title, fontsize=10)

def look_at(eye, center, up):
    """Create a look-at (view) matrix."""
    f = np.array(center, dtype=np.float32) - np.array(eye, dtype=np.float32)
    f /= np.linalg.norm(f)

    u = np.array(up, dtype=np.float32)
    u /= np.linalg.norm(u)

    s = np.cross(f, u)
    u = np.cross(s, f)

    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -np.matmul(m[:3, :3], np.array(eye, dtype=np.float32))

    return m

def process_single_model_with_projections(loader, renderer, mesh_path, rig_path, view_params, ax, title):
    loader.load_rig_data(rig_path)
    loader.load_mesh(mesh_path)
    input_dict = loader.query_mesh_rig()

    bbox_center = loader.bbox_center
    distance = loader.bbox_scale * 2
    
    camera_position = bbox_center + distance * view_params["cam_pos_offset"]
    look_at_matrix = look_at(camera_position, bbox_center, view_params["up_vector"])
    renderer.set_camera(look_at_matrix)
    renderer.align_light_to_camera()
    color_img, _ = renderer.render(input_dict)

    ax.imshow(color_img)
    ax.axis('off')
    ax.set_title(title)
    
def compute_frechet_distance(mu1, cov1, mu2, cov2):
    """Fréchet Distance between two Gaussians (from StyleGAN paper)"""
    sqrt_cov = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(sqrt_cov):  # 数值不稳定时去掉虚部
        sqrt_cov = sqrt_cov.real
    return np.sum((mu1 - mu2)**2) + np.trace(cov1 + cov2 - 2*sqrt_cov)

def kl_divergence_gaussians(mu1, cov1, mu2, cov2):
    """
    KL(Q || P):  Q= N(mu2, cov2),  P= N(mu1, cov1)
    """
    eps = 1e-6  # 防止奇异矩阵
    cov1 += np.eye(cov1.shape[0]) * eps
    cov2 += np.eye(cov2.shape[0]) * eps

    inv_cov1 = np.linalg.inv(cov1)
    trace_term = np.trace(inv_cov1 @ cov2)
    mean_term = (mu1 - mu2).T @ inv_cov1 @ (mu1 - mu2)
    logdet_term = np.log(np.linalg.det(cov1) / np.linalg.det(cov2))

    d = mu1.shape[0]
    kl = 0.5 * (trace_term + mean_term - d + logdet_term)
    return kl

def collect_vae_embeddings(vae, dataloader, device):
    vae.eval()
    all_mu = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collect VAE Train Embedding"):
            batch = batch.to(device)
            mu, logvar = vae.encode(batch.node_pos, batch.bond_index)
            z = vae.reparameterize(mu, logvar)  # or use mu if you want deterministic
            all_mu.append(z.cpu())
    return torch.cat(all_mu, dim=0)  # [N_nodes_train, latent_dim]

def save_skeleton_file(filepath, positions, edge_index, root_joint_idx=0):
    """ Saves skeleton data. """
    num_nodes = positions.shape[0]
    positions_np = positions.cpu().numpy()
    try:
        with open(filepath, 'w') as f:
            for i in range(num_nodes):
                x, y, z = positions_np[i]
                f.write(f"joints joint{i} {x:.7f} {y:.7f} {z:.7f}\n")
            f.write(f"root joint{root_joint_idx}\n")
            edge_index_cpu = edge_index.cpu()
            for i in range(edge_index_cpu.shape[1] // 2):
                parent = edge_index_cpu[0, 2*i].item()
                child = edge_index_cpu[1, 2*i].item()
                f.write(f"hier joint{parent} joint{child}\n")
    except Exception as e:
        print(f"ERROR: Failed to save skeleton file {filepath}: {e}")