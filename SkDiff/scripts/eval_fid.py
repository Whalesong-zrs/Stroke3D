import sys
import os
import torch
import numpy as np
from scipy import linalg
from glob import glob
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from models.vae import NodeCoordVAE

method_dirs = {
    "gt1": "datasets/test",
    "gt2": "datasets/test_diverse",
    "stroke3d": [
        "outputs/skeleton_xy",
        "outputs/skeleton_yz",
        "outputs/skeleton_xz",
    ],
}
method_patterns = {
    "gt1": "*.txt",
    "gt2": "*.txt",
    "stroke3d": "*.txt",
}
selected_methods = ["stroke3d"]
common_only = True
output_dir = "outputs/fid"
os.makedirs(output_dir, exist_ok=True)


def main():
    # === Data Preparing ===
    print("Building rig paths...")
    rig_paths = build_rig_paths(selected_methods, common_only=common_only)

    # === VAE Loading (Freeze) ===
    print("Loading and freezing VAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = NodeCoordVAE(
        coord_dim=3,
        hidden_channels=64,
        latent_dim=32,
        norm_type="layer"
    ).to(device)
    vae_ckpt_path = "ckpts/Sk-VAE.pt"
    if not vae_ckpt_path or not os.path.isfile(vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint path not found or specified: {vae_ckpt_path}")
    print(f"Loading VAE weights from: {vae_ckpt_path}")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae_state_dict = vae_ckpt.get('model_state_dict', vae_ckpt.get('model', vae_ckpt))
    if vae_state_dict:
        if 'encoder' in vae_state_dict and 'decoder' in vae_state_dict:
            vae.encoder.load_state_dict(vae_state_dict['encoder'])
            vae.decoder.load_state_dict(vae_state_dict['decoder'])
        else:
            vae.load_state_dict(vae_state_dict)
        print("VAE weights loaded.")
    else:
        raise KeyError(f"Could not find VAE state dict in checkpoint: {vae_ckpt_path}")

    # === FID Computation ===
    for method_name, gt_paths, pred_paths in rig_paths:
        fid_score = compute_skeleton_fid(method_name, gt_paths, pred_paths, vae, device=device)
        print(f"[{method_name}] FID: {fid_score:.4f}")


def load_method(method_name):
    method_pattern = method_patterns.get(method_name, "*.txt")
    method_dir = method_dirs[method_name]

    if isinstance(method_dir, list):
        # Keep one prediction per UUID grouped across the three GNNDiff views.
        paths_by_dir = {}
        uuids_by_dir = {}
        for view_dir in method_dir:
            rig_paths = sorted(glob(os.path.join(view_dir, "**", method_pattern), recursive=True))
            uuids = [extract_uuid(path, method_name) for path in rig_paths]
            paths_by_dir[view_dir] = rig_paths
            uuids_by_dir[view_dir] = uuids

        reference_dir = method_dir[0]
        uuids = uuids_by_dir[reference_dir]
        all_rig_paths = []
        for uuid in uuids:
            entries = []
            for view_dir in method_dir:
                if uuid not in uuids_by_dir[view_dir]:
                    raise ValueError(f"Missing {uuid} in {view_dir}")
                entries.append(paths_by_dir[view_dir][uuids_by_dir[view_dir].index(uuid)])
            all_rig_paths.append(entries)
    else:
        all_rig_paths = sorted(glob(os.path.join(method_dir, "**", method_pattern), recursive=True))
        uuids = [extract_uuid(path, method_name) for path in all_rig_paths]

    return uuids, all_rig_paths


def extract_uuid(path, method_name=None):
    if method_name is not None and "unirig" in method_name:
        return os.path.normpath(path).split(os.sep)[-2]
    return os.path.basename(path).split('.')[0].split('_pred')[0]


def build_rig_paths(selected_methods, common_only=True):
    uuid_map = {}
    paths_map = {}

    # Merge the standard and diverse test sets before matching predictions.
    uuids, paths = load_method("gt1")
    uuid_map["gt"] = uuids
    paths_map["gt"] = paths
    uuids, paths = load_method("gt2")
    uuid_map["gt"].extend(uuids)
    paths_map["gt"].extend(paths)
    for m in selected_methods:
        uuids, paths = load_method(m)
        uuid_map[m] = uuids
        paths_map[m] = paths

    if common_only:
        common_uuids = set(uuid_map[selected_methods[0]])
        for m in selected_methods[1:]:
            common_uuids &= set(uuid_map[m])
        common_uuids = sorted(common_uuids)
        print(f"Found {len(common_uuids)} common objects across selected methods.")
    else:
        for m in selected_methods:
            print(f"Found {len(uuid_map[m])} rig paths for {m}.")

    # 构造任务队列
    rig_paths = []
    for m in selected_methods:
        gt_paths = []
        pred_paths = []
        uuids = common_uuids if common_only else uuid_map[m]
        for uuid in uuids:
            if uuid in uuid_map[m]:
                gt_paths.append(paths_map["gt"][uuid_map["gt"].index(uuid)])
                pred_paths.append(paths_map[m][uuid_map[m].index(uuid)])
        rig_paths.append((m, gt_paths, pred_paths))

    return rig_paths

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def parse_rig_txt(rig_path):
    joints = []
    joints_names = []
    bones = []
    joint_name_to_idx = {}

    with open(rig_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'joints':
                joint_name = parts[1]
                joint_pos = [float(parts[2]), float(parts[3]), float(parts[4])]
                joint_name_to_idx[joint_name] = len(joints)
                joints.append(joint_pos)
                joints_names.append(joint_name)
            elif parts[0] == 'hier':
                parent_joint = joint_name_to_idx[parts[1]]
                child_joint = joint_name_to_idx[parts[2]]
                bones.append([parent_joint, child_joint])

    joints = np.array(joints) if joints else np.zeros((1, 3))
    bones = np.array(bones) if bones else np.zeros((0, 2), dtype=int)

    return joints, bones


def extract_skeleton_feature(rig_path, vae_model, device="cpu", random=False):
    joints, bones = parse_rig_txt(rig_path)  # (N,3), (E,2)

    x = torch.tensor(joints, dtype=torch.float32).to(device)
    edge_index = torch.tensor(bones.T, dtype=torch.long).to(device)

    mu, logvar = vae_model.encode(x, edge_index)
    z = vae_model.reparameterize(mu, logvar, random=random).detach().cpu().numpy()

    return z, joints, bones


gt_features_already = {}

def compute_skeleton_fid(method_name, gt_paths, pred_paths, vae_model, device="cpu"):
    vae_model.eval()
    gt_features = []
    gt_joints = []
    gt_bones = []
    pred_features = []
    pred_joints = []
    pred_bones = []

    for p in gt_paths:
        if p in gt_features_already:
            feature, joints, bones = gt_features_already[p]
        else:
            feature, joints, bones = extract_skeleton_feature(p, vae_model, device=device, random=False)
            gt_features_already[p] = (feature, joints, bones)
        gt_features.append(feature)
        gt_joints.append(joints)
        gt_bones.append(bones)

    for pred_entry in pred_paths:
        # A method with multiple inference views contributes one sample per UUID.
        # Use the first available view, preserving the previous first-match behavior.
        if isinstance(pred_entry, list):
            pred_entry = next((p for p in pred_entry if os.path.isfile(p)), None)
        if pred_entry is None:
            continue
        feature, joints, bones = extract_skeleton_feature(pred_entry, vae_model, device=device, random=False)
        pred_features.append(feature)
        pred_joints.append(joints)
        pred_bones.append(bones)

    np.save(os.path.join(output_dir, f"{method_name}.npy"), {
        "gt_features": gt_features,
        "gt_joints": gt_joints,
        "gt_bones": gt_bones,
        "gt_paths": gt_paths,
        "pred_features": pred_features,
        "pred_joints": pred_joints,
        "pred_bones": pred_bones,
        "pred_paths": pred_paths
    })

    gt_features = np.array([f.mean(axis=0) for f in gt_features])
    pred_features = np.array([f.mean(axis=0) for f in pred_features])

    mu_g, sigma_g = gt_features.mean(axis=0), np.cov(gt_features, rowvar=False)
    mu_p, sigma_p = pred_features.mean(axis=0), np.cov(pred_features, rowvar=False)

    return calculate_frechet_distance(mu_g, sigma_g, mu_p, sigma_p)


if __name__ == "__main__":
    method_dirs["rignet"] = "/root/autodl-fs-data3/RigNet/magicarti_data/anthropomorphic"
    print("============= Evaluating Anthropomorphic =============")
    main()

    method_dirs["rignet"] = "/root/autodl-fs-data3/RigNet/magicarti_data/character"
    print("============= Evaluating Character =============")
    main()

    method_dirs["rignet"] = "/root/autodl-fs-data3/RigNet/magicarti_data/animal"
    print("============= Evaluating Animal =============")
    main()

    method_dirs["rignet"] = "/root/autodl-fs-data3/RigNet/magicarti_data/plant"
    print("============= Evaluating Plant =============")
    main()

    method_dirs["rignet"] = "/root/autodl-fs-data3/RigNet/magicarti_data"
    print("============= Evaluating All =============")
    main()
