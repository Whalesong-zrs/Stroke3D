import os
import sys
import json
import numpy as np
from tqdm import tqdm
from glob import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from utils.eval_utils import chamfer_dist, joint2bone_chamfer_dist, bone2bone_chamfer_dist


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
output_dir = "outputs/chamfer"
os.makedirs(output_dir, exist_ok=True)


def main():
    print("Building rig paths...")
    rig_paths = build_rig_paths(selected_methods, common_only=common_only)

    for method_name, gt_paths, pred_paths in rig_paths:
        print(f"============= {method_name} =============")
        output_path = os.path.join(output_dir, f"{method_name}.json")
        evaluate_chamfer(method_name, gt_paths, pred_paths, output_path)


def extract_uuid(path, method_name):
    if method_name is not None and "unirig" in method_name:
        uuid = os.path.normpath(path).split(os.sep)[-2]
    else:
        uuid = os.path.basename(path).split('.')[0].split('_pred')[0]
    return uuid

def evaluate_chamfer(method_name, gt_paths, pred_paths, output_path):
    avg_j2j_cd = 0.0
    avg_j2b_cd = 0.0
    avg_b2b_cd = 0.0
    num_valid = 0

    eval_results = {}

    for gt_path, pred_entry in tqdm(zip(gt_paths, pred_paths), desc=method_name, total=len(gt_paths)):
        uuid = extract_uuid(gt_path, None)
        gt_joints, gt_bones = parse_rig_txt(gt_path)

        # 如果 pred_entry 是列表（多个推理结果），则选择最小值
        if isinstance(pred_entry, list):
            j2j_cds, j2b_cds, b2b_cds = [], [], []
            for pred_path in pred_entry:
                if not os.path.exists(pred_path):
                    continue
                pred_joints, pred_bones = parse_rig_txt(pred_path)
                j2j_cds.append(chamfer_dist(pred_joints, gt_joints))
                j2b_cds.append(joint2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones))
                b2b_cds.append(bone2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones))
            j2j_cd, j2b_cd, b2b_cd = min(j2j_cds), min(j2b_cds), min(b2b_cds)
            pred_entry = pred_entry[j2j_cds.index(j2j_cd)]
        else:
            pred_joints, pred_bones = parse_rig_txt(pred_entry)
            if "unirig" in method_name:
                pred_joints /= 2
            j2j_cd = chamfer_dist(pred_joints, gt_joints)
            j2b_cd = joint2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones)
            b2b_cd = bone2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones)

        eval_results[uuid] = {
            "j2j": j2j_cd,
            "j2b": j2b_cd,
            "b2b": b2b_cd,
            "gt_path": gt_path,
            "pred_path": pred_entry,
        }

        avg_j2j_cd += j2j_cd
        avg_j2b_cd += j2b_cd
        avg_b2b_cd += b2b_cd
        num_valid += 1

    if num_valid > 0:
        print(f"Average J2J Chamfer Distance: {avg_j2j_cd/num_valid}")
        print(f"Average J2B Chamfer Distance: {avg_j2b_cd/num_valid}")
        print(f"Average B2B Chamfer Distance: {avg_b2b_cd/num_valid}")

        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=4)

        print(f"Successfully saved to {output_path} with {num_valid} valid samples.")

def load_method(method_name):
    method_dir = method_dirs[method_name]
    method_pattern = method_patterns.get(method_name, "*.txt")
    if isinstance(method_dirs[method_name], list):
        method_dir = method_dir[0]

    if isinstance(method_dirs[method_name], list):
        rig_paths_map = {}
        uuids_map = {}
        for method_dir in method_dirs[method_name]:
            rig_paths = sorted(glob(os.path.join(method_dir, "**", method_pattern), recursive=True))
            uuids = [extract_uuid(p, method_name) for p in rig_paths]
            rig_paths_map[method_dir] = rig_paths
            uuids_map[method_dir] = uuids

        all_rig_paths = []  # [[model1_path1, model1_path2, ...], [model2_path1, model2_path2, ...], ...]
        for uuid in uuids_map[method_dir]:
            all_rig_paths.append([rig_paths_map[method_dir][uuids_map[method_dir].index(uuid)] for method_dir in method_dirs[method_name]])
    else:
        rig_paths = sorted(glob(os.path.join(method_dir, "**", method_pattern), recursive=True))
        uuids = [extract_uuid(p, method_name) for p in rig_paths]
        all_rig_paths = rig_paths

    return uuids, all_rig_paths


def build_rig_paths(selected_methods, common_only=True):
    uuid_map = {}
    paths_map = {}

    # 加载 GT 及所有方法的文件路径
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
            gt_paths.append(paths_map["gt"][uuid_map["gt"].index(uuid)])
            pred_paths.append(paths_map[m][uuid_map[m].index(uuid)])
        rig_paths.append((m, gt_paths, pred_paths))

    return rig_paths


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


if __name__ == "__main__":
    main()
