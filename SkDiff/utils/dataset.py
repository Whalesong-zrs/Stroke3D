import os 
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from .misc import torchify_dict

view_name_map = {
    "xy": "front view",
    "yz": "side view",
    "xz": "top view"
}

def build_dataloader(cfg, mode, featurizer, distributed=False, rank=0, world_size=1):
    dataset = get_dataset(cfg.dataset, transform=Compose([featurizer]), mode=mode)
    
    if mode == "train":
        batch_size = cfg.train.batch_size
        shuffle = not distributed
    elif mode == "val":
        batch_size = 1
        shuffle = False
    else:
        raise ValueError("Mode error.")
    
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(mode=="train"))
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        follow_batch=featurizer.follow_batch,
    )


def get_dataset(config, transform, mode):
    if mode == 'train':
        dataset = NodeDataset(config, mode, transform)
    elif mode == 'val':
        dataset = NodeDataset(config, mode, transform, random_view=False, random_tag=False, random_flip=False, use_tag=True)
    return dataset


class NodeData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_node_dicts(info_dict=None, **kwargs):
        instance = NodeData(**kwargs)

        if info_dict is not None:
            for key, item in info_dict.items():
                instance[key] = item
            instance['orig_keys'] = list(info_dict.keys())

        return instance
    

class NodeDataset(Dataset):
    def __init__(self, config, mode, transform=None, random_view=True, random_tag=True, random_flip=True, use_tag=True):
        super().__init__()

        if mode == 'train':
            meta_file = config.train_file
            self.mode = mode
        elif mode == 'val':
            meta_file = config.val_file
            self.mode = mode
        else:
            raise NotImplementedError('Unknown mode: %s, mode must be train or val' % mode)
            
        assert os.path.exists(meta_file), f"Error: The file '{meta_file}' does not exist!"

        self.meta_file = meta_file
        self.clip_emb_root = config.clip_emb_root
        self.transform = transform
        self.random_view = random_view
        self.random_tag = random_tag
        self.random_flip = random_flip
        self.use_tag = use_tag
        self.use_view_tag = True
        self.default_view = "xy"

        self.meta_data = None

        with open(self.meta_file, 'r') as f:
            self.meta_data = json.load(f)

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        data = self.data_process(self.meta_data[idx])
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def data_process(self, item_info):
        rig_path = item_info['rig_path']
        caption = item_info['caption']
        mesh_path = item_info.get('mesh_path', None)
        category = item_info.get('category', None)
        uuid = item_info.get('uuid', None)
        tags = item_info.get('tags', [])
        node_positions_dict = {}
        edges = []

        with open(rig_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == "joints":
                    # 从 "jointX" 中解析出数字ID作为key
                    node_idx = int(parts[1].replace("joint", ""))
                    node_pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    # 将坐标存入字典，key是其真实的ID
                    node_positions_dict[node_idx] = node_pos
                elif parts[0] == "hier":
                    node_idx1 = int(parts[1].replace("joint", ""))
                    node_idx2 = int(parts[2].replace("joint", ""))
                    edges.append((node_idx1, node_idx2))

        if not node_positions_dict:
             # 处理空文件或没有关节的文件
            num_nodes = 0
            node_pos_array = np.zeros((0, 3))
        else:
            num_nodes = max(node_positions_dict.keys()) + 1
            # 3. 创建一个空的numpy数组，然后根据ID填充，确保顺序正确
            node_pos_array = np.zeros((num_nodes, 3))
            for idx, pos in node_positions_dict.items():
                node_pos_array[idx] = pos

        # 边列表 to 边 index
        row, col = [], []
        for node1, node2 in edges:
            row += [node1, node2]
            col += [node2, node1]
        bond_index = np.array([row, col]) if edges else np.empty((2, 0))

        view = self.default_view
        if item_info.get('view', None) is not None:
            view = item_info['view']

        if self.random_view:
            # 随机选择一个视角
            if category == "character" or category == "anthropomorphic":
                probs = [0.9, 0.07, 0.03]
            else:
                probs = [1/3, 1/3, 1/3]
            view = np.random.choice(["xy", "yz", "xz"], p=probs)

        # 根据视角旋转关节
        if view == 'yz':
            axis_order = [2, 1, 0]  # z, y, x
            node_pos_array = node_pos_array[:, axis_order]
            if self.random_flip and np.random.random() < 0.5:
                node_pos_array[:, 0] *= -1  # 随机翻转z轴
        elif view == 'xz':
            axis_order = [0, 2, 1]  # x, z, y
            node_pos_array = node_pos_array[:, axis_order]

        if self.use_tag:
            # 测试集添加所有 tag，训练集随机添加部分 tag
            if self.random_tag is False:
                if len(tags) > 0:
                    tags_str = ", ".join(tags)
                    caption = f"{caption.replace('.', '')}, {tags_str}."
                if self.use_view_tag:
                    view_name = view_name_map[view]
                    caption = f"{caption.replace('.', '')}, {view_name}."
            else:
                if np.random.random() < 0.7 and len(tags) > 0:
                    selected_tags = np.random.choice(tags, size=np.random.randint(1, len(tags)+1), replace=False)
                    tags_str = ", ".join(selected_tags)
                    caption = f"{caption.replace('.', '')}, {tags_str}."
                if np.random.random() < 0.3:
                    view_name = view_name_map[view]
                    caption = f"{caption.replace('.', '')}, {view_name}."

        # clip_emb_path = os.path.join(self.clip_emb_root, f"{base_name}.pt")
        # if os.path.exists(clip_emb_path):
        #     clip_emb = torch.load(clip_emb_path)
        # else:
        #     clip_emb = None
    
        info_dict = {
            "rig_path": rig_path, 
            "mesh_path": mesh_path,
            "num_nodes": num_nodes, 
            "node_pos": node_pos_array,
            "num_bonds": len(edges),
            "bond_index": bond_index,
            "caption": caption,
            "clip_emb": None,
            "category": category,
            "uuid": uuid,
            "view": view
        }

        info_dict = torchify_dict(info_dict)
        data = NodeData.from_node_dicts(info_dict)
        return data

if __name__ == '__main__':
    root = '/home/zrs/skdream_data/data/joint_txt'
    node_dataset = NodeDataset(root)
    data = node_dataset[0]
    print(data)