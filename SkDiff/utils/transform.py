import torch 
import math
import random
import numpy as np

class FeaturizeGraph(object):
    def __init__(self, use_rotate, angle_range=(-10, 10), use_perturb=True, perturb_prob=0.5):
        super().__init__()

        self.use_rotate = use_rotate
        self.use_perturb = use_perturb
        self.perturb_prob = perturb_prob
        self.angle_range = angle_range
        self.axes = ['x', 'y', 'z']

        self.follow_batch = ['halfedge_type']
        self.exclude_keys = ['orig_keys', "num_nodes", "node_pos", "num_bonds", "bond_index"]
    
    @staticmethod
    def _rotation_matrix(angle_deg: float, axis: str, device):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        if axis == 'x':
            mat = [[1, 0, 0], [0, c, -s], [0, s, c]]
        elif axis == 'y':
            mat = [[c, 0,  s], [0, 1, 0], [-s, 0, c]]
        elif axis == 'z':
            mat = [[c, -s, 0], [s,  c, 0], [0, 0, 1]]
        else:
            raise ValueError("axis must be 'x', 'y' or 'z'")
        return torch.tensor(mat, dtype=torch.float32, device=device)

    def __call__(self, data):
        '''
            data_type: {
                "file_path": file_path, 
                "node_pos": node 3d positions, 
                "num_bonds": len(edges),
                "bond_index": bond index,
            }
        '''

        pos = data.node_pos


        # bounds = np.array([pos.min(axis=0), pos.max(axis=0)])
        min_values = pos.min(axis=0).values
        max_values = pos.max(axis=0).values
        bounds = torch.stack([min_values, max_values])

        data.translation = (bounds[0] + bounds[1])[None, :] / 2
        data.scale = (bounds[1] - bounds[0]).max()

        pos = pos - data.translation
        pos = pos / (data.scale + 1e-5)

        # 随机挑选十分之一的节点进行位置扰动
        if self.use_perturb:
            if self.perturb_prob == None or np.random.random() < self.perturb_prob:
                num_nodes = pos.shape[0]
                if num_nodes > 10:
                    perturb_indices = random.sample(range(num_nodes), num_nodes // 10)
                    perturb_amount = torch.randn_like(pos[perturb_indices])
                    perturb_amount = torch.clamp(perturb_amount, -0.5, 0.5)
                    pos[perturb_indices] += perturb_amount * 0.1

        if self.use_rotate:
            # === 随机旋转 ===
            
            axis   = random.choice(self.axes)
            angle  = random.uniform(*self.angle_range)    # [-20, 20]
            R = self._rotation_matrix(angle, axis, pos.device)
            pos = torch.matmul(pos, R.T)     # (N,3)

            # === 写回 data ===
            data.node_pos      = pos                 # 归一化后坐标
            data.rot_axis      = axis                     # str
            data.rot_angle_deg = angle                    # float

            return data
        else:
            data.node_pos = pos
            return data
        
        

        
        
        