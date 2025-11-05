import torch
import numpy as np
import math
import cv2
from skeleton.rig_parser import Info
from collections import OrderedDict

def assign_ccm_to_bones(joints_3d, bones):
    """Assign colors to bones based on the spatial 3D coordinates of their midpoints."""
    
    # Normalize the joints to fit within the (0,1)^3 cube
    if isinstance(joints_3d,torch.Tensor):
        joints_3d = joints_3d.detach().cpu()
    joints_3d = np.array(joints_3d)
    min_coords = joints_3d.min(axis=0)
    max_coords = joints_3d.max(axis=0)
    normalized_joints = (joints_3d - min_coords) / (max_coords - min_coords)
    
    # Compute the midpoints of each bone
    midpoints = [(normalized_joints[bone[0]] + normalized_joints[bone[1]]) / 2 for bone in bones]
    midpoints = np.array(midpoints)
    
    # Map the midpoint coordinates to colors using a colormap
    colors = (midpoints * 255).astype(np.uint8)
    return colors

def draw_ccm_with_depth(canvas: np.ndarray, joints_3d, joints, bones, parts, depth_values) -> np.ndarray:

    H, W, C = canvas.shape
    stickwidth = 4
    depth_canvas = np.copy(canvas)[:,:,0]
    bone_colors = assign_ccm_to_bones(joints_3d,bones)

    for i in range(len(bones)):
        (j1_index, j2_index) = bones[i]
        color = (int(bone_colors[i,0]),int(bone_colors[i,1]),int(bone_colors[i,2]))
        j1 = joints[j1_index]
        j2 = joints[j2_index]
        

        if j1 is None or j2 is None:
            continue
        for joint in (j1,j2):
            x, y = joint[0], joint[1]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
        
        
        
        Y = np.array([j1[0], j2[0]]) * float(H)
        X = np.array([j1[1], j2[1]]) * float(W)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
        
        j1_d = float(depth_values[j1_index])
        j2_d = float(depth_values[j2_index])
        # Create an empty black image
        mask = np.zeros((canvas.shape[0],canvas.shape[1]), dtype=np.uint8)

        # Draw the polygon on the image
        cv2.fillPoly(mask, [polygon], 255)

        y, x = np.where(mask == 255)
        p_j1 = np.sqrt((y - X[0])**2 + (x - Y[0])**2)

        normalized_dist = np.clip(p_j1 / (length + 1e-6), 0, 1)
        color_values = np.clip(normalized_dist * (j2_d - j1_d) + j1_d, 0, 255).astype(np.uint8)

        depth_canvas[y, x] = color_values
    canvas = np.concatenate([canvas,np.expand_dims(depth_canvas,-1)],axis=-1)
    return canvas

def process_depth(depth_values):
    depth_values = -np.array(depth_values)
    depth_values = (depth_values-np.min(depth_values))/(np.max(depth_values)-np.min(depth_values)+1e-6)
    depth_values = depth_values*0.8 + 0.2
    
    depth_values = (depth_values*255).astype(np.uint8)
    return depth_values

def get_skeleton_info(sk_dir):
    rig_info = Info(sk_dir)
    rig_info.assign_joint_part(rig_info.root)
    joint_dict = rig_info.get_joint_dict()
    part_dict = rig_info.get_part_dict()
    bones_idx = rig_info.get_bones_idx()
    parts = list(part_dict.values())
    joints = list(joint_dict.values())
    return joints,bones_idx,parts

def xfm_points(points, matrix):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def project_joints(joints:torch.Tensor,mvp:torch.Tensor,shift=(0,0,0),scale=1) -> np.ndarray:
    ''' default execute on cuda
       joints: (B,N,3) or (N,3)
       mvp: (B,4,4)
       return: joints_2d: (B,N,2), joints_depth: (B,N)
    '''
    B = mvp.shape[0]
    if len(joints.shape) == 2:
        joints = joints.unsqueeze(0).repeat(B,1,1)
    
    joints = joints
    joints[...,0] = joints[...,0] - shift[0]
    joints[...,1] = joints[...,1] - shift[1]
    joints[...,2] = joints[...,2] - shift[2]
    joints = joints / scale
    # print(joints[None,...].shape,joints[None,...])
    
    joints_2d = xfm_points(joints.cuda(), mvp.cuda())
    joints_depth = joints_2d[...,3:4].detach().cpu().numpy()
    joints_2d = joints_2d[...,0:2]/joints_2d[...,3:4]/2 + 0.5 #(-1,1)homo -> (0,1)ecu
    joints_2d = joints_2d.detach().cpu().numpy()
    
    return joints_2d,joints_depth
def sort_bones_depth(joint_depth,bones_idx):
    '''
    joint_depth: (N,)
    bones_idx: (K,2)
    return: sorted_bones_idx (K,2)
    '''
    depth_dict = OrderedDict()
    for bone in bones_idx:
        bone_depth = (joint_depth[bone[0]]+joint_depth[bone[1]])/2
        depth_dict[bone] = bone_depth
    sorted_dict = OrderedDict(sorted(depth_dict.items(),key=lambda x:x[1],reverse=True))
    return list(sorted_dict.keys())
    

