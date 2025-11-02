import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
import pickle
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from skimage import io as skio
from instantmesh.utils.train_util import instantiate_from_config
from instantmesh.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
    get_custom_input_cameras
)
from instantmesh.utils.mesh_util import save_obj, save_obj_with_mtl
from instantmesh.utils.infer_util import remove_background, resize_foreground, save_video
from PIL import Image

def get_render_cameras(batch_size=1, M=120, radius=5.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False, with_mask=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            res = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )
            frame = res['img']
            if with_mask:
                mask = res['mask']
                frame = torch.cat([frame, mask], dim=-3)
        else:
            with torch.cuda.amp.autocast():
                frame = model.forward_synthesizer(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames

###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--gpu', type=int, default=0, help='GPU Device to use.')
parser.add_argument('--seed', type=int, default=66, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.0, help='Render distance.')
parser.add_argument('--num_view', type=int, default=4, choices=[4, 6], help='Number of input views.')
parser.add_argument('--repeat_num', type=int, default=4, help='Number of repeats.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
parser.add_argument('--save_img', action='store_true', help='Save rendered imgs.')
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device(f'cuda:{args.gpu}')
# device = torch.device('cuda')

print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
model_ckpt_path = infer_config.model_path
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()


run_repeat = args.repeat_num
num_view = args.num_view
radius = args.distance # previous as 5.0
# folders = os.listdir(args.input_path)
folders = [item for item in os.listdir(args.input_path) 
           if os.path.isdir(os.path.join(args.input_path, item))]
# print(folders)
# print(len(folders))
# exit(0)
for i in range(len(folders)):
    oid = folders[i]
    # make output directories
    save_dir = os.path.join(args.output_path,oid)
    os.makedirs(save_dir, exist_ok=True)
    camera_dict = pickle.load(open(os.path.join(args.input_path,oid,'cam_dict.pkl'),'rb'))
    
    outputs = []
    for p in range(run_repeat):
        input_files = []
        for j in range(num_view):
            input_files.append(os.path.join(args.input_path,oid,f'gen_{p}_{j}.png'))
        # print(input_files)
        
        imgs = []
        for idx, image_file in enumerate(input_files):
            # img = np.array(Image.open(image_file))
            print(image_file)
            img = skio.imread(image_file)
            
            mask = img[:,:,-1]
            _img = img[:,:,:-1]
            mask = np.stack([mask,mask,mask],axis=-1).astype(bool)
            
            new_img = np.ones_like(_img)*255
            new_img[mask] = _img[mask]
            
            # skio.imsave('new_img.png',new_img)
            # img = remove_background(img, rembg_session)
            
            imgs.append(new_img)
        images = np.stack(imgs,axis=0) #(4,256,256,3)
        images = np.asarray(images, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(0,3,1,2).contiguous().float()     # (4,3,256,256)
        outputs.append({'name': f'{oid}_{p}', 'images': images, 'camera':camera_dict})
        
        
    ###############################################################################
    # Stage 2: Reconstruction.
    ###############################################################################

    chunk_size = 20 if IS_FLEXICUBES else 1

    for idx, sample in enumerate(outputs):
        
        name = sample['name']
        if os.path.exists(os.path.join(save_dir,name)) and len(os.listdir(os.path.join(save_dir,name))) == 12:
            continue
        print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')
        cam_dict = sample['camera']
        input_cameras =get_custom_input_cameras(cam_dict['azimuth'],camera_dict['elevation'],radius=radius).to(device)

        images = sample['images'].unsqueeze(0).to(device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
        with torch.no_grad():
            # get triplane
            planes = model.forward_planes(images, input_cameras)

            # get mesh
            mesh_path_idx = os.path.join(save_dir, f'{name}.obj')

            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=args.export_texmap,
                **infer_config,
            )
            if args.export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                # rotation correction
                angle = torch.deg2rad(torch.tensor(90))
                rotation_matrix_x = torch.tensor([
                        [1, 0, 0],
                        [0, torch.cos(angle), -torch.sin(angle)],
                        [0, torch.sin(angle), torch.cos(angle)]
                    ]).to(device)
                rotation_matrix_y = torch.tensor([
                        [torch.cos(angle), 0, torch.sin(angle)],
                        [0, 1, 0],
                        [-torch.sin(angle), 0, torch.cos(angle)]
                    ]).to(device)
                vertices = vertices@rotation_matrix_x@rotation_matrix_y
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path_idx,
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, mesh_path_idx)
            print(f"Mesh saved to {mesh_path_idx}")
            
            torch.cuda.empty_cache()

            # get video
            if args.save_video:
                video_path_idx = os.path.join(save_dir, f'{name}.mp4')
                render_size = infer_config.render_resolution
                render_cameras = get_render_cameras(
                    batch_size=1, 
                    M=120, 
                    radius=radius, 
                    elevation=20.0,
                    is_flexicubes=IS_FLEXICUBES,
                ).to(device)
                
                frames = render_frames(
                    model, 
                    planes, 
                    render_cameras=render_cameras, 
                    render_size=render_size, 
                    chunk_size=chunk_size, 
                    is_flexicubes=IS_FLEXICUBES,
                )

                save_video(
                    frames,
                    video_path_idx,
                    fps=30,
                )
                print(f"Video saved to {video_path_idx}")
            if args.save_img:
                render_size = infer_config.render_resolution
                render_cameras = get_render_cameras(
                    batch_size=1, 
                    M=12, 
                    radius=radius, 
                    elevation=camera_dict['elevation'][0],
                    is_flexicubes=IS_FLEXICUBES,
                ).to(device)
                
                frames = render_frames(
                    model, 
                    planes, 
                    render_cameras=render_cameras, 
                    render_size=render_size, 
                    chunk_size=chunk_size, 
                    is_flexicubes=IS_FLEXICUBES,
                    with_mask=True
                )
                frames_img = [(frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in frames]
                os.makedirs(os.path.join(save_dir, name),exist_ok=True)
                for k in range(len(frames_img)):
                    img_save_dir = os.path.join(save_dir, name,f'{k}.png')
                    
                    skio.imsave(img_save_dir, frames_img[k])