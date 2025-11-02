import torch
import torchvision.transforms as T
import numpy as np
import skimage.io as io
import os
import skeleton.render_one_pose as sr
import pickle
from skdream.utils.camera import create_camera_to_world_matrix
from skdream.pipeline_skdream import load_skdream_pipeline
from argparse import ArgumentParser
import json
from rembg import remove,new_session
from PIL import Image
from glob import glob
from render import util

def normalize_to_cube(points,scale_factor=1.0):
    min_vals = points.min(dim=0).values
    max_vals = points.max(dim=0).values

    center = (min_vals + max_vals) / 2.0
    centered_points = points - center
    
    bbox_size = max_vals - min_vals
    max_length = bbox_size.max()
    scale = 1 / max_length
    scale = scale * scale_factor

    normalized_points = centered_points * scale

    return normalized_points,-center,scale

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--num_views',type=int,default=4)
    parser.add_argument('--mvc_ckpt',type=str)
    parser.add_argument('--data_dir',type=str,default='objsk_eval')
    parser.add_argument('--neg_prompt',type=str,default='')
    parser.add_argument('--cond_scale',type=float,default=1.0)
    parser.add_argument('--repeat_num',type=int,default=1)
    parser.add_argument('--gpu',type=int,default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    data_dir = args.data_dir
    eval_dict = json.load(open(os.path.join(data_dir,'eval.json'),'r'))
    # for k in eval_dict.copy().keys():
    #     if os.path.exists(os.path.join(args.save_dir,k)):
    #         print(f"{os.path.join(args.save_dir,k)} already exists, skip.")
    #         del eval_dict[k]
    if len(eval_dict) == 0:
        print("No new data to process, exiting.")
        exit(0)

    sk_list = [os.path.join(data_dir,'cano_sk',k+'.txt') for k in eval_dict.keys()]
    np.random.seed(0)
    azim_list = np.random.randint(0,360,len(sk_list))
    elev_list = np.random.randint(0,30,len(sk_list))
    
    # Generate conditions
    for si,sk_file in enumerate(sk_list):
        # prepare skeleton
        obj_name = sk_file.split('/')[-1].split('.')[0]
        save_dir = os.path.join(args.save_dir,obj_name)

        os.makedirs(save_dir,exist_ok=True)
        cam_dict = {'mv':[],'mvp':[],'campos':[],'c2w':[],'elevation':[],'azimuth':[],'distance':[]}
        # prepare camera
        num_views = args.num_views
        fovy = np.deg2rad(30)
        proj_mtx = util.perspective(fovy, 1.0, 0.5, 1000)
        
        elevation = elev_list[si]
        azimuth = azim_list[si]
        distance = 2.5
        for i in range(num_views):
            rotate_x =  np.deg2rad(elevation)
            rotate_y =  np.deg2rad(azimuth)
            mv     = util.translate(0, 0, -distance) @ (util.rotate_x(-rotate_x) @ util.rotate_y(-rotate_y))
            mvp    = proj_mtx @ mv
            campos = torch.linalg.inv(mv)[:3, 3]
            c2w = create_camera_to_world_matrix(elevation,azimuth,cam_dist=1)
            c2w = torch.tensor(c2w,dtype=mv.dtype,device=mv.device)
            
            cam_dict['mv'].append(mv)
            cam_dict['mvp'].append(mvp)
            cam_dict['campos'].append(campos)
            cam_dict['elevation'].append(elevation)
            cam_dict['azimuth'].append(azimuth)
            cam_dict['distance'].append(distance)
            cam_dict['c2w'].append(c2w)
            
            azimuth = azimuth + 360//num_views
        cam_dict['mvp'] = torch.stack(cam_dict['mvp'],dim=0)
        cam_dict['mv'] = torch.stack(cam_dict['mv'],dim=0)
        cam_dict['campos'] = torch.stack(cam_dict['campos'],dim=0)
        cam_dict['c2w'] = torch.stack(cam_dict['c2w'],dim=0)
        
        with open(os.path.join(save_dir,'cam_dict.pkl'),'wb') as f:
            pickle.dump(cam_dict,f)
        # print(obj_name)
        # prepare skeleton
        joints,bones_idx,parts = sr.get_skeleton_info(sk_file)
        joints = torch.tensor(joints)
        joints_2d,joints_depth = sr.project_joints(joints,cam_dict['mvp'])
        sk_list = []
        for iv in range(num_views):
            sorted_bones_idx = sr.sort_bones_depth(joints_depth[iv],bones_idx)
            # draw skeleton
            canvas = np.zeros((512,512,3),dtype=np.uint8)
            depth_values = sr.process_depth(joints_depth[iv])
            canvas = sr.draw_ccm_with_depth(canvas,joints,joints_2d[iv],sorted_bones_idx,parts,depth_values)
            img = Image.fromarray(canvas,mode='RGBA').resize((256,256),resample=Image.Resampling.NEAREST)
            img.save(os.path.join(save_dir,f'cond_{iv}.png'))
            
    # Generate images
    if args.neg_prompt == 'default':
        neg_prompt = 'low poly, white model, noise, strange color, ugly, oversaturated, doubled face, b&w, sepia, freckles, paintings, sketches, worst quality, low quality, lowres, monochrome, grayscale, error, blurry, artifacts,'
    elif args.neg_prompt == '':
        neg_prompt = None
    else:
        neg_prompt = args.neg_prompt
    mvc_ckpt = args.mvc_ckpt
    pipe = load_skdream_pipeline(pretrained_controlnet_name_or_path=mvc_ckpt,
                                    pretrained_model_name_or_path='./ckpt/mvdream-sd21-diffusers-lzq',
                                        num_views=num_views,weights_dtype=torch.float16,device=device)
    cond_list = sorted([os.path.join(args.save_dir,k) for k in eval_dict.keys()])
    cond_channels = pipe.controlnet.conditioning_channels
    
    rembg_session = new_session("is_general_use")
    
    for folder in cond_list:
        if len(glob(folder+'/gen*.png')) == num_views*args.repeat_num:
            continue
        cam_dict = pickle.load(open(os.path.join(folder,'cam_dict.pkl'),'rb'))
        c2w = cam_dict['c2w'].reshape(1,num_views,-1).to(device)

        cond_imgs = []
        for i in range(num_views):
            cond_imgs.append(Image.open(os.path.join(folder,f'cond_{i}.png')))
        def rgb2binary(image):
            gray_array = np.array(image.convert('RGB').convert('L'))
            threshold = 1  
            binary_array = np.where(gray_array > threshold, 1, 0).astype(np.uint8)* 255
            return np.stack([binary_array]*3,axis=-1)
        if cond_channels == 4:
            cond_imgs = [np.array(cond.convert('RGBA')) for cond in cond_imgs]
            conds = cond_imgs
        if cond_channels == 3:
            cond_imgs = [np.array(cond.convert('RGB')) for cond in cond_imgs]
            conds = cond_imgs
        elif cond_channels == 1:
            cond_imgs = [rgb2binary(cond) for cond in cond_imgs]
            conds = [cond[...,:1] for cond in cond_imgs]
        
        xm = T.Compose([T.ToTensor()])
        cond_tensor = [xm(x) for x in conds]
        cond_tensor = torch.stack(cond_tensor,dim=0).unsqueeze(0).to(device)
        cond_tensor = cond_tensor * 2 -1

        
        for r in range(args.repeat_num):
            prompt = eval_dict[folder.split('/')[-1]]['caption']
            images = pipe(prompt=prompt,negative_prompt=neg_prompt,hint=cond_tensor,c2ws=c2w,
                          guidance_scale=7.5,controlnet_conditioning_scale=args.cond_scale,
                          guess_mode=False,blind_control_until_step=None,output_type="numpy").images
            images = (images*255).astype(np.uint8)
            
            for i in range(num_views):
                io.imsave(os.path.join(folder,f'mix_{r}_{i}.png'),(conds[i][:,:,:3]*0.3+images[i]*0.8).astype(np.uint8))
                mask = remove(images[i],only_mask=True,session=rembg_session,post_process_mask=True,alpha_matting=False,alpha_matting_foreground_threshold=150)
                io.imsave(os.path.join(folder,f'gen_{r}_{i}.png'),np.concatenate((images[i],mask[...,None]),axis=-1))
    
