import os
import torch
import numpy as np
import skimage.io as io
import pickle
import json
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import shutil

# 导入必要的模块 (假设它们在您的Python路径下)
# SKDream and rendering utilities
from skdream.pipeline_skdream import load_skdream_pipeline
from skdream.utils.camera import create_camera_to_world_matrix
import skeleton.render_one_pose as sr
from render import util

# Alignment evaluation utilities
import torchvision.transforms as T
from skalign.model import SkalignModel

# Background removal
from rembg import new_session, remove

def generate_conditions_and_camera(item_id, sk_file, data_dir, save_dir, num_views=4):
    """
    为单个骨骼生成条件图和相机参数字典。
    这部分在排序后会被移动到最终的DPO数据目录中。
    """
    # 临时保存到主输出目录下的 meta 和 skeleton_d 文件夹
    cond_save_dir = os.path.join(save_dir, 'skeleton_d', item_id)
    meta_save_dir = os.path.join(save_dir, 'meta', item_id)
    os.makedirs(cond_save_dir, exist_ok=True)
    os.makedirs(meta_save_dir, exist_ok=True)

    # --- 相机参数生成 ---
    cam_dict = {'mv':[],'mvp':[],'campos':[],'c2w':[],'elevation':[],'azimuth':[],'distance':[]}
    fovy = np.deg2rad(30)
    proj_mtx = util.perspective(fovy, 1.0, 0.5, 1000)
    
    eval_meta = json.load(open(os.path.join(data_dir, 'eval.json'), 'r'))[item_id]
    elevation = eval_meta.get('elevation', np.random.randint(0, 30))
    azimuth = eval_meta.get('azimuth', np.random.randint(0, 360))
    distance = 2.5
    
    for i in range(num_views):
        rotate_x = np.deg2rad(elevation)
        rotate_y = np.deg2rad(azimuth)
        mv = util.translate(0, 0, -distance) @ (util.rotate_x(-rotate_x) @ util.rotate_y(-rotate_y))
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        c2w = create_camera_to_world_matrix(elevation, azimuth, cam_dist=1)
        c2w = torch.tensor(c2w, dtype=mv.dtype, device=mv.device)
        
        cam_dict['mv'].append(mv); cam_dict['mvp'].append(mvp); cam_dict['campos'].append(campos)
        cam_dict['elevation'].append(elevation); cam_dict['azimuth'].append(azimuth)
        cam_dict['distance'].append(distance); cam_dict['c2w'].append(c2w)
        azimuth += 360 // num_views
        
    cam_dict['mvp'] = torch.stack(cam_dict['mvp'], dim=0)
    cam_dict['mv'] = torch.stack(cam_dict['mv'], dim=0)
    cam_dict['campos'] = torch.stack(cam_dict['campos'], dim=0)
    cam_dict['c2w'] = torch.stack(cam_dict['c2w'], dim=0)
    
    with open(os.path.join(meta_save_dir, 'cam_dict.pkl'), 'wb') as f:
        pickle.dump(cam_dict, f)

    # --- 条件图生成 ---
    joints, bones_idx, parts = sr.get_skeleton_info(sk_file)
    joints = torch.tensor(joints)
    joints_2d, joints_depth = sr.project_joints(joints, cam_dict['mvp'])
    
    cond_images = []
    for iv in range(num_views):
        sorted_bones_idx = sr.sort_bones_depth(joints_depth[iv], bones_idx)
        canvas = np.zeros((512, 512, 3), dtype=np.uint8) # 使用RGBA
        depth_values = sr.process_depth(joints_depth[iv])
        canvas = sr.draw_ccm_with_depth(canvas, joints, joints_2d[iv], sorted_bones_idx, parts, depth_values)
        img = Image.fromarray(canvas, mode='RGBA').resize((256, 256), resample=Image.Resampling.NEAREST)
        img.save(os.path.join(cond_save_dir, f'cond_{iv}.png'))
        cond_images.append(img)
        
    return cond_images, cam_dict

def generate_samples(pipe, prompt, cond_images, c2w, args):
    """
    调用SKDream Pipeline生成一组多视图图像。
    """
    xm = T.Compose([T.ToTensor()])
    cond_tensors = [xm(np.array(img.convert('RGBA'))) for img in cond_images]
    cond_tensor = torch.stack(cond_tensors, dim=0).unsqueeze(0).to(pipe.device)
    cond_tensor = cond_tensor * 2 - 1

    images_np = pipe(
        prompt=prompt,
        negative_prompt=args.neg_prompt,
        hint=cond_tensor,
        c2ws=c2w.reshape(1, args.num_views, -1).to(pipe.device),
        guidance_scale=7.5,
        controlnet_conditioning_scale=args.cond_scale,
        guess_mode=False,blind_control_until_step=None,output_type="numpy"
    ).images
    
    return (images_np * 255).astype(np.uint8)

def evaluate_alignment(generated_images, cond_images, dino, skalign_model, transform, device):
    """
    计算一组生成图像和条件图之间的平均SKA score。
    """
    imgs_transformed = []
    for img_np in generated_images:
        img_pil = Image.fromarray(img_np)
        mask_pil = remove(img_pil, only_mask=True)
        img_black_bg = Image.new("RGB", img_pil.size, (0, 0, 0))
        img_black_bg.paste(img_pil, mask=mask_pil)
        imgs_transformed.append(transform(img_black_bg))

    conds_transformed = [transform(img.convert("RGB")) for img in cond_images]

    imgs_t = torch.stack(imgs_transformed).to(device)
    conds_t = torch.stack(conds_transformed).to(device)

    with torch.no_grad():
        imgs_ft = skalign_model(dino(imgs_t, return_ret=True)['x_norm_patchtokens'])
        conds_ft = skalign_model(dino(conds_t, return_ret=True)['x_norm_patchtokens'])
        mean_cos = torch.mean(torch.cosine_similarity(imgs_ft, conds_ft, dim=-1))

    return mean_cos.item()

def save_image_set(images, save_dir, rembg_session):
    """
    保存一组图像到指定的文件夹。
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, img_np in enumerate(images):
        img_rgba = remove(img_np, session=rembg_session)
        io.imsave(os.path.join(save_dir, f'gen_{i}.png'), img_rgba)

def main(args):
    # --- 1. 初始化和加载模型 ---
    device = torch.device(f'cuda:{args.gpu}')
    
    print(f"正在加载模型到 GPU:{args.gpu}...")
    pipe = load_skdream_pipeline(
        pretrained_controlnet_name_or_path=args.mvc_ckpt,
        pretrained_model_name_or_path='./ckpt/mvdream-sd21-diffusers-lzq',
        num_views=args.num_views,
        weights_dtype=torch.float16,
        device=device
    )
    
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dino = torch.hub.load('facebookresearch/dinov2','dinov2_vitl14_reg',pretrained=False,source='github')
    dino.load_state_dict(torch.load('ckpt/cosa/dinov2_vitl14_reg4_pretrain.pth'),strict=True)
    dino.eval().to(device)
    
    skalign_model = SkalignModel(1024, 3)
    skalign_model.load_state_dict(torch.load('ckpt/cosa/cosa.pth'))
    skalign_model.eval().to(device)

    rembg_session = new_session("isnet-general-use")
    print(f"模型加载完毕。")

    # --- 2. 准备数据和目录 ---
    os.makedirs(os.path.join(args.output_dir, 'win_mv'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'lose_mv'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'skeleton_d'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'meta'), exist_ok=True)
    
    eval_dict = json.load(open(os.path.join(args.data_dir, 'eval.json'), 'r'))
    
    # 【改动】直接获取所有任务，不再根据rank进行分配
    item_ids_this_process = sorted(list(eval_dict.keys()))
    
    print(f"总共分配到 {len(item_ids_this_process)} 个任务。")

    # --- 3. 主循环 ---
    # 【改动】简化tqdm的描述
    progress_bar = tqdm(item_ids_this_process, desc=f"Processing on GPU {args.gpu}")
    for item_id in progress_bar:
        progress_bar.set_postfix_str(f"ID: {item_id}")
        
        win_dir = os.path.join(args.output_dir, 'win_mv', item_id)
        lose_dir = os.path.join(args.output_dir, 'lose_mv', item_id)
        if os.path.exists(win_dir) and os.path.exists(lose_dir):
            continue
        
        sk_file = os.path.join(args.data_dir, 'cano_sk', f'{item_id}.txt')
        prompt = eval_dict[item_id]['caption']
        
        cond_images, cam_dict = generate_conditions_and_camera(item_id, sk_file, args.data_dir, args.output_dir, args.num_views)

        images_a = generate_samples(pipe, prompt, cond_images, cam_dict['c2w'], args)
        images_b = generate_samples(pipe, prompt, cond_images, cam_dict['c2w'], args)

        score_a = evaluate_alignment(images_a, cond_images, dino, skalign_model, transform, device)
        score_b = evaluate_alignment(images_b, cond_images, dino, skalign_model, transform, device)

        if score_a >= score_b:
            win_images, lose_images = images_a, images_b
        else:
            win_images, lose_images = images_b, images_a

        save_image_set(win_images, win_dir, rembg_session)
        save_image_set(lose_images, lose_dir, rembg_session)
        
    print(f"\n任务完成!")

if __name__ == '__main__':
    parser = ArgumentParser(description="根据分数直接排序生成DPO数据对（单卡版）")
    parser.add_argument('--data_dir', type=str, required=True, help="输入数据目录")
    parser.add_argument('--output_dir', type=str, required=True, help="DPO数据集的输出目录")
    parser.add_argument('--mvc_ckpt', type=str, required=True, help="ControlNet检查点路径")
    parser.add_argument('--gpu', type=int, default=0, help="本进程使用的GPU ID")
    parser.add_argument('--num_views', type=int, default=4, help="每个样本生成的视角数量")
    parser.add_argument('--neg_prompt', type=str, default='', help="负向提示词")
    parser.add_argument('--cond_scale', type=float, default=1.0, help="ControlNet条件强度")
    
    # --- 【改动】移除并行化参数 ---
    # parser.add_argument('--world_size', type=int, required=True, help="总进程数 (例如GPU数量)")
    # parser.add_argument('--rank', type=int, required=True, help="当前进程的编号 (0 到 world_size-1)")
    
    args = parser.parse_args()
    main(args)