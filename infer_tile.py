import torch
import numpy as np
import cv2
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
import numpy as np
from PIL import Image
import pickle
from argparse import ArgumentParser
import json
from rembg import remove,new_session
from glob import glob
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--num_view',type=int,default=4)
    parser.add_argument('--neg_prompt',type=str,default='')
    parser.add_argument('--cond_scale',type=float,default=1.0)
    parser.add_argument('--repeat_num',type=int,default=1)
    parser.add_argument('--gpu',type=int,default=0)
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    # data_dir = 'objsk_eval'
    # eval_dict = json.load(open(os.path.join(data_dir,'eval.json'),'r'))
    data_dir = '/home/zrs/skdream_model/objsk_eval2'
    eval_dict = json.load(open(os.path.join(data_dir,'eval.json'),'r'))
    
    # Generate images
    if args.neg_prompt == 'default':
        neg_prompt = 'low poly,white model,noise,strange color,ugly, oversaturated,doubled face, b&w,sepia, freckles, paintings, sketches, worst quality,low quality, lowres, monochrome, grayscale,error, blurry,artifacts,'
    elif args.neg_prompt == '':
        neg_prompt = None
    else:
        neg_prompt = args.neg_prompt
    
    # initialize the models and pipeline
    controlnet = [ControlNetModel.from_pretrained(
                    "ckpt/control_v11f1e_sd15_tile", torch_dtype=torch.float16,
                    local_files_only=True,allow_pickle=True),
                ControlNetModel.from_pretrained("ckpt/sd-controlnet-canny", torch_dtype=torch.float16),]
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "ckpt/stable-diffusion-v1-5", custom_pipeline="stable_diffusion_controlnet_img2img",controlnet=controlnet, 
        torch_dtype=torch.float16,local_files_only=True,allow_pickle=True,requires_safety_checker=False,safety_checker=None
    ).to(device)
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    # # get canny image
    def get_canny(image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image
    
    # cond_list = sorted(glob(os.path.join(args.save_dir,'*')))
    all_files = glob(os.path.join(args.save_dir, '*'))
    cond_list = sorted([f for f in all_files if not f.endswith('.txt')])
    
    num_view = args.num_view
    view_prompt = ['front','side','back','side']
    # rembg_session1 = new_session("u2net")
    # rembg_session2 = new_session("is_general_use")
    # rembg_session3 = new_session("sam")
    def view_prompt_idx(angle):
        if angle < 45 or angle > 315:
            return 0
        elif angle >= 45 and angle <= 135:
            return 1
        elif angle > 135 and angle < 225:
            return 2
        else:
            return 3
    
    for idx,folder in enumerate(cond_list):
        print(folder,f"{idx}/{len(cond_list)}")
        cam_dict = pickle.load(open(os.path.join(folder,'cam_dict.pkl'),'rb'))
        azimuths = cam_dict['azimuth']
        prompt = eval_dict[folder.split('/')[-1]]['caption'].split(',')[0]
        
        for r in range(args.repeat_num):
            for i in range(num_view):
                cur_prompt = prompt + f",{view_prompt[view_prompt_idx(azimuths[i])]} view"
                print(cur_prompt)
                print(folder)
                image = Image.open(folder+f'/gen_{r}_{i}.png').convert('RGB').resize((1024,1024),1)
                pred_image = pipe(
                            cur_prompt, controlnet_conditioning_scale=[1.0,0.5], image=image,num_inference_steps=20,
                            guidance_scale=7.5,negative_prompt=neg_prompt,
                            controlnet_conditioning_image=[image,get_canny(image)],width=image.size[0],height=image.size[1]
                        ).images[0]
                pred_image.save(os.path.join(folder,f'tile_{r}_{i}.png'))
    
