import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle
import numpy as np
from PIL import Image
import json
import io
import lmdb
# 正常使用
from skdream.utils.camera import get_camera,create_camera_to_world_matrix
# for debug
# from utils.camera import get_camera,create_camera_to_world_matrix

class SKDreamDatasetDPO(Dataset):
    def __init__(self, root_dir, tokenizer,p_simple_prompts=0,cond_channels=4,
                 transform=None,cond_transform=None,cvt_cam=True,use_lmdb=False,use_filtered_data=False):
        # use_lmdb=False use_filtered_data=False
        self.use_lmdb = use_lmdb  # 待修改 用不到
        if use_lmdb:
            self.lmdb_dir = [os.path.join(root_dir,'objsk_db0'),os.path.join(root_dir,'objsk_db1')]
        # 待定
        if use_filtered_data:  # 待修改 用不到
            self.file_ids = json.load(open(root_dir+'/train_eval.json'))['filtered_train']
        else:
            self.file_ids = json.load(open(root_dir+'/train_eval.json'))['train']
        
        use_gemini_cap = False  # 待修改 用不到
        if use_gemini_cap:
            gemini_cap_dict = json.load(open(root_dir+'/gemini_view0.json'))
            print(len(gemini_cap_dict.keys()))
            captions = []
            simple_captions = []
            for id in self.file_ids:
                value = gemini_cap_dict[str(id)+'_0']
                if isinstance(value,list):
                    cap = value[0]+','+value[1]
                    cap_simple = value[0]
                elif isinstance(value,str):
                    cap = value[2:-1].replace('\'','')
                    cap_simple = cap.split(',')[0]
                else:
                    print(id)
                    raise ValueError
                captions.append(cap+', 3d assets')
                simple_captions.append(cap_simple)
        else:
            caption_dict = json.load(open(root_dir+'/dpo_texturig_captions.json'))
            captions = [caption_dict[k] for k in self.file_ids]
            simple_captions = captions
        self.captions = captions
        self.simple_captions = simple_captions
        print(self.captions[:3],self.simple_captions[:3])

        self.lose_img_dir = os.path.join(root_dir, 'lose_mv')
        self.win_img_dir = os.path.join(root_dir, 'win_mv')

        self.cond_dir = os.path.join(root_dir, 'skeleton_d')
        self.meta_dir = os.path.join(root_dir, 'meta')

        self.transform = transform
        self.cond_transform = cond_transform
        self.p_simple_caption = p_simple_prompts
        print('tokenzing captions, this may take a while...')
        tokenized = self.tokenize_captions(tokenizer,self.captions) # 因为是dpo,不需要加""
        tokenized_simple = self.tokenize_captions(tokenizer,self.simple_captions)
        self.input_ids = tokenized
        self.empty_id = None
        self.simple_ids = tokenized_simple
        self.azimuth_indices = None  # 不一定用得上
        self.cond_channels = cond_channels
        self.cvt_cam = cvt_cam
    
    def __len__(self):
        return len(self.file_ids)
    
    def set_azimuth_indices(self,indices):
        self.azimuth_indices = indices
  
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        input_id = self.input_ids[idx] if np.random.rand()>self.p_simple_caption else self.simple_ids[idx]  # p_simple is always 0
        # 读相机
        meta_info = pickle.load(open(os.path.join(self.meta_dir, str(file_id), 'cam_dict.pkl'), 'rb'))


        # 读lose win 数据
        lose_img_files = [os.path.join(self.lose_img_dir, str(file_id), f"gen_{x}.png") for x in range(4)]
        win_img_files = [os.path.join(self.win_img_dir, str(file_id), f"gen_{x}.png") for x in range(4)]

        cond_files = [os.path.join(self.cond_dir, str(file_id), f"cond_{x}.png") for x in range(4)] 

        lose_img_4 = [Image.open(file) for file in lose_img_files]
        win_img_4 = [Image.open(file) for file in win_img_files]
        cond_4 = [Image.open(file) for file in cond_files]
        def set_bkg(image, value=(128, 128, 128, 0)):
            data = np.array(image)

            new_background = np.array(value)

            mask = data[:, :, 3] == 0
            data[mask] = new_background
            new_image = Image.fromarray(data)
            return new_image
        
        lose_img_4 = [set_bkg(img).convert('RGB') for img in lose_img_4]
        win_img_4 = [set_bkg(img).convert('RGB') for img in win_img_4]
        # 可以在这个地方debug一次
        for i in range(4):
            lose_img_4[i].save(f'/home/zrs/test_vis/lose_rgb{i}.png')
            win_img_4[i].save(f'/home/zrs/test_vis/win_rgb{i}.png')
            cond_4[i].save(f'/home/zrs/test_vis/rgb_cond{i}.png')
        # exit()
        def rgb2binary(image,return_array=False):
            gray_array = np.array(image.convert('RGB').convert('L'))
            threshold = 1  # 任何非黑色的像素值都会大于 0
            binary_array = np.where(gray_array > threshold, 1, 0).astype(np.uint8)
            if return_array:
                return np.expand_dims(binary_array,axis=-1) * 255
            new_image = Image.fromarray(binary_array * 255)
            return new_image
        
        if self.cond_channels == 4:
            pass
        elif self.cond_channels == 3:
            cond_4 = [cond.convert('RGB') for cond in cond_4]
        elif self.cond_channels == 1:
            # cond_4[0].save('cond_rgb.png')
            cond_4 = [rgb2binary(cond) for cond in cond_4]
            # cond_4[0].save('cond_grey.png')
        elif self.cond_channels == 5:
            # rgba + binary, directly return array
            cond_binary = [rgb2binary(cond,return_array=True) for cond in cond_4]
            cond_4 = [np.concatenate([c1,c2],axis=-1).astype(np.uint8) for c1,c2 in zip(cond_4,cond_binary)]
        
        # if self.transform:
        #     for i in range(len(img_4)):
        #         img_4[i] = self.transform(img_4[i])
        # if self.cond_transform:
        #     for i in range(len(cond_4)):
        #         cond_4[i] = self.transform(cond_4[i])
        # image = torch.stack(img_4,dim=0) # (4,3,256,256)
        # condition = torch.stack(cond_4,dim=0) # (4,4,256,256)

        if self.transform:
            for i in range(len(lose_img_4)):
                lose_img_4[i] = self.transform(lose_img_4[i])
            for i in range(len(win_img_4)):
                win_img_4[i] = self.transform(win_img_4[i])
        if self.cond_transform:
            for i in range(len(cond_4)):
                cond_4[i] = self.transform(cond_4[i])
        lose_image = torch.stack(lose_img_4, dim=0) # (4, 3, 256, 256)
        win_image = torch.stack(win_img_4, dim=0)
        condition = torch.stack(cond_4, dim=0)

        # Image.fromarray(((condition[0,0].numpy()/2+0.5)*255).astype(np.uint8)).save('cond_tensor.png')

        if self.cvt_cam:
            cam_4 = []
            # for az_i in azimuth_indices:
            #     elevation = meta_info['cam'+str(elevation_idx)][az_i]['elevation']
            #     azimuth = meta_info['cam'+str(elevation_idx)][az_i]['azimuth']
            #     cam = create_camera_to_world_matrix(elevation,azimuth,1) # distance is set as 1
            #     cam_4.append(torch.tensor(cam))
            for i in range(4):
                elevation = meta_info['elevation'][i]
                azimuth = meta_info['azimuth'][i]
                cam = create_camera_to_world_matrix(elevation,azimuth,1) # distance=1 as mvdream
                cam_4.append(torch.tensor(cam))
        else:
            cam_4 = [torch.tensor(meta_info['cam'+str(elevation_idx)][x]['mv']) for x in azimuth_indices]
        camera = torch.stack(cam_4,dim=0) # (4,4,4)

        data_dict = {}
        # data_dict["pixel_values"] = image
        data_dict["lose_image"] = lose_image
        data_dict["win_image"] = win_image

        data_dict["conditioning_pixel_values"] = condition
        data_dict["cameras"] = camera
        data_dict["input_ids"] = input_id
        data_dict["captions"] = self.captions[idx]
        data_dict["file_ids"] = file_id
        return data_dict
    @staticmethod
    def tokenize_captions(tokenizer,captions):
        inputs = tokenizer(
            text=captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return inputs.input_ids

import torch
def collate_fn(examples):

    # 原本
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # import pdb; pdb.set_trace()
    
    print(examples[0]["lose_image"].shape)
    print(type(examples[0]["lose_image"]))
    # DPO
    lose_values = torch.stack([example["lose_image"] for example in examples])
    lose_values = lose_values.to(memory_format=torch.contiguous_format).float()

    win_values = torch.stack([example["win_image"] for example in examples])
    win_values = win_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    cameras = torch.stack([example["cameras"] for example in examples])
    cameras = cameras.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])
    captions = [example["captions"] for example in examples]
    file_ids = [example["file_ids"] for example in examples]

    return {
        # "pixel_values": pixel_values, # 原本
        "lose_values": lose_values,
        "win_values": win_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "cameras": cameras,
        "input_ids": input_ids,
        "captions": captions,
        "file_ids": file_ids,
    }
    
if __name__ == "__main__":
    from transformers import AutoTokenizer, PretrainedConfig
    from torchvision import transforms

    
    tokenizer = AutoTokenizer.from_pretrained(
            '/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq',
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x:x*2-1), #(0,1)->(-1,1)
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x:x*2-1), #(0,1)->(-1,1)
        ]
    )

    dataset = SKDreamDatasetDPO(
        root_dir = '/home/zrs/mix_data/dpo_dataset',
        p_simple_prompts=0,
        cond_channels=4,
        tokenizer=tokenizer,
        transform=image_transforms,
        cond_transform=conditioning_image_transforms,
        use_lmdb=False,
        use_filtered_data=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=3,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    for step, batch in enumerate(train_dataloader):
        print(batch['lose_values'])
        print(batch['lose_values'].shape)
        # import pdb; pdb.set_trace()

    print(dataset[0])
    # import pdb; pdb.set_trace()