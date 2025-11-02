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

class SKDreamDataset(Dataset):
    def __init__(self, root_dir, tokenizer,p_simple_prompts=0,cond_channels=4,
                 transform=None,cond_transform=None,cvt_cam=True,use_lmdb=False,use_filtered_data=False):
        # use_lmdb=False use_filtered_data=False
        self.use_lmdb = use_lmdb
        if use_lmdb:
            self.lmdb_dir = [os.path.join(root_dir,'objsk_db0'),os.path.join(root_dir,'objsk_db1')]
            # self.lmdb_envs = []
            # self.lmdb_txns = []
            # if isinstance(lmdb_dir,list):
            #     for db_path in lmdb_dir:
            #     # db_path = lmdb_dir[0]
            #         env = lmdb.open(db_path,readonly=True)
            #         self.lmdb_envs.append(env)
            #         self.lmdb_txns.append(env.begin(write=False))
            # else:
            #     env = lmdb.open(db_path,readonly=True)
            #     self.lmdb_envs.append(env)
        
        if use_filtered_data:
            self.file_ids = json.load(open(root_dir+'/train_eval.json'))['filtered_train']#9548
        else:
            self.file_ids = json.load(open(root_dir+'/train_eval.json'))['train']
        
        
        use_gemini_cap = False
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
        # else:
        #     df = pd.read_csv(root_dir+'/Cap3d_captions.csv')
        #     self.file_ids = df.iloc[:,0].tolist()
        #     _captions = df.iloc[:,1].tolist()
        #     captions = [c.replace('"','').replace('.','')+', 3d assets' for c in _captions]
        #     simple_captions = ['' for c in captions]
        else:
            # xyy version
            # caption_dict = json.load(open(root_dir+'/Cap3d_captions.json'))
            caption_dict = json.load(open(root_dir+'/mixed_captions.json'))
            captions = [caption_dict[k] for k in self.file_ids]
            simple_captions = captions
        self.captions = captions
        self.simple_captions = simple_captions
        print(self.captions[:3],self.simple_captions[:3])
        # self.img_dir = os.path.join(root_dir,'rendered')
        self.img_dir = os.path.join(root_dir,'rgb_new')
        self.cond_dir = os.path.join(root_dir,'skeleton_d')
        self.meta_dir = os.path.join(root_dir,'meta')
        self.transform = transform
        self.cond_transform = cond_transform
        self.p_simple_caption = p_simple_prompts
        print('tokenzing captions, this may take a while...')
        tokenized = self.tokenize_captions(tokenizer,self.captions+[""])
        tokenized_simple = self.tokenize_captions(tokenizer,self.simple_captions)
        self.input_ids = tokenized[:-1]
        self.empty_id = tokenized[-1]
        self.simple_ids = tokenized_simple
        self.azimuth_indices = None
        self.cond_channels = cond_channels
        self.cvt_cam = cvt_cam
        # self.id_class = json.load(open(root_dir+'/id_class.json','r'))

    def __len__(self):
        return len(self.file_ids)
    
    def set_azimuth_indices(self,indices):
        self.azimuth_indices = indices
  
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        input_id = self.input_ids[idx] if np.random.rand()>self.p_simple_caption else self.simple_ids[idx]
        meta_info = pickle.load(open(self.meta_dir+'/{}.pkl'.format(str(file_id)),'rb'))
        elevation_num = len([x for x in meta_info.keys() if 'cam' in x]) # 4
        elevation_idx = np.random.randint(0,elevation_num) # 0, 1, 2, 3
        
        azimuth_num = len(meta_info['cam'+str(elevation_idx)]) # 32
        if self.azimuth_indices is not None: #for fixed azim
            azimuth_idx = self.azimuth_indices[idx]
        else:
            azimuth_idx = np.random.randint(0,azimuth_num)
        # anti-clock wise
        azimuth_gap = azimuth_num//4 # 按照正交的方式去选4个正交方位角
        azimuth_indices = [(azimuth_idx)%azimuth_num,(azimuth_idx+azimuth_gap)%azimuth_num,
                           (azimuth_idx+azimuth_gap*2)%azimuth_num,(azimuth_idx+azimuth_gap*3)%azimuth_num,]
        
        img_files = [os.path.join(self.img_dir,str(file_id),"{}_{}_{}_rgb.png".format(str(file_id),elevation_idx,x)) for x in azimuth_indices]
        # img_files = [os.path.join(self.img_dir,str(file_id),f"{str(elevation_idx*32+x+1).zfill(4)}.png") for x in azimuth_indices]
        cond_files = [os.path.join(self.cond_dir,str(file_id),"{}_{}_{}_sk.png".format(str(file_id),elevation_idx,x)) for x in azimuth_indices]
        if self.use_lmdb:
            shard_id = self.id_class[file_id][1]
            with lmdb.open(self.lmdb_dir[shard_id],readonly=True, lock=False) as env:
                with env.begin() as txn:
                    img_4 = [Image.open(io.BytesIO(txn.get(str(os.path.basename(file)).encode('utf-8')))) for file in img_files]
                    cond_4 = [Image.open(io.BytesIO(txn.get(str(os.path.basename(file)).encode('utf-8')))) for file in cond_files]
        else:
            img_4 = [Image.open(file) for file in img_files] # discard alpha channel
            cond_4 = [Image.open(file) for file in cond_files]
        
        def set_bkg(image,value=(128,128,128,0)):
            data = np.array(image)
            # Define the new background color (RGB)
            new_background = np.array(value)
            # Create a mask where the alpha channel is 0 (fully transparent)
            mask = data[:, :, 3] == 0
            # Apply the new background color where the mask is True
            data[mask] = new_background
            # Convert the numpy array back to an image
            new_image = Image.fromarray(data)
            return new_image
        img_4 = [set_bkg(img).convert('RGB') for img in img_4]
        # img_4[0].save('rgb1.png')
        # cond_4[0].save('rgb1_cond.png')
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
        
        if self.transform:
            for i in range(len(img_4)):
                img_4[i] = self.transform(img_4[i])
        if self.cond_transform:
            for i in range(len(cond_4)):
                cond_4[i] = self.transform(cond_4[i])
        image = torch.stack(img_4,dim=0) # (4,3,256,256)
        condition = torch.stack(cond_4,dim=0) # (4,4,256,256)
        # Image.fromarray(((condition[0,0].numpy()/2+0.5)*255).astype(np.uint8)).save('cond_tensor.png')


        def convert_cam(cam):
            cam = torch.linalg.inv(cam)
            new_cam = torch.zeros_like(cam)
            new_cam[:,0,:] = cam[:,2,:]
            new_cam[:,1,:] = cam[:,0,:]
            new_cam[:,2,:] = cam[:,1,:]
            new_cam[:,3,:] = cam[:,3,:]
            return new_cam
        if self.cvt_cam:
            cam_4 = []
            for az_i in azimuth_indices:
                elevation = meta_info['cam'+str(elevation_idx)][az_i]['elevation']
                azimuth = meta_info['cam'+str(elevation_idx)][az_i]['azimuth']
                cam = create_camera_to_world_matrix(elevation,azimuth,1) # distance is set as 1
                cam_4.append(torch.tensor(cam))
        else:
            cam_4 = [torch.tensor(meta_info['cam'+str(elevation_idx)][x]['mv']) for x in azimuth_indices]
        camera = torch.stack(cam_4,dim=0) # (4,4,4)
        
        data_dict = {}
        data_dict["pixel_values"] = image
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
    
    dataset = SKDreamDataset(
        root_dir = '/data04/xyy/3D/objaverse/objaverse_sk',
        p_simple_prompts=0,
        cond_channels=4,
        tokenizer=tokenizer,
        transform=image_transforms,
        cond_transform=conditioning_image_transforms,
        use_lmdb=False,
        use_filtered_data=False
    )
    # import pdb; pdb.set_trace()