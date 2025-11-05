import os
from skimage import io
import numpy as np
import torch
from argparse import ArgumentParser
import torchvision.transforms as T
from skalign.model import SkalignModel
import json

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eval_dir',type=str)
    parser.add_argument('--img_dir',type=str)
    parser.add_argument('--num_view',type=int,default=4)
    parser.add_argument('--repeat_num',type=int,default=4)
    # <<< 新增: 添加一个参数来接收迭代步数，用于命名输出文件
    parser.add_argument('--step', type=int, required=True, help="The checkpoint step number, used for naming the output file.")
    args = parser.parse_args()
    
    transform = T.Compose([
        # T.CenterCrop(224),
        T.ToTensor(),
        T.Resize((224,224), interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dino = torch.hub.load('facebookresearch/dinov2','dinov2_vitl14_reg',pretrained=False,source='github')
    dino.load_state_dict(torch.load('ckpt/cosa/dinov2_vitl14_reg4_pretrain.pth'),strict=True)
    # dino_repo_path = '/home/zrs/dinov2-main'  # <--- 修改为您在服务器上的实际路径
    # dino = torch.hub.load(dino_repo_path, 'dinov2_vitl14_reg', pretrained=False, source='local', trust_repo=True)
    
    for param in dino.named_parameters():
        param[1].requires_grad = False
    dino.eval()
    dino.cuda()

    model = SkalignModel(1024,3)
    state_dict = torch.load('ckpt/cosa/cosa.pth')
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    
    eval_dict = json.load(open(args.eval_dir,'r'))
    score_dict = {}
    img_dir = args.img_dir
    repeat_num = args.repeat_num
    view_num = args.num_view
    file_dir_list = os.listdir(img_dir)
    for file_dir in file_dir_list:
        img_list = []
        cond_list = []
        for i in range(repeat_num):
            for j in range(view_num):
                img_list.append(os.path.join(img_dir,file_dir,f'gen_{i}_{j}.png'))
        for j in range(view_num):
            cond_list.append(os.path.join(img_dir,file_dir,f'cond_{j}.png'))
        cond_list = cond_list * repeat_num
        
        imgs = []
        conds = []
        for i in range(len(img_list)):
            img = io.imread(img_list[i])
            mask = img[...,-1]
            mask = np.stack([mask,mask,mask],axis=-1)
            img_black_bg = img[...,:3] * (mask>0)
            imgs.append(transform(img_black_bg))
        for i in range(len(cond_list)):
            cond = io.imread(cond_list[i])[...,:3]
            conds.append(transform(cond))
        
        imgs_t = torch.stack(imgs)
        conds_t = torch.stack(conds)
        imgs_ft = model(dino(imgs_t.cuda(),return_ret=True)['x_norm_patchtokens'])#(B,1024)
        conds_ft = model(dino(conds_t.cuda(),return_ret=True)['x_norm_patchtokens'])#(B,1024)
        mean_cos = torch.mean(torch.cosine_similarity(imgs_ft,conds_ft,dim=-1))
        score_dict[file_dir] = (mean_cos.item())
        print(file_dir,mean_cos.item())

    # --- <<< 修改: 将所有print输出重构为先计算，再统一输出到文件和控制台 ---
    
    # 1. 先计算好所有的分数
    avg_score = np.mean(list(score_dict.values()))
    
    class_buckets = [0,0,0]
    class_counts = [0,0,0]
    subclass_buckets = [0,0,0,0,0]
    subclass_counts = [0,0,0,0,0]
    for k in eval_dict.keys():
        class_buckets[eval_dict[k]['class']] += score_dict[k]
        class_counts[eval_dict[k]['class']] += 1
        if eval_dict[k]['class'] == 0:
            subclass_buckets[eval_dict[k]['sub_class']] += score_dict[k]
            subclass_counts[eval_dict[k]['sub_class']] += 1
            
    class_score = [x/y if y > 0 else 0 for x,y in zip(class_buckets,class_counts)]
    subclass_score = [x/y if y > 0 else 0 for x,y in zip(subclass_buckets,subclass_counts)]
    class_mean = sum(class_score)/len(class_score) if len(class_score) > 0 else 0
    subclass_mean = sum(subclass_score)/len(subclass_score) if len(subclass_score) > 0 else 0

    # 2. 定义输出文件名，保存在当前评估的图片目录下
    output_filename = os.path.join(args.img_dir, f'eval_results_step_{args.step}.txt')

    # 3. 将结果写入文件
    with open(output_filename, 'w') as f:
        f.write(f'Evaluation Summary for Checkpoint Step: {args.step}\n')
        f.write('====================================================\n\n')
        
        f.write(f'Average alignment score: {avg_score:.4f}\n\n')
        
        f.write('--- Scores by Class ---\n')
        f.write(f'Class scores: {["{:.4f}".format(s) for s in class_score]}\n')
        f.write(f'Class mean: {class_mean:.4f}\n\n')
        
        f.write('--- Scores by Subclass ---\n')
        f.write(f'Subclass scores: {["{:.4f}".format(s) for s in subclass_score]}\n')
        f.write(f'Subclass mean: {subclass_mean:.4f}\n\n')
        
        f.write('--- Individual Scores ---\n')
        for file_dir, score in sorted(score_dict.items()):
            f.write(f'{file_dir}: {score:.4f}\n')

    # 4. 同时也在控制台打印简洁的总结和保存路径
    print("\n====================================================")
    print(f"Evaluation results have been saved to: {output_filename}")
    print('Average alignment score: {:.4f}'.format(avg_score))
    print('Class mean: {:.4f}'.format(class_mean))
    print('Subclass mean: {:.4f}'.format(subclass_mean))
    print("====================================================")