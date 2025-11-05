import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)


from transformers import CLIPTokenizer, CLIPTextModel

from utils.misc import load_config, seed_all
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sample/save_climb.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="vis/diff", help="Directory for saving visualization images.")
    parser.add_argument("--skeleton_output_base_dir", type=str, default="sample_sk/sample_sk_diff", help="Base directory for saving generated skeleton files.")

    args = parser.parse_args()

    # === Load YAML ===
    cfg = load_config(args.config)
    cfg_name = Path(args.config).stem

    print(f"Loaded configuration from: {args.config}")
    print(f"Using device: {args.device}")
    
    # Set seet
    seed_all(cfg.val.seed)

    # === Data ===
    featurizer = FeaturizeGraph(use_rotate=False)
    transform = Compose([
        featurizer,  
    ])

    data_mode = cfg.val.data_mode
    val_dataset = get_dataset(
        config = cfg.dataset,
        transform = transform,
        mode=data_mode
    )
    print(f"Clip_emb dataset size: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            follow_batch=featurizer.follow_batch)
    
    # === CLIP Loading (Freeze) ===
    print("Initializing CLIP model for captions...")
    clip_ckpt_path = cfg.model.clip.get('ckpt_path', None)
    if not clip_ckpt_path or not os.path.exists(clip_ckpt_path):
        clip_ckpt_path = None
    clip_model_name = 'openai/clip-vit-large-patch14'
    
    tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name).to(args.device)

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    print("CLIP loaded and frozen.")
    null_inputs = tokenizer("", return_tensors="pt", padding='max_length', truncation=True)
    null_ids = null_inputs["input_ids"].to("cuda")
    null_attention_mask = null_inputs["attention_mask"].to('cuda')
    null_outputs = text_encoder(input_ids=null_ids, attention_mask=null_attention_mask)
    null_emb = null_outputs.last_hidden_state

    save_root = cfg.dataset.clip_emb_root
    print(f"Embeddings will be saved to: {save_root}")
    os.makedirs(save_root, exist_ok=True) # Ensure the output directory exists

    null_save_path = os.path.join(save_root, 'null_emb.pt')
    torch.save(null_emb.cpu(), null_save_path)

    view_name_map = {
        "xy": "front view",
        "yz": "side view",
        "xz": "top view"
    }

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            uuid = batch.uuid[0]
            caption = batch.caption[0]
            unclosed_caption = caption.replace('.', '')

            for view in (None, "xy", "yz", "xz"):
                if view is not None:
                    caption = f"{unclosed_caption}, {view_name_map[view]}."
                    save_path = os.path.join(save_root, f"{uuid}_{view}.pt")
                else:
                    caption = f"{unclosed_caption}."
                    save_path = os.path.join(save_root, f"{uuid}.pt")

                text_inputs = tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True)
                input_ids = text_inputs["input_ids"].to(args.device)
                attention_mask = text_inputs["attention_mask"].to(args.device) 
                outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embedding = outputs.last_hidden_state

                torch.save(text_embedding.cpu(), save_path)
                # print(f"Saved embedding of '{caption}' to {save_path}")
