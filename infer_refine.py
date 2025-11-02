import subprocess
import os
from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mesh_dir',type=str)
    parser.add_argument('--tile_dir',type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--repeat_num',type=int)
    args = parser.parse_args()
    mesh_dir = args.mesh_dir
    tile_dir = args.tile_dir
    save_dir = args.save_dir
    # folders = os.listdir(mesh_dir)
    folders = [item for item in os.listdir(mesh_dir) if not item.endswith('.txt')]
  
    print("{} objects to refine".format(len(folders)))
    try:
        for folder in folders:
            for i in range(args.repeat_num):
                mesh_path = os.path.join(mesh_dir,folder,f'{folder}_{i}.obj')
                img_path = os.path.join(tile_dir,f'{folder}_{i}')
                out_path = os.path.join(save_dir,f'{folder}_{i}')
                subprocess.run(f'python uv_refine.py --config config/refine.json --base_mesh {mesh_path} --data_dir {img_path} --out_dir {out_path}'.split(),check=True)
    except Exception as e:
        print(e)