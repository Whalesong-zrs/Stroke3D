#!/usr/bin/env python3
"""Run UV refinement for a directory of reconstructed meshes."""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mesh-dir', '--mesh_dir', dest='mesh_dir', type=Path, required=True)
    parser.add_argument('--tile-dir', '--tile_dir', dest='tile_dir', type=Path, required=True)
    parser.add_argument('--save-dir', '--save_dir', dest='save_dir', type=Path, required=True)
    parser.add_argument('--repeat-num', '--repeat_num', dest='repeat_num', type=int, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/refine.json'))
    args = parser.parse_args()
    if args.repeat_num < 1:
        raise ValueError('repeat_num must be positive')

    script = Path(__file__).resolve().with_name('uv_refine.py')
    folders = sorted(path for path in args.mesh_dir.expanduser().iterdir() if path.is_dir())
    print("{} objects to refine".format(len(folders)))
    for folder in folders:
        for repeat in range(args.repeat_num):
            name = f'{folder.name}_{repeat}'
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    '--config',
                    str(args.config),
                    '--base_mesh',
                    str(folder / f'{name}.obj'),
                    '--data_dir',
                    str(args.tile_dir.expanduser() / name),
                    '--out_dir',
                    str(args.save_dir.expanduser() / name),
                ],
                check=True,
            )
