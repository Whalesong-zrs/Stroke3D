#!/bin/bash
__conda_setup="$('/root/autodl-fs-data3/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/autodl-fs-data3/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/autodl-fs-data3/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/autodl-fs-data3/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate base
echo "python path: $(which python)"

cd SkDiff
python scripts/train_parallel.py