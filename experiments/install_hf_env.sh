#!/usr/bin/env bash
# Robust conda env creation + package install on 5090.
# Run:  nohup bash install_hf_env.sh > /tmp/install_hf_env.log 2>&1 &
set -uo pipefail
export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"

echo "=== [$(date)] start ==="
echo "PATH=$PATH"
echo "which conda: $(which conda)"
echo "conda version: $(conda --version)"

# Ensure ToS accepted
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 | tail -1
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>&1 | tail -1

# Force-skip the interactive channel prompt by setting auto-yes
conda config --set always_yes true 2>&1 | tail -1
conda config --set auto_update_conda false 2>&1 | tail -1

# Step 1: create env (verbose, full output)
if $HOME/miniconda3/envs/llava/bin/python --version 2>/dev/null; then
    echo "[skip] env llava exists"
else
    echo "[step] conda create -n llava python=3.12"
    conda create -n llava python=3.12 -y --override-channels -c defaults 2>&1 | tail -20
    echo "[check] env list:"
    conda env list
fi

# Step 2: pytorch
if $HOME/miniconda3/envs/llava/bin/python -c "import torch" 2>/dev/null; then
    echo "[skip] torch installed"
else
    echo "[step] conda install pytorch"
    conda install -n llava pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y 2>&1 | tail -20
fi

# Step 3: transformers
if $HOME/miniconda3/envs/llava/bin/python -c "import transformers" 2>/dev/null; then
    echo "[skip] transformers installed"
else
    echo "[step] pip install transformers + deps"
    $HOME/miniconda3/envs/llava/bin/pip install transformers accelerate pillow sentencepiece 2>&1 | tail -20
fi

# Step 4: verify
echo "=== verify ==="
$HOME/miniconda3/envs/llava/bin/python -c "
import torch, transformers
print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'transformers {transformers.__version__}')
print('OK')
"
echo "=== [$(date)] DONE ==="
