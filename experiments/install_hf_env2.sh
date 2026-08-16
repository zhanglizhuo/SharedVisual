#!/usr/bin/env bash
# Install PyTorch + HF transformers on 5090 into conda env 'llava' (already created).
# Uses aliyun mirror (tsinghua is 403-blocked for some packages).
# Idempotent + backgroundable:  nohup bash install_hf_env2.sh > /tmp/install_hf_env2.log 2>&1 &
set -uo pipefail
PY=$HOME/miniconda3/envs/llava/bin/python
PIP=$HOME/miniconda3/envs/llava/bin/pip
MIRROR=https://mirrors.aliyun.com/pypi/simple/

echo "=== [$(date)] start install_hf_env2.sh ==="

# Step 1: torch + torchvision via pip (CPU+CUDA wheel from aliyun)
if $PY -c "import torch" 2>/dev/null; then
    echo "[skip] torch already installed"
else
    echo "[step] pip install torch torchvision (aliyun)"
    $PIP install --no-input torch torchvision -i $MIRROR 2>&1 | tail -15
fi

# Step 2: transformers + accelerate + pillow + sentencepiece
if $PY -c "import transformers, accelerate, PIL, sentencepiece" 2>/dev/null; then
    echo "[skip] transformers stack already installed"
else
    echo "[step] pip install transformers accelerate pillow sentencepiece (aliyun)"
    $PIP install --no-input transformers accelerate pillow sentencepiece -i $MIRROR 2>&1 | tail -15
fi

# Step 3: verify
echo "=== verify ==="
$PY -c "
import torch, transformers
print(f'torch {torch.__version__}, CUDA avail {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'transformers {transformers.__version__}')
print('INSTALL_OK')
"
echo "=== [$(date)] DONE ==="
