# Runner for exp_flowers102_difficulty.py on V100 (broadsense@192.168.20.35).
# V100 has NO external network: fully offline. 5 of 6 encoder features are
# already cached in ~/works/lzhuo/SharedVisual/results/flowers_*.npz from the
# geometry pilot; only ResNet-50 gets extracted (torchvision weights cached).
# Launch remotely with:
#   ssh broadsense@192.168.20.35 "nohup bash run_flowers102_difficulty_v100.sh > /tmp/run_v1.log 2>&1 &"
set -e
cd ~/works/lzhuo/SharedVisual
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl_env
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] starting"
python -u experiments/exp_flowers102_difficulty.py
echo "[$(date '+%F %T')] done rc=$?"