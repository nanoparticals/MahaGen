#!/bin/bash
#SBATCH --job-name=haoxwgpu
##SBATCH --workdir=/public/home/users/chensl/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --error=%j.err              # 输出错误日志
#SBATCH --output=%j.out             # 输出标准日志
#SBATCH --account=users

###
eval "$(conda shell.bash hook)"
source /public/home/users/haoxw/miniconda3/etc/profile.d/conda.sh
conda activate generate310
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
PYBIN=$(python -c "import sys,os; print(os.path.dirname(sys.executable))")
echo $PYBIN
export PYTHONUNBUFFERED=1  
python train_gen_supervised.py --data /public/home/users/haoxw/generate_AI/branch_Li/data/Li_selfies_valid.csv --smiles-col 1 --keep-header --iters 1000000 --hidden 256 --layers 2 --cell GRU --lr 2e-4 --optim adam --accum 64 --clip-grad 1.0 --cuda --max-len 240 --sample-len 200 --outdir ckpts/gen_selfies_token_h256_acc64_cond_origin_1w --sample-every 1000 --save-every 20000 --save-samples ckpts/gen_selfies_token_h256_acc64_cond_origin_1w_vgae/samples.selfies --fresh-log --lr-milestones 400000 900000 1200000 --lr-gamma 0.5 --mode selfies --z-bank /public/home/users/haoxw/generate_AI/branch_Li/ckpts/z_stats/z_bank.npy --cond-scale 1.8 --cond-dim 64 --z-mode align --z-hardneg-prob 0.05 --outdir ckpts/gen_selfies_token_h256_acc64_cond_origin_1w_vgae
