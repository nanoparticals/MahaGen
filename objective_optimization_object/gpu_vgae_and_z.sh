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
python train_vgae_ref2.py   --input Data_origin.csv   --smiles-col 0 --keep-header   --label-cols D,P,EG,r0   --target-weights 0.45,0.25,0.20,0.10   --prop-weight 1.0 --zscore   --epochs 1200 --batch-size 64   --z-dim 16 --hidden 128 --layers 2 --max-nodes 72   --recon-weight 3.0 --kl-weight 0.8 --kl-burnin 50 --kl-warmup 150   --lr 5e-4 --weight-decay 0.0   --device cuda   --save ckpts/vgae_z16.pt   --log-csv ckpts/vgae_z16_log.csv
#python train_vgae_ref2.py \
 # --input Data_origin.csv \
  #--smiles-col 0 --keep-header \
  #--epochs 1200 --batch-size 64 \
  #--z-dim 16 --hidden 128 --layers 2 --max-nodes 72 \
  #--recon-weight 3.0 --kl-weight 0.8 --kl-burnin 50 --kl-warmup 150 \
  #--lr 5e-4 --weight-decay 0.0 \
  #--device cuda \
  #--save ckpts/vgae_struct_only_z16.pt \
  #--log-csv ckpts/vgae_struct_only_z16_log.csv'''

#python make_z_bank.py --smiles-csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/aug_brics_1w.csv --keep-header --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/z_stats_gen_origin_1w  --vgae-ckpt /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/vgae_struct_only_z16.pt



