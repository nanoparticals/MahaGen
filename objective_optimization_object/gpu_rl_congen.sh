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
python train_rl_multiobj.py \
  --vocab-csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/aug_brics_1w.csv --keep-header --seq-col 1 \
  --max-len 100  --hidden 256 --layers 2 --cell GRU \
  --resume /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gen_selfies_token_h256_acc64_cond_origin_1w_vgae/step_1000000.pt \
  --z-dim 16 --z-source vgae --vgae-ckpt ckpts/vgae_li2022_DPEr0_z16.pt \
  --predictor /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/chemprop_DPER0_cv5_new_4/fold_2/model_14/model.pt \
  --predictor-stats ckpts/et_stats_DPER0.json \
  --targets D P EG r0 --goals max max max max --weights 0.35 0.30 0.25 0.10 --zscore \
  --iters 10000 --batch 10 --lr 2e-5 --cuda \
  --outdir ckpts/gpu_rl_congen \
  --vgae-mean ckpts/z_stats_16/z_mean.npy \
  --vgae-cov  ckpts/z_stats_16/z_cov_fixed.npy \
  --vgae-zbank ckpts/z_stats_16/z_bank.npy \
  --vgae-weight 0 \
  --vgae-gate off \
  --vgae-warmup 2000 \
  --vgae-anneal cosine:0.10,0.02 \
  --invalid-penalty -2.0 \
  --save-samples ckpts/gpu_rl_congen/samples.selfies --sample-every 1 \
  --len-target 50 --len-lambda 0.05 --ha-target 50 --ha-lambda 0.08 \
  --min-len 10 --no-repeat-ngram 4 --repeat-penalty 1.6 \
  --entropy-coef 0.02 --kl-coef 0.08 --eos-bias 2 --top-p 0.9 \
  --unique-bonus 0.4 \
  --dupe-penalty 0.6 \
  --dupe-escalate \
  --dupe-window 400\
  --reward-tighten-q 0.50 


