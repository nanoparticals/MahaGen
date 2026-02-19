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
  --vocab-csv /public/home/users/haoxw/generate_AI/branch_OLED/data/opv_selfies_valid.csv --keep-header --seq-col 1 \
  --max-len 80 --hidden 256 --layers 2 --cell GRU \
  --resume /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gen_selfies_token_h256_acc64_cond_origin_1w_vgae/final_step_1000000.pt \
  --z-source vgae --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/vgae_opv_4props.pt \
  --predictor /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/chemprop_DPER0_cv5_new_6/fold_0/model_0/model.pt \
  --predictor-stats /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/OPV_predictor_stats.json \
  --targets homo gap MolMR LogP --goals min min max max --weights 0.35 0.35 0.2 0.10 --zscore \
  --iters 200000 --batch 64 --lr 2e-5 --cuda \
  --outdir ckpts/gpu_rl_vgae_soft_congen_16 \
  --vgae-mean  /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/z_stats/z_mean.npy \
  --vgae-cov   /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/z_stats/z_cov.npy  \
  --vgae-zbank /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/z_stats/z_bank.npy \
  --vgae-gate off \
  --vgae-warmup 1000 --vgae-weight 0.02 \
  --invalid-penalty -1000.0 \
  --save-samples ckpts/gpu_rl_vgae_soft_congen_16/samples.selfies --sample-every 1 \
  --len-target 50 --len-lambda 0.05 --ha-target 50 --ha-lambda 0.08 \
  --min-len 10 --no-repeat-ngram 3 --repeat-penalty 1.3 \
  --entropy-coef 0.01 --kl-coef 0.12 --eos-bias 2 --top-p 0.85 \
  --unique-bonus 0.2 \
  --dupe-penalty 0.2 --dupe-escalate --dupe-window 2000 \
  --offline-csv /public/home/users/haoxw/generate_AI/branch_OLED/data/opv_selfies_valid.csv \
  --offline-col selfies \
  --pulse-every 1 --pulse-k 10 \
  --pulse-sup-weight 1.0 \
  --sample-hard-gate --sample-report \
  --gate-metric euclid \
  --gate-calibrate-teacher \
  --gate-tighten-q 0.3 \
  --heur-weight 0  \
  --teacher-pass-cover 1 \
  --dupe-canonical \
  --sample-mode buffer

# 确保训练完成后再进行下一步
echo "Training finished. Starting Analysis..."

# =======================
# 3. 分析与绘图 (Analysis)
# =======================

# 3.1 Analyze Results
python /public/home/users/haoxw/generate_AI/branch_OLED/code/analyze_selfies_sa_reward.py   --input /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/samples.selfies --predictor /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/chemprop_DPER0_cv5_new_6/fold_0/model_0/model.pt   --predictor-stats /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/OPV_predictor_stats.json  --targets homo gap MolMR LogP --goals min min max max --weights 0.35 0.35 0.2 0.10 --zscore   --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/vgae_opv_4props.pt --vgae-mean /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/z_stats/z_mean.npy   --outdir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs


# 3.3 Add RDKit LogP
python ./code/add_rdkit_logp_and_parity.py \
  --csv  /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules.csv \
  --out  /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --plot /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/logp_parity.png \
  --quiet-rdkit

# 3.4 Make Paper Figs (Log file handled dynamically here)
# 注意：这里将固定的文件名替换为了变量 ${SLURM_JOB_ID}.out
python /public/home/users/haoxw/generate_AI/branch_OLED/code/make_all_paper_figs.py \
    --gen_csv /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules.csv \
    --train_csv /public/home/users/haoxw/generate_AI/branch_OLED/data/opv.csv \
    --log /public/home/users/haoxw/generate_AI/branch_OLED/${SLURM_JOB_ID}.out \
    --out_dir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/paper_figs_all

# 3.5 Radar Plot
python /public/home/users/haoxw/generate_AI/branch_OLED/code/radar_six_axes.py \
  --csv /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --prop-cols homo pred_homo gap pred_gap MolMR pred_MolMR LogP pred_LogP \
  --manual-range homo -7.0 -4.0 gap 3 8 MolMR 0 200 LogP 0 15 SA 1 8 Reward 0 2.5 \
  --outdir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs \
  --units homo eV gap eV MolMR a.u. LogP a.u. SA score Reward a.u. --dedup   

# 3.6 Plot B
python /public/home/users/haoxw/generate_AI/branch_OLED/code/plt_b.py \
  --train /public/home/users/haoxw/generate_AI/branch_OLED/data/opv.csv \
  --gen   /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --train-col smiles --gen-col smiles \
  --train-mode smiles --gen-mode smiles \
  --out /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/figure_b_chemspace_gen.png

# 3.7 Plot C-F
python /public/home/users/haoxw/generate_AI/branch_OLED/code/plt_c_f.py \
  --train /public/home/users/haoxw/generate_AI/branch_OLED/data/opv.csv \
  --gen /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --outdir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/ \
  --bins 50

echo "All jobs completed."