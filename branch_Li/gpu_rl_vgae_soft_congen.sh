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
  --vocab-csv /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron_selfies.csv --keep-header --seq-col 1 \
  --max-len 20 --hidden 256 --layers 2 --cell GRU \
  --resume /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gen_selfies_token_h256_acc64_cond_origin_1w_vgae/final_step_1000000.pt \
  --z-source vgae --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_Li/ckpts/vgae_Li_iron_4props.pt \
  --predictor /public/home/users/haoxw/generate_AI/branch_Li/ckpts/chemprop_DPER0_cv5_new_7/fold_0/model_0/model.pt \
  --predictor-stats /public/home/users/haoxw/generate_AI/branch_Li/ckpts/li_predictor_stats.json \
  --targets mu gap cv TPSA --goals max max max max --weights 0.35 0.30 0.25 0.10 --zscore \
  --iters 200000 --batch 64 --lr 2e-5 --cuda \
  --outdir ckpts/gpu_rl_vgae_soft_congen_16 \
  --vgae-mean  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/z_stats/z_mean.npy \
  --vgae-cov   /public/home/users/haoxw/generate_AI/branch_Li/ckpts/z_stats/z_cov.npy  \
  --vgae-zbank /public/home/users/haoxw/generate_AI/branch_Li/ckpts/z_stats/z_bank.npy \
  --vgae-gate off \
  --vgae-warmup 1000 --vgae-weight 0.02 \
  --invalid-penalty -1000.0 \
  --save-samples ckpts/gpu_rl_vgae_soft_congen_16/samples.selfies --sample-every 10 \
  --len-target 20 --len-lambda 0.20 --ha-target 10 --ha-lambda 0.40 \
  --min-len 2 --no-repeat-ngram 3 --repeat-penalty 1.15 \
  --entropy-coef 0.01 --kl-coef 0.08 --eos-bias 4 --top-p 0.85 \
  --unique-bonus 0.10 \
  --dupe-penalty 2.0 --dupe-window 5000 \
  --offline-csv /public/home/users/haoxw/generate_AI/branch_Li/data/Li_selfies_valid.csv \
  --offline-col selfies \
  --pulse-every 1 --pulse-k 10 \
  --pulse-sup-weight 2.0 \
  --sample-hard-gate --sample-report \
  --gate-metric euclid \
  --gate-calibrate-teacher \
  --gate-tighten-q 0.3 \
  --heur-weight 0.0 \
  --teacher-pass-cover 0.95 \
  --dupe-canonical \
  --sample-mode buffer

echo ">>> Training finished. Starting Analysis..."

# =======================
# 3. 分析与绘图 (Analysis & Plotting)
# =======================

# 3.1 Analyze Results
python /public/home/users/haoxw/generate_AI/branch_Li/code/analyze_selfies_sa_reward.py \
  --input /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/samples.selfies \
  --predictor /public/home/users/haoxw/generate_AI/branch_Li/ckpts/chemprop_DPER0_cv5_new_7/fold_0/model_0/model.pt \
  --predictor-stats /public/home/users/haoxw/generate_AI/branch_Li/ckpts/li_predictor_stats.json \
  --targets mu gap cv TPSA \
  --goals max max max max \
  --weights 0.35 0.30 0.25 0.10 \
  --zscore \
  --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_Li/ckpts/vgae_Li_iron_4props.pt \
  --vgae-mean /public/home/users/haoxw/generate_AI/branch_Li/ckpts/z_stats/z_mean.npy \
  --outdir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs


# 3.3 Add RDKit LogP (注意这里保留了您原脚本的 ./code 相对路径，请确保提交任务时在正确目录下)
python ./code/add_rdkit_logp_and_parity.py \
  --csv  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules.csv \
  --out  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --plot /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/TPSA_parity.png \
  --quiet-rdkit

# 3.4 Make Paper Figs
# 注意：这里使用了 ${SLURM_JOB_ID}.out 来自动对应本次任务的日志文件
python /public/home/users/haoxw/generate_AI/branch_Li/code/make_all_paper_figs.py \
    --gen_csv /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules.csv \
    --train_csv /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron.csv \
    --log /public/home/users/haoxw/generate_AI/branch_Li/156831.out \
    --out_dir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/paper_figs_all

## 3.5 Radar Plot
python /public/home/users/haoxw/generate_AI/branch_Li/code/radar_six_axes.py \
  --csv /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --prop-cols mu pred_mu gap pred_gap cv pred_cv TPSA pred_TPSA \
  --manual-range mu 0 9 gap 0.1 0.35 cv 5 45 TPSA 0 60 SA 1 8 Reward 0 2 \
  --outdir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs \
  --units mu Debye gap Hartree cv cal/molK TPSA A^2 SA score Reward a.u. --dedup

# 3.6 Plot B
python /public/home/users/haoxw/generate_AI/branch_Li/code/plt_b.py \
  --train /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron.csv \
  --gen   /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --train-col smiles --gen-col smiles \
  --train-mode smiles --gen-mode smiles \
  --out /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/figure_b_chemspace_gen.png    

# 3.7 Plot C-F
python /public/home/users/haoxw/generate_AI/branch_Li/code/plt_c_f.py \
  --train /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron.csv \
  --gen /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical_with_rdkit.csv \
  --outdir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_16/outputs/ \
  --bins 50

echo ">>> All jobs completed."
