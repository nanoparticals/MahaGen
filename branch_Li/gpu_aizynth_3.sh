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
python /public/home/users/haoxw/generate_AI/branch_Li/code/analyze_selfies_sa_reward.py   --input /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/samples.selfies --predictor //public/home/users/haoxw/generate_AI/branch_Li/ckpts/chemprop_DPER0_cv5_new_4/fold_0/model_0/model.pt   --predictor-stats /public/home/users/haoxw/generate_AI/branch_Li/ckpts/li_predictor_stats.json  --targets mu gap cv TPSA --goals max max max max --weights 0.35 0.30 0.25 0.10 --zscore   --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_Li/ckpts/vgae_Li_iron_4props.pt --vgae-mean /public/home/users/haoxw/generate_AI/branch_Li/ckpts/z_stats/z_mean.npy   --outdir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs
python /public/home/users/haoxw/generate_AI/branch_Li/code/unscale_properties.py \
  --csv  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules.csv \
  --stats /public/home/users/haoxw/generate_AI/branch_Li/ckpts/li_predictor_stats.json \
  --out  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules_physical.csv
python ./code/add_rdkit_logp_and_parity.py \
  --csv  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules_physical.csv \
  --out  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules_physical_with_rdkit.csv \
  --plot /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/TPSA_parity.png \
  --quiet-rdkit
python /public/home/users/haoxw/generate_AI/branch_Li/code/make_all_paper_figs.py \
    --gen_csv /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules_physical.csv \
    --train_csv /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron.csv \
    --log /public/home/users/haoxw/generate_AI/branch_Li/154322.out \
    --out_dir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/paper_figs_all
python /public/home/users/haoxw/generate_AI/branch_Li/code/radar_six_axes.py \
  --csv /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules_physical_with_rdkit.csv \
  --prop-cols mu mu_phys gap gap_phys cv cv_phys TPSA TPSA_phys \
  --manual-range mu 0 9 gap 0.1 0.35 cv 5 45 TPSA 0 60 SA 1 8 Reward 0 2 \
  --outdir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs \
  --units mu Debye gap Hartree cv cal/molK TPSA A^2 SA score Reward a.u. --dedup
python /public/home/users/haoxw/generate_AI/branch_Li/code/plt_b.py \
  --train /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron.csv \
  --gen   /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules_physical_with_rdkit.csv \
  --train-col smiles --gen-col smiles \
  --train-mode smiles --gen-mode smiles \
  --out /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/figure_b_chemspace_gen.png    
python /public/home/users/haoxw/generate_AI/branch_Li/code/plt_c_f.py \
  --train /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron.csv \
  --gen /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/all_molecules_physical_with_rdkit.csv \
  --outdir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_12/outputs/ \
  --bins 50  