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
python /public/home/users/haoxw/generate_AI/branch_OLED/code/analyze_selfies_sa_reward.py   --input /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/samples.selfies --predictor /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/chemprop_DPER0_cv5_new_4/fold_0/model_0/model.pt   --predictor-stats /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/OPV_predictor_stats.json  --targets homo gap MolMR LogP --goals min min max max --weights 0.35 0.30 0.25 0.10 --zscore   --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/vgae_opv_4props.pt --vgae-mean /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/z_stats/z_mean.npy   --outdir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs
python ./code/unscale_properties.py   --csv /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules.csv   --stats /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/OPV_predictor_stats.json   --out /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules_physical.csv
python ./code/add_rdkit_logp_and_parity.py \
  --csv  /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules_physical.csv \
  --out  /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules_physical_with_rdkit.csv \
  --plot /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/logp_parity.png \
  --quiet-rdkit
python /public/home/users/haoxw/generate_AI/branch_OLED/code/make_all_paper_figs.py \
    --gen_csv /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules_physical.csv \
    --train_csv /public/home/users/haoxw/generate_AI/branch_OLED/data/opv.csv \
    --log /public/home/users/haoxw/generate_AI/branch_OLED/154323.out \
    --out_dir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/paper_figs_all
python /public/home/users/haoxw/generate_AI/branch_OLED/code/radar_six_axes.py \
  --csv /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules_physical_with_rdkit.csv \
  --prop-cols homo phys_homo gap phys_gap MolMR phys_MolMR LogP phys_LogP \
  --manual-range homo -7.0 -4.0 gap 3 8 MolMR 0 200 LogP 0 15 SA 1 8 Reward 0 2.5 \
  --outdir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs \
  --units homo eV gap eV MolMR a.u. LogP a.u. SA score Reward a.u. --dedup   
python /public/home/users/haoxw/generate_AI/branch_OLED/code/plt_b.py \
  --train /public/home/users/haoxw/generate_AI/branch_OLED/data/opv.csv \
  --gen   /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules_physical_with_rdkit.csv \
  --train-col smiles --gen-col smiles \
  --train-mode smiles --gen-mode smiles \
  --out /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/figure_b_chemspace_gen.png
python /public/home/users/haoxw/generate_AI/branch_OLED/code/plt_c_f.py \
  --train /public/home/users/haoxw/generate_AI/branch_OLED/data/opv.csv \
  --gen /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/all_molecules_physical_with_rdkit.csv \
  --outdir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_11/outputs/ \
  --bins 50   