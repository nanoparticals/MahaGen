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
#python /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/plt/code/analyze_selfies_sa_reward.py \
#  --input /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/samples.selfies \
#  --predictor /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/chemprop_DPER0_cv5_new_4/fold_2/model_14/model.pt \
#  --predictor-stats /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/et_stats_DPER0.json \
#  --targets D P EG r0 \
#  --goals max max max max \
#  --weights 0.25 0.25 0.4 0.10 \
#  --zscore \
#  --vgae-ckpt /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/vgae_struct_only_z16.pt \
#  --vgae-mean /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/z_stats/z_mean.npy \
#  --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs
#python /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/plt/code/unscale_properties.py \
#  --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/all_molecules.csv \
#  --stats /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/et_stats_DPER0.json \
#  --out /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/all_molecules_physical.csv
#python /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/plt/code/radar_six_axes.py \
#  --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/all_molecules_physical.csv \
#  --prop-cols D D_phys_km_s P P_phys_kbar EG EG_phys_m_s r0 r0_phys_g_cm3 \
#  --manual-range D 5 10 P 200 400  EG 4000 10000 r0 1.0 2 SA 1 8 Reward 0 2.5  \
#  --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs \
#  --units D km/s P kbar EG J/g r0 g/cm3 SA score Reward a.u. --dedup
#python /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/plt/code/make_all_paper_figs.py \
#  --gen_csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/all_molecules_physical.csv \
#  --train_csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin_selfies.csv \
#  --log /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/154281.out \
#  --out_dir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/paper_figs_all
python /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/plt/code/plt_b_origin.py \
  --train /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin_selfies.csv \
  --gen   /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/all_molecules_physical.csv \
  --train-col smiles --gen-col selfies \
  --train-mode smiles --gen-mode selfies \
  --out /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/figure_b_chemspace_gen_2w.png
#python /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/plt/code/plt_c_f.py \
#  --train /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin.csv \
#  --gen /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/all_molecules_physical.csv \
#  --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_41/outputs/ \
#  --bins 50