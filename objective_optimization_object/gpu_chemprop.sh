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
"$PYBIN/chemprop_train" --data_path /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin_log1p.csv --target_columns D P EG r0 --dataset_type regression --save_dir ckpts/chemprop_DPER0_cv5_new_8 --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --batch_size 64 --epochs 10000 --ensemble_size 10 --num_folds 5 --seed 42 --save_preds  --save_smiles_splits --features_generator morgan_count --no_features_scaling --hidden_size 300 --depth 4 --dropout 0.35 --ensemble_size 10
