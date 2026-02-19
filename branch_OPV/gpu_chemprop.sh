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
"$PYBIN/chemprop_train" \
 --data_path /public/home/users/haoxw/generate_AI/branch_OLED/data/opv.csv \
 --target_columns homo gap MolMR LogP \
 --dataset_type regression \
 --save_dir ckpts/chemprop_DPER0_cv5_new_6 \
 --split_type random --split_sizes 0.8 0.1 0.1 \
 --batch_size 64 --epochs 100 \
 --ensemble_size 1 --num_folds 5 --seed 42 \
 --save_preds --save_smiles_splits \
 --hidden_size 1200 --depth 6 --dropout 0.1 \
 --features_generator rdkit_2d_normalized \
 --no_features_scaling

