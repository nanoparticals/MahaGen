#!/bin/bash
#SBATCH -J haoxw
#SBATCH --account=users
#SBATCH -p cu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --error=%.err              # 输出错误日志
#SBATCH --output=%.out             # 输出标准日志

source /public/home/users/haoxw/intel/oneapi/setvars.sh
ulimit -s unlimited

mpirun /public/software/vasp.5.4.1/bin/vasp_gam > vasp.out

