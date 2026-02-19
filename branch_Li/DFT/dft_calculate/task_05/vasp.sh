#!/bin/bash
#SBATCH -J haoxw
#SBATCH --account=users
#SBATCH -p cu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --error=%.err              # 输出错误日志
#SBATCH --output=%.out             # 输出标准日志

source /public/home/users/haoxw/intel/oneapi/setvars.sh
ulimit -s unlimited

mpirun /public/home/users/haoxw/software/vasp.6.4.2/bin/vasp_std > vasp.out

