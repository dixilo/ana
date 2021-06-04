#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J atm_test
#SBATCH --mail-user=suzuki.junya.4r@kyoto-u.ac.jp
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module use /global/common/software/cmb/${NERSC_HOST}/default/modulefiles
module load cmbenv
source cmbenv

#run the application:
srun -n 16 -c 4 --cpu_bind=cores python3 ./atmsim.py
