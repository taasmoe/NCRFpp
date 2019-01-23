#!/bin/bash

#SBATCH --job-name=inf5820
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3936
#SBATCH --cpus-per-task=8

# Increase this number when you really need parallel computing
# (don't set it to more than 6 or 8 cores):
#SBATCH --ntasks-per-node=4


source /cluster/bin/jobsetup
set -o errexit

module purge
module use -a /projects/nlpl/software/modulefiles/
module load nlpl-pytorch

echo $SUBMITDIR

python main.py --config norne.config
