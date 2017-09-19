#!/bin/bash

#SBATCH -n 1
#SBATCH -t 0-3:00
#SBATCH -p serial_requeue
#SBATCH --mem=20000
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nweir@fas.harvard.edu

source new-modules.sh
source activate PYTO_SEG_ENV

python3 /n/denic_lab/Users/nweir/python_packages/csth-imaging/slurm_scripts/get_test_set.py
