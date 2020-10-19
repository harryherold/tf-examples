#!/bin/bash
#SBATCH -J VGG16
#SBATCH --account=p_hpdlf_itwm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH -p ml
#SBATCH --gres=gpu:1
#SBATCH --exclusive

module load modenv/ml
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

cd $HOME/workspace/python/vgg16

source venv/bin/activate

#srun measure_epochs.sh
srun measure_batchsizes.sh

deactivate
