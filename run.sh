#!/bin/bash
#SBATCH -J train_CIFAR10_deepmind
#SBATCH -p gpu # partition (queue)
#SBATCH --gres=gpu:1
#SBATCH --mem=6G # memory pool for all cores
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH -p serial_requeue         # Partition to submit to
#SBATCH --mem=16000               # memory pool for all cores
#SBATCH --export=ALL
#SBATCH -o Jobs/Job.%x.%N.%j.out # STDOUT
#SBATCH -e Jobs/Job.%x.%N.%j.err # STDERR

#SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
#SBATCH --mail-user=sromeropinto@g.harvard.edu # send-to address

module load Anaconda3/5.0.1-fasrc01
module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01
source activate Distributional_RL_Pytorch

cd /n/holylfs/LABS/uchida_lab/sandraromeropinto/Distributional_RL_Gym/ || exit

python -u train_CIFAR10_deepmind.py