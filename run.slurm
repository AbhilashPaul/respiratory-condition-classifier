#!/bin/bash
#SBATCH --job-name=prj-645-g16
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=stdout_%j.out
#SBATCH --error=stderror_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

source ~/software/init-conda.sh
conda activate enel645

python main.py

conda deactivate
