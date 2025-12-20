#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=[h200train]
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=berman.ed@northeastern.edu

module load python/3.13.5

eval "$(poetry env activate)"

export WANDB_MODE=online

python -m fourierflow.commands train --trial 0 experiments/plasticity/ffno/4_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/ffno/8_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/ffno/12_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/ffno/16_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/ffno/20_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/ffno/24_layers/config.yaml

python -m fourierflow.commands train --trial 0 experiments/plasticity/geo-fno/4_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/geo-fno/8_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/geo-fno/12_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/geo-fno/16_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/geo-fno/20_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/plasticity/geo-fno/24_layers/config.yaml

