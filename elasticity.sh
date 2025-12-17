#!/bin/bash

python -m fourierflow.commands train --trial 0 experiments/elasticity/ffno/4_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/ffno/8_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/ffno/12_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/ffno/16_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/ffno/20_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/ffno/24_layers/config.yaml

python -m fourierflow.commands train --trial 0 experiments/elasticity/geo-fno/4_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/geo-fno/8_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/geo-fno/12_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/geo-fno/16_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/geo-fno/20_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/elasticity/geo-fno/24_layers/config.yaml

