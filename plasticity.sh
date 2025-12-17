#!/bin/bash

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

