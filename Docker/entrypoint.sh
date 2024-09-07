#!/bin/bash
set -e

# Navigate to the volume you are making
cd /home/dir/RL_CoveragePlanning

# Go to the first package you are installing
cd rl-baselines3-zoo
pip install -e .
cd ..

# Go to the first package you are installing
cd viewpointPlaygroundEnv
pip install -e .
cd ..

# Finish up
echo "Local package installed!"

# Hand over the control back to the docekr container
exec "$@"