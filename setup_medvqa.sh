#!/bin/bash

echo "Setting up MEDVQA environment on cluster..."

# --- micromamba setup ---
export PATH="/netscratch/lrippe/bin:$PATH"
eval "$(micromamba shell hook -s bash)"

# --- activate environment ---
echo "Activating medvqa environment..."
micromamba activate /netscratch/lrippe/envs/medvqa

# --- PYTHONPATH for project ---
export PYTHONPATH="/netscratch/lrippe/project_scripts/medvqa:$PYTHONPATH"
echo "Python path includes project directory: /netscratch/lrippe/project_scripts/medvqa"

# --- Info summary ---
echo "Environment setup complete!"
echo "Micromamba environment: medvqa"
echo "Python executable: $(which python)"
python --version 2>&1 | head -1