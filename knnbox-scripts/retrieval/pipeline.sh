#!/bin/bash
source /home/tdinh/.bashrc

# Set environment vars
ENV_VAR_PATH=$1
# Check if the file exists
if [ -f "$ENV_VAR_PATH" ]; then
    # Export environment variables
    export $(grep -v '^\s*#' "$ENV_VAR_PATH" | grep -v '^\s*$' | xargs)
else
    echo "ENV_VAR_PATH not passed in."
fi

conda activate /home/tpalzer/miniconda3/envs/k
which python
bash retrieve.sh ${ENV_VAR_PATH}

conda activate /home/tpalzer/miniconda3/envs/embed
which python
python embed.py