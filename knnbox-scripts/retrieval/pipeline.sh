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

# Clean start
rm -rf "data/${MT_MODEL}/${DATASET}/${DATASTORE_NAME}_${LAYER}/${CUSTOM_FILE_NAME}.bin"
rm -rf "/project/OML/tdinh/knn-qe/${MT_MODEL}/${CUSTOM_FILE_NAME}/${DATASTORE_NAME}_${LAYER}"

conda activate /home/tpalzer/miniconda3/envs/k
which python
bash retrieve.sh ${ENV_VAR_PATH}

conda activate /home/tpalzer/miniconda3/envs/embed
which python
python embed.py

conda activate /home/tpalzer/miniconda3/envs/k
which python
python reformat_output.py \
  --dataset ${CUSTOM_FILE_NAME} \
  --mt_model ${MT_MODEL} \
  --emb_layer ${LAYER} \
  --datastore ${DATASTORE_NAME} \
  --output_root_dir "/project/OML/tdinh/knn-qe"