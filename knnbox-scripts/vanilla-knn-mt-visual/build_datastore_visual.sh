:<<!
[script description]: build a datastore for vanilla-knn-mt visualization
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
source ${PROJECT_PATH}/knnbox-scripts/retrieval/.env

if [[ ${MT_MODEL} == "deltalm_base_ft_ted"  ]]; then
  BASE_MODEL=$PROJECT_PATH/models/deltalm_base_ft_ted/checkpoint_best.pt
  if [[ ${DATASTORE_NAME} == "reduced_ted"* ]]; then
    PORTION="${DATASTORE_NAME#reduced_ted_}"
    DATA_PATH=$PROJECT_PATH/data-bin/reduced_ted_deltalm/${PORTION}
    DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla-visual/$MT_MODEL/reduced_ted/${PORTION}_${LAYER}
  else
    DATA_PATH=$PROJECT_PATH/data-bin/${DATASTORE_NAME}_deltalm
    DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla-visual/$MT_MODEL/${DATASTORE_NAME}_${LAYER}
  fi
  ARCH=vanilla_knn_mt_visual@deltalm_base
  MAX_TOKENS=2048
elif [[ ${MT_MODEL} == "ted_new"  ]]; then
  BASE_MODEL=$PROJECT_PATH/models/ted_new/checkpoint_best.pt
  if [[ ${DATASTORE_NAME} == "reduced_ted"* ]]; then
    PORTION="${DATASTORE_NAME#reduced_ted_}"
    DATA_PATH=$PROJECT_PATH/data-bin/reduced_ted/${PORTION}
    DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla-visual/$MT_MODEL/reduced_ted/${PORTION}_${LAYER}
  else
    DATA_PATH=$PROJECT_PATH/data-bin/${DATASTORE_NAME}
    DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla-visual/$MT_MODEL/${DATASTORE_NAME}_${LAYER}
  fi
  ARCH=vanilla_knn_mt_visual@transformer
  MAX_TOKENS=4096
fi

python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--dataset-impl mmap \
--valid-subset train \
--skip-invalid-size-inputs-valid-test \
--max-tokens ${MAX_TOKENS} \
--user-dir $PROJECT_PATH/knnbox/models \
--arch ${ARCH} \
--knn-mode build_datastore \
--knn-datastore-path $DATASTORE_SAVE_PATH \
#--bpe fastbpe \
#--arch vanilla_knn_mt_visual@transformer_wmt19_de_en \
