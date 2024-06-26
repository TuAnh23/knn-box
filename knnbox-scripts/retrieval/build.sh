:<<! 
[script description]: build a datastore for vanilla-knn-mt visualization
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/models/ted_new/checkpoint_best.pt
DATA_PATH=$PROJECT_PATH/data-bin/ted_new
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla-visual/enc_store

export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  
 python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--dataset-impl mmap \
--valid-subset train \
--skip-invalid-size-inputs-valid-test \
--max-tokens 4096 \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt_visual@transformer \
--knn-mode build_datastore \
--knn-datastore-path $DATASTORE_SAVE_PATH \
#--bpe fastbpe \
#--arch vanilla_knn_mt_visual@transformer_wmt19_de_en \
