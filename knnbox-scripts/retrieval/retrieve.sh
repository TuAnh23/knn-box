:<<!
[script description]: use vanilla-knn-mt to translate
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line will speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..

# Set environment vars
ENV_VAR_PATH=$1
# Check if the file exists
if [ ! -f "$ENV_VAR_PATH" ]; then
  ENV_VAR_PATH=${PROJECT_PATH}/knnbox-scripts/retrieval/.env
  echo "ENV_VAR_PATH not passed in. Set ENV_VAR_PATH=${ENV_VAR_PATH}"
fi

export $(grep -v '^\s*#' "$ENV_VAR_PATH" | grep -v '^\s*$' | xargs)

if [[ ${MT_MODEL} == "deltalm_base_ft_ted"  ]]; then
  BASE_MODEL=$PROJECT_PATH/models/deltalm_base_ft_ted/checkpoint_best.pt
  DATA_PATH=$PROJECT_PATH/data-bin/ted_deltalm
elif [[ ${MT_MODEL} == "ted_new"  ]]; then
  BASE_MODEL=$PROJECT_PATH/models/ted_new/checkpoint_best.pt
  DATA_PATH=$PROJECT_PATH/data-bin/ted_new
fi

python $PROJECT_PATH/knnbox-scripts/retrieval/retrieve.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang en --target-lang de \
--gen-subset train \
--max-tokens 2048 \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe=sentencepiece \
--user-dir $PROJECT_PATH/knnbox/models \
#--bpe "fastbpe" \
#--bpe-codes $PROJECT_PATH/models/ted/ende30k.fastbpe.code \
#--tokenizer moses
#--remove-bpe \
