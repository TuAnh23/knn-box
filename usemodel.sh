:<<!
[script description]: use neural machine translation model to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID  


# this line speed up faiss. base nmt dosent need faiss, 
# we set this environment variable here just for fair comparison.
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_PATH=$PROJECT_PATH/data-bin/ted
#DATA_PATH=$PROJECT_PATH/data-bin/iwslt14.de-en
BASE_MODEL=$PROJECT_PATH/models/ted/checkpoint_best.pt
#BASE_MODEL=$PROJECT_PATH/models/ted/checkpoint_best.pt

python usemodel.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang en --target-lang de \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--max-tokens 2048 \
--scoring sacrebleu \
--bpe "fastbpe" \
--bpe-codes $PROJECT_PATH/models/ted/ende30k.fastbpe.code \
--tokenizer moses --remove-bpe \
