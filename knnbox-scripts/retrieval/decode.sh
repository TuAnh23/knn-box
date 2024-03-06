export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  

HERE=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/models/ted_new/checkpoint_best.pt
DATA_PATH=$PROJECT_PATH/data-bin/ted_new

TEST_INPUT=$PROJECT_PATH/sample/ted/spm.tst.de-en.de
PRED_LOG=$HERE/en-de.decode.log


fairseq-generate $DATA_PATH \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path $BASE_MODEL \
      --batch-size 256 \
      --tokenizer moses \
      --beam 4 \
      --remove-bpe=sentencepiece > $PRED_LOG

grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > ./hyp.txt
grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > ./ref.txt