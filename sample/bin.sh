export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  

TEXT=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/europarl
echo $PROJECT_PATH
# Binarize the data for training
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/spm.train.de-en \
    --testpref $TEXT/spm.test.de-en \
    --validpref $TEXT/spm.dev.de-en \
    --destdir ../data-bin/eurowan \
    --thresholdtgt 0 --thresholdsrc 0 \
    --workers 8 \
    --joined-dictionary \
    #--validpref $TEXT/spm.dev.de-en \
    #--testpref $TEXT/spm.test.de-en \
