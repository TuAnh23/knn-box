export CUDA_VISIBLE_DEVICES=2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  

TEXT=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )
fairseq-train \
    $TEXT/../data-bin/eurowan \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 2 \
    --max-tokens 4096 \
    --max-epoch 20 \