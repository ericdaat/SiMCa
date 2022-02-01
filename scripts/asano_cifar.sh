#!/bin/bash

type=10
# the network
hc=1        # number of heads
ncl=512     # number of clusters
arch='alexnet'

# optimization
nopts=400   # number of SK-optimizations
epochs=400  # numbers of epochs

# queue
max_queue_len=3
queue_start_epoch=10

# other
device='1' # cuda device
bs=256      # batchsize
lr=0.001   # learning rate

dir="./data"
folder=cifar${type}/${arch}-K${ncl}_lr${lr}_bs${bs}_hc${hc}_nopt${nopts}_n${epochs}_linear_sequential
EXP=${folder}
mkdir -p ${EXP}

python3 src/asano/cifar.py \
  --arch ${arch} \
  --device ${device} \
  --exp ${EXP} \
  --datadir ${dir} \
  --type ${type} \
  --batch-size ${bs} \
  --lr ${lr} \
  --nopts ${nopts} \
  --epochs ${epochs} \
  --hc ${hc} \
  --max_queue_len ${max_queue_len} \
  --max_queue_len ${queue_start_epoch} \
  --ncl ${ncl} | tee -a ${EXP}/log.txt
