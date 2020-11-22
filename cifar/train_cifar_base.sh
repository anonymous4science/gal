#!/bin/bash
DATA_DIR='/path-to-cifar-data/'
OUTPUT_DIR='./checkpoints/'
DATASET=$1
MODEL='efficientnet-b1'
BATCH_SIZE=320
OPTIMIZER=${2:-sgd}
if [ "${OPTIMIZER}" = "sgd" ]; then
python train_cifar.py \
	-a ${MODEL} \
	--train-batch ${BATCH_SIZE} \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--lr 0.05 \
	--wd 1e-5 \
	--gamma 0.1 \
	--datapath ${DATA_DIR} \
	--checkpoint "${OUTPUT_DIR}/${DATASET}/${MODEL}/standard_${OPTIMIZER}" \
	--mtype 'standard'
elif [ "${OPTIMIZER}" = "lookahead" ]; then
python train_cifar.py \
	-a ${MODEL} \
	--train-batch ${BATCH_SIZE} \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--lr 0.05 \
	--wd 1e-5 \
	--gamma 0.1 \
	--datapath ${DATA_DIR} \
	--checkpoint "${OUTPUT_DIR}/${DATASET}/${MODEL}/standard_${OPTIMIZER}" \
	--optimizer 'lookahead' \
	--mtype 'standard'
elif [ "${OPTIMIZER}" = "adabound" ]; then
python train_cifar.py \
	-a ${MODEL} \
	--train-batch ${BATCH_SIZE} \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--lr 0.001 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath ${DATA_DIR} \
	--checkpoint "${OUTPUT_DIR}/${DATASET}/${MODEL}/standard_${OPTIMIZER}" \
	--optimizer 'adabound' \
	--mtype 'standard'
fi