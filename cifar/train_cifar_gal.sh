#!/bin/bash
DATA_DIR='/path-to-cifar-data/'
OUTPUT_DIR='./checkpoints/'
# DATASET: cifar10 | cifar100
DATASET=${1:-cifar10} 
MODEL='efficientnet-b1'
BATCH_SIZE=320
# GAL_ARCH on CIFAR-10: '100 32 16', GAL_ARCH on CIFAR-100: '256 64 32' 
OPTIMIZER=${2:-sgd}
GAL_ARCH=$3
GAL_SCALE=${4:-1}
GAL_RATIO=${5:-0.01}
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
	--checkpoint "${OUTPUT_DIR}/${DATASET}/${MODEL}/gal_${OPTIMIZER}" \
	--mtype 'gal' \
	--gal_scale ${GAL_SCALE} \
	--gal_arch ${GAL_ARCH} \
	--gal_ratio ${GAL_RATIO} 
elif [ "${OPTIMIZER}" = "lookahead" ]; then
python train_cifar.py \
	-a ${MODEL} \
	--train-batch ${BATCH_SIZE} \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--optimizer 'lookahead' \
	--lr 0.05 \
	--wd 1e-5 \
	--gamma 0.1 \
	--datapath ${DATA_DIR} \
	--checkpoint "${OUTPUT_DIR}/${DATASET}/${MODEL}/gal_${OPTIMIZER}" \
	--mtype 'gal' \
	--gal_scale ${GAL_SCALE} \
	--gal_arch ${GAL_ARCH} \
	--gal_ratio ${GAL_RATIO}
elif [ "${OPTIMIZER}" = "adabound" ]; then
python train_cifar.py \
	-a ${MODEL} \
	--train-batch ${BATCH_SIZE} \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--optimizer 'adabound' \
	--lr 0.001 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath ${DATA_DIR} \
	--checkpoint "${OUTPUT_DIR}/${DATASET}/${MODEL}/gal_${OPTIMIZER}" \
	--mtype 'gal' \
	--gal_scale ${GAL_SCALE} \
	--gal_arch ${GAL_ARCH} \
	--gal_ratio ${GAL_RATIO}
fi