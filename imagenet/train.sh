#!/bin/bash
DATA='/path-to-imagenet-images/'
./distributed_train.sh 8 ${DATA} \
	--model efficientnet_b2 \
	-b 100 \
	--sched step \
	--epochs 90 \
	--decay-epochs 2.4 \
	--decay-rate .97 \
	--opt rmsproptf \
	--opt-eps .001 \
	-j 8 \
	--warmup-lr 1e-6 \
	--weight-decay 1e-5 \
	--drop 0.3 \
	--drop-connect 0.2 \
	--model-ema \
	--model-ema-decay 0.9999 \
	--aa rand-m9-mstd0.5 \
	--remode pixel \
	--reprob 0.2 \
	--amp \
	--lr .016 