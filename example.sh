#!/bin/bash

ROOT=~/Downloads
OUTPUT=~/Downloads/output
INIT_N=100
N_SAMPLE=500
INIT_EPOCHS=500
EPOCHS=150
LR=0.05
SKIP=10
SEED=42 

python cifar10_resnet_main.py --root $ROOT --skip $SKIP --num-workers 2 --init-num-labelled $INIT_N --batch-size 100 --seed $SEED --output $OUTPUT --epochs $EPOCHS --init-epochs $INIT_EPOCHS --lr $LR --num-to-sample $N_SAMPLE --train-on-updated True --active-learning True


# --root The root folder for the downloaded and processed dataset.
# --skip We only start to accumulate prediction changes after this many epochs.
# --num-workers The number of workers to process the data.
# --init-num-labelled The size of the initial labelled training set.
# --batch-size The batch size for training.
# --seed The seed for the random number generator.
# --output The file name of the output file.
# --epochs The number of epochs to train on the updated training set after 1 round of active learning.
# --init-epochs The number of epochs to train on the initial labelled training set.
# --lr The learning rate.
# --num-to-sample The number of unlabelled examples to query.
# --train-on-updated Whether to train on the updated training set with the queried examples.
# --active-learning Whether to perform active learning or not.
