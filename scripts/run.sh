#!/bin/bash

echo Running on $HOSTNAME
module load miniconda3

source activate pytorch

echo Running version $name

num_obj=$1
name=$2
encoder=$3
bs=$4
cmap=$5

data="/home/sarthmit/c-swm-v0/data/wshapes_observed"

save="/home/sarthmit/scratch/Causal-SWM/Observed/"$name"/"

name=$name"_"$encoder"_"$num_obj"_"$cmap
echo $name

python ../train_baselines.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" --encoder $encoder --name $name --embedding-dim 10 --num-objects $num_obj --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save --predict-diff --modular
