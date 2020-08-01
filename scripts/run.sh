#!/bin/bash

echo Running on $HOSTNAME

source activate pytorch

num_obj=$1
name=$2
encoder=$3
bs=$4
cmap=$5
seed=$6

data="/home/sarthmit/scratch/C-SWM/Data/Observed/wshapes_observed"
#data="/home/sarthmit/scratch/C-SWM/Data/Unobserved/wshapes_unobserved"
#data="/home/sarthmit/scratch/C-SWM/Data/FixedUnobserved/wshapes_fixedunobserved"

save="/home/sarthmit/scratch/C-SWM/Models/Observed/"$name"_"$seed"/"
#save="/home/sarthmit/scratch/C-SWM/Models/Unobserved/"$name"/"
#save="/home/sarthmit/scratch/C-SWM/Models/FixedUnobserved/"$name"/"

name=$name"_"$encoder"_"$num_obj"_"$cmap
echo $name

python ../train_baselines.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" --encoder $encoder --name $name --embedding-dim 10 --num-objects $num_obj --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save --seed $seed --predict-diff --cswm
