#!/bin/bash

echo Running on $HOSTNAME


num_obj=$1
name=$2
encoder=$3
bs=$4
cmap=$5
run=$6
loss=$7
emb=$8

save="Models_"$emb"/Unobserved/"$name"_"$run"/"

name=$name"_"$loss"_"$encoder"_"$num_obj"_"$cmap
echo $name

python ../train_reward_predictor.py --save-folder $save""$name
python ../train_reward_predictor.py --save-folder $save""$name --random
python ../train_reward_predictor.py --save-folder $save""$name --finetune
