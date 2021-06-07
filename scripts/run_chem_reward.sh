#!/bin/bash

echo Running on $HOSTNAME


num_obj=$1
name=$2
encoder=$3
bs=$4
num_colors=$5
max_steps=$6
movement=$7
graph=$8
run=$9
loss={10}
emb=${11}

save="Models_"$emb"/Chemistry/"$name"_"$run"/"

name=$name"_"$loss"_"$encoder"_"$num_obj"_"$num_colors"_"$max_steps"_"$movement"_"$graph
echo $name

python ../train_reward_predictor.py --save-folder $save""$name
python ../train_reward_predictor.py --save-folder $save""$name --random
python ../train_reward_predictor.py --save-folder $save""$name --finetune
