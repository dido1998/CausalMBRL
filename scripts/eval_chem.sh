#!/bin/bash

echo Running on $HOSTNAME

num_obj=$1
name=$2
encoder=$3
num_colors=$4
max_steps=$5
movement=$6
graph=$7
seed=$8
loss=$9
mode=${10}
emb=${11}

dir="models_"$emb

data="data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-$mode-$graph.h5"

save=$dir"/Chemistry/"$name"_"$seed"/"

name=$name"_"$loss"_"$encoder"_"$num_obj"_"$num_colors"_"$max_steps"_"$movement"_"$graph
echo $name

extras=""
if [[ $name == *"LSTM"* ]]; then
	extras="--recurrent"
fi

if [[ $name == *"RIM"* ]]; then
	extras="--recurrent"
fi

if [[ $name == *"SCOFF"* ]]; then
	extras="--recurrent"
fi

echo $extras

python ./eval.py --dataset $data \
	--save-folder $save""$name --save $dir $extras
