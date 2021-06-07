#!/bin/bash

echo Running on $HOSTNAME

num_obj=$1
name=$2
encoder=$3
cmap=$4
seed=$5
loss=$6
mode=$7
emb=$8
steps=$9

dir="Models_"$emb

env=WShapesRL-Observed-$mode-$num_obj-$cmap-v0
#env=WShapesRL-Unobserved-Train-$num_obj-$cmap-v0
#env=WShapesRL-FixedUnobserved-Train-$num_obj-$cmap-v0

save=$dir"/Observed/"$name"_"$seed"/"

name=$name"_"$loss"_"$encoder"_"$num_obj"_"$cmap
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

if [[ $name == *"NLL"* ]]; then
	extras=$extras" --finetune"
fi

python ./test_planning.py --save-folder $save""$name --save $dir \
    --num-eval 1000 --num-steps $steps \
    --env-id $env --random $extras
