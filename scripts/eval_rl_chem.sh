#!/bin/bash

echo Running on $HOSTNAME

num_obj=$1
name=$2
encoder=$3
num_colors=$5
max_steps=$6
movement=$7
graph=$8
seed=$9
loss=${10}
mode=${11}
emb=${12}
steps=${13}

dir="models_"$emb

#env=WShapesRL-Observed-$mode-$num_obj-$cmap-v0
#env=WShapesRL-Unobserved-Train-$num_obj-$cmap-v0
env=ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0

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

if [[ $name == *"NLL"* ]]; then
	extras=$extras" --finetune"
fi

python ./test_planning.py --save-folder $save""$name --save $dir \
    --num-eval 1000 --num-steps $steps \
    --env-id $env --random $extras
