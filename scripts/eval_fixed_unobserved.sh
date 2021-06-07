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

dir="Models_"$emb

data="data/FixedObserved/wshapes_fixedunobserved"

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

python ../eval.py --dataset $data"_"$num_obj"_"$cmap"_"$mode".h5" \
	--save-folder $save""$name --save $dir $extras
