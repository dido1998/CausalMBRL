#!/bin/bash

echo Running on $HOSTNAME
source activate cswm

num_obj=$1
name=$2
encoder=$3
cmap=$4
seed=$5
loss=$6
mode=$7
emb=$8

dir="Models"
#dir="Models_"$emb

data="/home/sarthmit/scratch/C-SWM/Data/Observed/wshapes_observed"
#data="/home/sarthmit/scratch/C-SWM/Data/Unobserved/wshapes_unobserved"
#data="/home/sarthmit/scratch/C-SWM/Data/FixedUnobserved/wshapes_fixedunobserved"

save="/home/sarthmit/scratch/C-SWM/"$dir"/Observed/"$name"_"$seed"/"
#save="/home/sarthmit/scratch/C-SWM/Models/Unobserved/"$name"/"
#save="/home/sarthmit/scratch/C-SWM/Models/FixedUnobserved/"$name"/"

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
