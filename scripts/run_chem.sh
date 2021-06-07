#!/bin/bash

echo Running on $HOSTNAME
conda activate py37

num_obj=$1
name=$2
encoder=$3
bs=$4
num_colors=$5
max_steps=$6
movement=$7
graph=$8
seed=$9
loss=${10}
emb=${11}


train_data="data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$graph.h5"
valid_data="data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$graph.h5"

save="models_"$emb"/Chemistry/"$name"_"$seed"/"

name=$name"_"$loss"_"$encoder"_"$num_obj"_"$num_colors"_"$max_steps"_"$movement"_"$graph
echo $name

if [[ $loss == "NLL" ]]; then
	extras=""
else
	extras="--contrastive"
fi

if [[ $name == *"VAE"* ]]; then
	python ../train_baselines.py --dataset $train_data \
		--encoder $encoder --name $name --embedding-dim-per-object $emb \
		--num-objects $num_obj \
		--valid-dataset $valid_data --epochs 100 \
		--pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
		--seed $seed --predict-diff --vae $extras
elif [[ $name == *"AE"* ]]; then
        python ../train_baselines.py --dataset $train_data \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	        --num-objects $num_obj \
                --valid-dataset $valid_data --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff $extras
elif [[ $name == *"Modular"* ]]; then
        python ../train_baselines.py --dataset $train_data \
                --encoder $encoder --name $name --embedding-dim-per-object $emb\
	       	--num-objects $num_obj \
                --valid-dataset $valid_data --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --modular $extras 
elif [[ $name == *"GNN"* ]]; then
        python ../train_baselines.py --dataset $train_data \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $valid_data --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --gnn $extras
elif [[ $name == *"LSTM"* ]]; then
        python ../train_recurrent.py --dataset $train_data \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $valid_data --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff $extras
elif [[ $name == *"RIM"* ]]; then
        python ../train_recurrent.py --dataset $train_data \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $valid_data --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --rim $extras
elif [[ $name == *"SCOFF"* ]]; then
        python ../train_recurrent.py --dataset $train_data \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $valid_data --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --scoff $extras
fi
