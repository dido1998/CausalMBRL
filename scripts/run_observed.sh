#!/bin/bash

echo Running on $HOSTNAME
conda activate py37

num_obj=$1
name=$2
encoder=$3
bs=$4
cmap=$5
seed=$6
loss=$7
emb=$8


data="data/Observed/wshapes_observed"

save="models_"$emb"/Observed/"$name"_"$seed"/"

name=$name"_"$loss"_"$encoder"_"$num_obj"_"$cmap
echo $name

if [[ $loss == "NLL" ]]; then
	extras=""
else
	extras="--contrastive"
fi

if [[ $name == *"VAE"* ]]; then
	python ./train_baselines.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" \
		--encoder $encoder --name $name --embedding-dim-per-object $emb \
		--num-objects $num_obj \
		--valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 \
		--pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
		--seed $seed --predict-diff --vae $extras
elif [[ $name == *"AE"* ]]; then
        python ./train_baselines.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	        --num-objects $num_obj \
                --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff $extras
elif [[ $name == *"Modular"* ]]; then
        python ./train_baselines.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" \
                --encoder $encoder --name $name --embedding-dim-per-object $emb\
	       	--num-objects $num_obj \
                --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --modular $extras 
elif [[ $name == *"GNN"* ]]; then
        python ./train_baselines.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --gnn $extras
elif [[ $name == *"LSTM"* ]]; then
        python ./train_recurrent.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff $extras
elif [[ $name == *"RIM"* ]]; then
        python ./train_recurrent.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --rim $extras
elif [[ $name == *"SCOFF"* ]]; then
        python ./train_recurrent.py --dataset $data"_"$num_obj"_"$cmap"_train.h5" \
                --encoder $encoder --name $name --embedding-dim-per-object $emb \
	       	--num-objects $num_obj \
                --valid-dataset $data"_"$num_obj"_"$cmap"_valid.h5" --epochs 100 \
                --pretrain-epochs 100 --batch-size $bs --silent --save-folder $save \
                --seed $seed --predict-diff --scoff $extras
fi
