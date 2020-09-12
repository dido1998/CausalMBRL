#!/bin/bash

num_obj=$1
num_colors=$2
seed=$3
contrastive_loss=$4
max_steps=$5
movement=$6
time=$7
save_folder=rim-$num_obj-$num_colors-$max_steps-$movement-$seed-$contrastive_loss-$time
truth=True

rm -r "$save_folder-1"
rm -r "$save_folder-2"
rm -r "$save_folder-3"

mkdir "$save_folder-1"
mkdir "$save_folder-2"
mkdir "$save_folder-3"

touch "$save_folder-1/train.log"
touch "$save_folder-2/train.log"
touch "$save_folder-3/train.log"


if [ $contrastive_loss == $truth ]
then
	save_folder_="$save_folder-1"
  	python train_recurrent.py --dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "rim_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 32 --seed 1 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --rim --contrastive | tee -a "$save_folder_/train.log"

    save_folder_="$save_folder-2"
  	python train_recurrent.py --dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "rim_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 32 --seed 2 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --rim --contrastive | tee -a "$save_folder_/train.log"

	save_folder_="$save_folder-3"
  	python train_recurrent.py --dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "rim_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 32 --seed 3 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --rim --contrastive | tee -a "$save_folder_/train.log"

else
    save_folder_="$save_folder-1"
  	python train_recurrent.py --dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "rim_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 32 --seed 1 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --rim | tee -a "$save_folder_/train.log"

    save_folder_="$save_folder-2"
  	python train_recurrent.py --dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "rim_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 32 --seed 2 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --rim | tee -a "$save_folder_/train.log"

	save_folder_="$save_folder-3"
  	python train_recurrent.py --dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "rim_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 32 --seed 3 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --rim | tee -a "$save_folder_/train.log"

fi

