#!/bin/bash

num_obj=$1
num_colors=$2
seed=$3
contrastive_loss=$4
max_steps=$5
movement=$6
save_folder=ae-$num_obj-$num_colors-$max_steps-$movement-$seed-$contrastive_loss
truth=True

rm -r $save_folder
mkdir $save_folder
touch $save_folder/train.log

if [ $contrastive_loss == $truth ]
then
	save_folder_="$save_folder-1"
  	python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "ae_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 512 --seed 1 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --contrastive | tee -a "$save_folder_/train.log"

	save_folder_="$save_folder-2"
  	python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "ae_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 512 --seed 2 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --contrastive | tee -a "$save_folder_/train.log"

	save_folder_="$save_folder-3"
  	python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "ae_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 512 --seed 3 \
	--epochs 100 --pretrain-epochs 100 --predict-diff --contrastive | tee -a "$save_folder_/train.log"
else
	save_folder_="$save_folder-1"
  	python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "ae_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 512 --seed 1 \
	--epochs 100 --pretrain-epochs 100 --predict-diff | tee -a "$save_folder_/train.log"

	save_folder_="$save_folder-2"
  	python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "ae_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 512 --seed 2 \
	--epochs 100 --pretrain-epochs 100 --predict-diff  | tee -a "$save_folder_/train.log"

	save_folder_="$save_folder-3"
  	python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
	--encoder "medium" --name  "ae_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
	--save-folder $save_folder_ --batch-size 512 --seed 3 \
	--epochs 100 --pretrain-epochs 100 --predict-diff | tee -a "$save_folder_/train.log"

fi

