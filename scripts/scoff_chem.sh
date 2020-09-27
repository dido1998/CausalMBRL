#!/bin/bash

module load miniconda3
source activate cswm-delayed

num_obj=$1
num_colors=$2
seed=$3
contrastive_loss=$4
max_steps=$5
movement=$6
idx=$7
time=$8
edge=$9
if [ -z "$9" ]
then
	save_folder=scoff-$num_obj-$num_colors-$max_steps-$movement-$seed-$contrastive_loss-$time
	dataset1=data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5
	dataset2=data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5
	dataset3=data/ColorChanging${time}RL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5
else
	save_folder=scoff-$num_obj-$num_colors-$edge-$max_steps-$movement-$seed-$contrastive_loss-$time
	dataset1=data/ColorChanging${time}RL_$num_obj-$num_colors-$edge-$max_steps-$movement-train-$seed.h5
	dataset2=data/ColorChanging${time}RL_$num_obj-$num_colors-$edge-$max_steps-$movement-test-$seed.h5
	dataset3=data/ColorChanging${time}RL_$num_obj-$num_colors-$edge-$max_steps-$movement-valid-$seed.h5
fi


truth=True
save_folder="/home/sarthmit/scratch/C-SWM-delayed/"$save_folder

rm -r "$save_folder-$idx"
mkdir "$save_folder-$idx"

touch "$save_folder-$idx/train.log"

if [ $contrastive_loss == $truth ]
then
	save_folder_="$save_folder-$idx"
	python train_recurrent.py --dataset $dataset1 \
	--encoder "medium" --name  "scoff_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset $dataset2 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset $dataset3 \
	--save-folder $save_folder_ --batch-size 32 --seed $idx \
	--epochs 100 --pretrain-epochs 100 --predict-diff --scoff --contrastive | tee -a "$save_folder_/train.log"	
else
        save_folder_="$save_folder-$idx"
        python train_recurrent.py --dataset $dataset1 \
	--encoder "medium" --name  "scoff_$num_obj-$size-$num_colors-$seed"   \
	--eval-dataset $dataset2 \
	--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
	--valid-dataset $dataset3 \
	--save-folder $save_folder_ --batch-size 32 --seed $idx \
	--epochs 100 --pretrain-epochs 100 --predict-diff --scoff | tee -a "$save_folder_/train.log"
fi

