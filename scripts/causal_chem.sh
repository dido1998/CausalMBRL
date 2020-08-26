#!/bin/bash

num_obj=$1
num_colors=$2
seed=$3
max_steps=$4
movement=$5
save_folder=causal-$num_obj-$num_colors-$max_steps-$movement-$seed
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

save_folder_="$save_folder-1"
python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
--encoder "medium" --name  "causal_$num_obj-$size-$num_colors-$seed"   \
--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
--save-folder $save_folder_ --batch-size 512 --seed 1 \
--epochs 100 --pretrain-epochs 100 --predict-diff --modular --learn-edges | tee -a "$save_folder_/train.log"

save_folder_="$save_folder-2"
python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
--encoder "medium" --name  "causal_$num_obj-$size-$num_colors-$seed"   \
--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
--save-folder $save_folder_ --batch-size 512 --seed 2 \
--epochs 100 --pretrain-epochs 100 --predict-diff --modular --learn-edges | tee -a "$save_folder_/train.log"

save_folder_="$save_folder-3"
python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5 \
--encoder "medium" --name  "causal_$num_obj-$size-$num_colors-$seed"   \
--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5 \
--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5 \
--save-folder $save_folder_ --batch-size 512 --seed 3 \
--epochs 100 --pretrain-epochs 100 --predict-diff --learn-edges | tee -a "$save_folder_/train.log"



