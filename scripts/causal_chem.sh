#!/bin/bash

num_obj=$1
num_colors=$2
graph=$3
max_steps=$4
movement=$5
save_folder=causal-$num_obj-$num_colors-$max_steps-$movement-$graph
truth=True

lsparse=1.0
save_folder_="$save_folder-nll"
rm -r $save_folder_
mkdir "$save_folder_"
touch "$save_folder_/train.log"
python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$graph.h5 \
--graph  data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-graph-$graph \
--encoder "medium" --name  "causal_$num_obj-$size-$num_colors-$graph"   \
--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$graph.h5 \
--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$graph.h5 \
--save-folder $save_folder_ --batch-size 512 --seed 1   --predict-diff \
--epochs 0 --finetune-epochs 200  --pretrain-epochs 0 --num-graphs 20  --lsparse $lsparse --modular --causal | tee -a "$save_folder_/train.log"



lsparse=1.0
save_folder_="$save_folder-contrastive"
echo $save_folder_
rm -r "$save_folder_"
mkdir "$save_folder_"
touch "$save_folder_/train.log"
python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$graph.h5 \
--graph  data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-graph-$graph \
--encoder "medium" --name  "causal_$num_obj-$size-$num_colors-$graph"   \
--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$graph.h5 \
--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$graph.h5 \
--save-folder $save_folder_ --batch-size 512 --seed 1 --causal --contrastive \
--epochs 0 --finetune-epochs 200 --pretrain-epochs 0 --num-graphs 20 --lsparse $lsparse --predict-diff --modular  | tee -a "$save_folder_/train.log"


lsparse=2.0
save_folder_="$save_folder-$lsparse"
rm -r "$save_folder_"
mkdir "$save_folder_"
touch "$save_folder_/train.log"
python train_baselines.py --dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$graph.h5 \
--graph  data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-graph-$graph \
--encoder "medium" --name  "causal_$num_obj-$size-$num_colors-$graph"   \
--eval-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$graph.h5 \
--embedding-dim-per-object 32 --num-objects $num_obj --action-dim $num_colors  \
--valid-dataset data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$graph.h5 \
--save-folder $save_folder_ --batch-size 512 --seed 1 \
--epochs 0 --finetune-epochs 300  --pretrain-epochs 0 --num-graphs 20 --lsparse $lsparse --predict-diff --modular --causal | tee -a "$save_folder_/train.log"




