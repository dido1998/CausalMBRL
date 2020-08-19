#!/bin/bash


num_obj=$1
num_colors=$2
seed=$3
graph=$4
max_steps=$5
movement=$6
python ../data_gen/env.py --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --fname data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-$seed.h5  --num-episodes 1000 --episode-length 100 --seed $seed --save_graph --save_graph_location data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-graph-$seed --graph $graph
python ../data_gen/env.py --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --fname data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-valid-$seed.h5  --num-episodes 200 --episode-length 100 --seed 3 --load_graph --load_graph_location data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-graph-$seed
python ../data_gen/env.py --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --fname data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-test-$seed.h5  --num-episodes 10000 --episode-length 10 --seed 2 --load_graph --load_graph_location data/ColorChangingRL_$num_obj-$num_colors-$max_steps-$movement-train-graph-$seed

