#!/bin/bash


num_obj=$1
num_colors=$2
graph=$3
max_steps=$4
movement=$5
edge=$6
python data_gen/env.py --env-id ColorChangingTimeRL-$num_obj-$num_colors-$movement-$edge-$max_steps-v0 --fname data/ColorChangingTimeRL_$num_obj-$num_colors-$edge-$max_steps-$movement-train-$graph.h5  --num-episodes 1000 --episode-length 100 --seed 1 --save_graph --save_graph_location data/ColorChangingTimeRL_$num_obj-$num_colors-$edge-$max_steps-$movement-train-graph-$graph --graph $graph
python data_gen/env.py --env-id ColorChangingTimeRL-$num_obj-$num_colors-$movement-$edge-$max_steps-v0 --fname data/ColorChangingTimeRL_$num_obj-$num_colors-$edge-$max_steps-$movement-valid-$graph.h5  --num-episodes 200 --episode-length 100 --seed 3 --load_graph --load_graph_location data/ColorChangingTimeRL_$num_obj-$num_colors-$edge-$max_steps-$movement-train-graph-$graph
python data_gen/env.py --env-id ColorChangingTimeRL-$num_obj-$num_colors-$movement-$edge-$max_steps-v0 --fname data/ColorChangingTimeRL_$num_obj-$num_colors-$edge-$max_steps-$movement-test-$graph.h5  --num-episodes 10000 --episode-length 10 --seed 2 --load_graph --load_graph_location data/ColorChangingTimeRL_$num_obj-$num_colors-$edge-$max_steps-$movement-train-graph-$graph

