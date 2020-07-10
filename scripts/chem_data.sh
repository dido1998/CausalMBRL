#!/bin/bash

source activate gscan

num_obj=$1
num_colors=$2
python data_gen/env.py --env_id ColorChanging-$num_obj-$num_colors-v0 --fname data/ColorChanging_$num_obj-$num_colors-train.h5  --num_episodes 1000 --episode-length 100 --seed 1
python data_gen/env.py --env_id ColorChanging-$num_obj-$num_colors-v0 --fname data/ColorChanging_$num_obj-$num_colors-valid.h5  --num_episodes 200 --episode-length 100 --seed 3
python data_gen/env.py --env_id ColorChanging-$num_obj-$num_colors-v0 --fname data/ColorChanging_$num_obj-$num_colors-test.h5  --num_episodes 10000 --episode-length 10 --seed 2



