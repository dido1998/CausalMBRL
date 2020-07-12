#!/bin/bash
  
num_obj=$1
cmap=$2

echo $num_obj
echo $cmap

python data_gen/env.py --env_id WShapesUnobserved-Train-$num_obj-$cmap-v0 --fname data/wshapes_unobserved_$num_obj"_"$cmap"_train.h5"  --num_episodes 1000 --episode-length 100 --seed 1
python data_gen/env.py --env_id WShapesUnobserved-Train-$num_obj-$cmap-v0 --fname data/wshapes_unobserved_$num_obj"_"$cmap"_valid.h5"  --num_episodes 200 --episode-length 100 --seed 2
python data_gen/env.py --env_id WShapesUnobserved-Test-$num_obj-$cmap-v0 --fname data/wshapes_unobserved_$num_obj"_"$cmap"_test.h5"  --num_episodes 10000 --episode-length 10 --seed 3
python data_gen/env.py --env_id WShapesUnobserved-0shot-$num_obj-$cmap-v0 --fname data/wshapes_unobserved_$num_obj"_"$cmap"_0shot.h5"  --num_episodes 10000 --episode-length 10 --seed 4
