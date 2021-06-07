#!/bin/bash
  
num_obj=$1
cmap=$2

echo $num_obj
echo $cmap

direc="data/Unobserved/"

python data_gen/env.py --env-id WShapesRL-Unobserved-Train-$num_obj-$cmap-v0 --fname $direc"wshapes_unobserved_"$num_obj"_"$cmap"_train.h5"  --num-episodes 1000 --episode-length 100 --seed 1
python data_gen/env.py --env-id WShapesRL-Unobserved-Train-$num_obj-$cmap-v0 --fname $direc"wshapes_unobserved_"$num_obj"_"$cmap"_valid.h5"  --num-episodes 200 --episode-length 100 --seed 2
python data_gen/env.py --env-id WShapesRL-Unobserved-Train-$num_obj-$cmap-v0 --fname $direc"wshapes_unobserved_"$num_obj"_"$cmap"_test.h5"  --num-episodes 10000 --episode-length 10 --seed 3
python data_gen/env.py --env-id WShapesRL-Unobserved-ZeroShotShape-$num_obj-$cmap-v0 --fname $direc"wshapes_unobserved_"$num_obj"_"$cmap"_ZeroShotShape.h5"  --num-episodes 10000 --episode-length 10 --seed 4
