#!/bin/bash
  
num_obj=$1
cmap=$2

echo $num_obj
echo $cmap

python data_gen/env.py --env-id WShapesRL-Observed-Train-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_train.h5"  --num-episodes 1000 --episode-length 100 --seed 1
python data_gen/env.py --env-id WShapesRL-Observed-Train-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_valid.h5"  --num-episodes 200 --episode-length 100 --seed 2
python data_gen/env.py --env-id WShapesRL-Observed-Train-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_test-v0.h5"  --num-episodes 10000 --episode-length 10 --seed 3
python data_gen/env.py --env-id WShapesRL-Observed-Test-v1-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_test-v1.h5"  --num-episodes 10000 --episode-length 10 --seed 4
python data_gen/env.py --env-id WShapesRL-Observed-Test-v2-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_test-v2.h5"  --num-episodes 10000 --episode-length 10 --seed 5
python data_gen/env.py --env-id WShapesRL-Observed-Test-v3-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_test-v3.h5"  --num-episodes 10000 --episode-length 10 --seed 6
python data_gen/env.py --env-id WShapesRL-Observed-ZeroShot-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_ZeroShot.h5"  --num-episodes 10000 --episode-length 10 --seed 7
python data_gen/env.py --env-id WShapesRL-Observed-ZeroShot-$num_obj-$cmap-v0 --fname data/wshapes_observed_$num_obj"_"$cmap"_ZeroShotShape.h5"  --num-episodes 10000 --episode-length 10 --seed 8