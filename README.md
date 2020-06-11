# Ingredients for Causal Learning in model-based RL


## Generate data

### Chemistry Env
```
python data_gen/env.py --env_id ColorChanging-<num_obj>-<num_colors>-v0 --fname data/ColorChanging_<num_obj>-<num_colors>-train.h5  --num_episodes <num_epicodes> --episode-length <episode_length> --seed 1
```
Note: Refer to chem_data.sh to generate data for expts. usage: ./chem_data.sh <num_obj> <num_colors>

### Physics Env 

```
python data_gen/env.py --env_id WShapesObserved-Train-<num_obj>-<cmap>-v0 --fname data/wshapes_observed_<num_obj>_<cmap>_train.h5  --num_episodes 1000 --episode-length 100 --seed 1
```

Note: Refer to gen_observed.sh and gen_unobserved.sh to generate data for expts. usage: ./gen_observed.sh <num_obj> <cmap>   ./gen_unobserved.sh <num_obj> <cmap>


## Experiments
For AE, VAE, Modular (default is AE)
```
python train_causal_baselines.py --dataset <train_file> --encoder <size> --name  <folder_name> --eval-dataset <test_file> \
--num-graphs 10  --embedding-dim <emb-dim --num-objects <num_obj> --action-dim <num_actions>   --valid-dataset <validation_file> --save-folder <save_folder> --batch-size <batch_size> --seed 5 \
--epochs 100 --pretrain-epochs 100 --predict-diff [--vae] [--modular]
```
For LSTM, RIM (default is LSTM)
```
python train_causal_lstm.py --dataset <train_file> --encoder <size> --name  <folder_name> --eval-dataset <test_file> \
--num-graphs 10  --embedding-dim <emb-dim --num-objects <num_obj> --action-dim <num_actions>   --valid-dataset <validation_file> --save-folder <save_folder> --batch-size <batch_size> --seed 5 \
--epochs 100 --pretrain-epochs 100 --predict-diff [--rim]
