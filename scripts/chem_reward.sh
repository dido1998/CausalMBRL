#!/bin/bash

num_obj=$1
num_colors=$2
seed=$3
contrastive_loss=$4
max_steps=$5
movement=$6
model=$7
save_folder=models_without_seeds/$model-$num_obj-$num_colors-$max_steps-$movement-$seed-$contrastive_loss
truth=True

rm -r "$save_folder-1"
rm -r "$save_folder-2"
rm -r "$save_folder-3"

mkdir "$save_folder-1"
mkdir "$save_folder-2"
mkdir "$save_folder-3"

touch "$save_folder-1/train_rl.log"
touch "$save_folder-2/train_rl.log"
touch "$save_folder-3/train_rl.log"

save_folder_="$save_folder-1"

python train_reward_predictor.py --save_folder "$save_folder_" --random
python train_reward_predictor.py --save_folder "$save_folder_" --finetune
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 1
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 5
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 10

python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 1
python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 5
python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 10


save_folder_="$save_folder-2"

python train_reward_predictor.py --save_folder "$save_folder_" --random
python train_reward_predictor.py --save_folder "$save_folder_" --finetune
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 1
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 5
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 10

python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 1
python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 5
python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 10

save_folder_="$save_folder-3"

python train_reward_predictor.py --save_folder "$save_folder_" --random
python train_reward_predictor.py --save_folder "$save_folder_" --finetune
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 1
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 5
python test_planning.py --save_folder "$save_folder_" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 10

python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 1
python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 5
python test_planning.py --save_folder "$save_folder_" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_" --num-steps 10