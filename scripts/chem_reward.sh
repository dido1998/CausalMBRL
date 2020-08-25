#!/bin/bash

source ../../../tensor2tensor/bin/activate

num_obj=$1
num_colors=$2
seed=$3
contrastive_loss=$4
max_steps=$5
movement=$6
model=$7
save_folder=models_without_seeds/$model-$num_obj-$num_colors-$max_steps-$movement-$seed-$contrastive_loss
truth=True

echo "$save_folder"

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
model_folder="${model}_${num_obj}--${num_colors}-${seed}"
python train_reward_predictor.py --save-folder "$save_folder_/$model_folder" --random
python train_reward_predictor.py --save-folder "$save_folder_/$model_folder" --finetune
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 1
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 5
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 10

python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 1
python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 5
python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 10


save_folder_="$save_folder-2"
model_folder="${model}_${num_obj}--${num_colors}-${seed}"
python train_reward_predictor.py --save-folder "$save_folder_/$model_folder" --random
python train_reward_predictor.py --save-folder "$save_folder_/$model_folder" --finetune
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 1
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 5
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 10

python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 1
python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 5
python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 10

save_folder_="$save_folder-3"

model_folder="${model}_${num_obj}--${num_colors}-${seed}"
python train_reward_predictor.py --save-folder "$save_folder_/$model_folder" --random
python train_reward_predictor.py --save-folder "$save_folder_/$model_folder" --finetune
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 1
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 5
python test_planning.py --save-folder "$save_folder_/$model_folder" --random --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 10

python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 1
python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 5
python test_planning.py --save-folder "$save_folder_/$model_folder" --finetune --env-id ColorChangingRL-$num_obj-$num_colors-$movement-$max_steps-v0 --save "$save_folder_/$model_folder" --num-steps 10
