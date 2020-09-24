#!/bin/bash

num_obj=$1
num_colors=$2
seed=$3
contrastive_loss=$4
max_steps=$5
movement=$6
idx=$7
time=$8
edge=$9

run_file='scripts/lstm_chem.sh'

sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 1 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 2 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 3 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 1 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 2 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 3 $time $edge

run_file='scripts/rim_chem.sh'

sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 1 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 2 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 3 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 1 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 2 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 3 $time $edge

run_file='scripts/scoff_chem.sh'

sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 1 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 2 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=24:0:0 $run_file $num_obj $num_colors $seed \
        True $max_steps $movement 3 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 1 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 2 $time $edge
sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=36:0:0 $run_file $num_obj $num_colors $seed \
        False $max_steps $movement 3 $time $edge
