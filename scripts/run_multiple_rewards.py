import os

models=["AE", "VAE", "Modular"]
encoders=["small", "medium", "large"]
cmap="Blues"

for model in models:
    for encoder in encoders:
        if encoder is "small":
            bs = 1024
        elif encoder is "medium":
            bs = 512
        else:
            bs = 256
        for i in [3,5]:
            for run in range(1,4):
                os.system(f"sbatch --gres=gpu:1 --mem=12G --account=rrg-bengioy-ad --time=12:0:0 run_reward.sh  {i} {model} {encoder} {bs} {cmap} {run}")
