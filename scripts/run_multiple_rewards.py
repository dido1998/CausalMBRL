import os

runs=3
models=["AE", "VAE", "Modular", "GNN"]
encoders=["medium"]
losses=["NLL", "Contrastive"]
cmap="Blues"
embs=[32]

for emb in embs:
    for loss in losses:
        for model in models:
            for encoder in encoders:
                if encoder is "small":
                    bs = 1024
                elif encoder is "medium":
                    bs = 512
                else:
                    bs = 256
                for i in [3,5]:
                    for run in range(1,runs+1):
                        os.system(f"sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=12:0:0 run_reward.sh  {i} {model} {encoder} {bs} {cmap} {run} {loss} {emb}")
