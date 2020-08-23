import os

runs=5

models=["AE", "VAE", "Modular", "GNN"]#, "LSTM", "RIM", "SCOFF"]

encoders=["medium"]
#encoders=["small", "medium", "large"]

losses=["NLL", "Contrastive"]

embs=[4, 32]
cmap="Blues"
#cmap="Sets"

for emb in embs:
    for loss in losses:
        for model in models:
            if model == "VAE" and loss == "Contrastive":
                continue
            for run in range(1,runs+1):
                for encoder in encoders:
                    if encoder is "small":
                        bs = 1024
                        time = 24
                    elif encoder is "medium":
                        bs = 512
                        time = 48
                    else:
                        bs = 128
                        time = 96
                    for i in [3,5]:
                         os.system(f"sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time={time}:0:0 run.sh  {i} {model} {encoder} {bs} {cmap} {run} {loss} {emb}")
