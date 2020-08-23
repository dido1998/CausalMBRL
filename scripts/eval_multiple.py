import os

models=["AE", "VAE", "Modular", "GNN", "LSTM", "RIM", "SCOFF"]

encoders=["medium"]
losses=["NLL", "Contrastive"]

modes=["test-v0"]
#modes=["test", "test-v0", "test-v1", "test-v2", "test-v3", "ZeroShotShape", "ZeroShot"]

embs=[4]#, 32]
cmap="Blues"
#cmap="Sets"

for emb in embs:
    for mode in modes:
        for loss in losses:
            for model in models:
                for run in range(1,4):
                    for encoder in encoders:
                        for i in [3,5]:
                             os.system(f"sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=3:0:0 eval.sh {i} {model} {encoder} {cmap} {run} {loss} {mode} {emb}")
