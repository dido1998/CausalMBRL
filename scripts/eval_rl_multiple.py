import os

models=["AE", "VAE", "Modular", "GNN"]#, "LSTM", "RIM", "SCOFF"]

encoders=["medium"]
losses=["NLL", "Contrastive"]

modes=["Train"]
#modes=["Train", "Test-v1", "Test-v2", "Test-v3", "ZeroShotShape", "ZeroShot"]

embs=[32]
#embs=[4, 32]

cmap="Blues"
#cmap="Sets"

steps=[1,5,10]

for step in steps:
    for emb in embs:
        for mode in modes:
            for loss in losses:
                for model in models:
                    for run in range(1,4):
                        for encoder in encoders:
                            for i in [3,5]:
                                 os.system(f"sbatch --gres=gpu:1 --mem=16G --account=rrg-bengioy-ad --time=3:0:0 eval_rl.sh {i} {model} {encoder} {cmap} {run} {loss} {mode} {emb} {step}")
