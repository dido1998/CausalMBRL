import os

model="Modular"
encoders=["small", "medium", "large"]
cmap="Blues"

for encoder in encoders:
    if encoder is "small":
        bs = 1024
    elif encoder is "medium":
        bs = 512
    else:
        bs = 256
    for i in [3,5]:
        os.system("sbatch --gres=gpu:1 --mem=12G --account=rrg-bengioy-ad --time=72:0:0 run.sh  {} {} {} {} {}".format(str(i), model, encoder, str(bs), cmap))
