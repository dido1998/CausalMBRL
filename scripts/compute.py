import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="Models")
parser.add_argument('--runs', type=int, default=3)
args = parser.parse_args()

folder=args.folder
seeds=args.runs

with open(folder+'/eval.txt', 'r') as f:
    lines = f.read().split('\n')[:-1]

metrics = dict()

for line in lines:
    x = line.split(':')
    y = x[0].split('/')
    seed = int(y[0][-1])
    model = y[-1]

    if model not in metrics.keys():
        metrics[model] = np.zeros([seeds,9]) - np.inf

    scores = [float(i) for i in x[-1].split('&')[:-1]]
    metrics[model][seed-1] = scores

for key in sorted(metrics.keys()):
    metrics[key][:,2::3] /= 7500.0

    means = np.around(np.mean(metrics[key], axis=0), decimals=2)
    stds = np.around(np.std(metrics[key], axis=0), decimals=2)

    string=key+' & '
    for i, (m,s) in enumerate(zip(means, stds)):
        if i+1 % 3 == 0 and "Contrastive" in key:
            string += ' - & '
        else:
            string += '\\g{'+str(m)+'}{'+str(s)+'} & '
    
    string=string[:-2]

    print(string)
