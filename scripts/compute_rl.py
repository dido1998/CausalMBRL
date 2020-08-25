import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="Models")
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--mode', type=str, default='Model')
parser.add_argument('--objects', type=str, default=None)
parser.add_argument('--loss', type=str, default=None)
args = parser.parse_args()

folder=args.folder
seeds=args.runs

with open(folder+'/eval_rl_1.txt', 'r') as f:
    lines1 = f.read().split('\n')[:-1]

with open(folder+'/eval_rl_5.txt', 'r') as f:
    lines5 = f.read().split('\n')[:-1]

with open(folder+'/eval_rl_10.txt', 'r') as f:
    lines10 = f.read().split('\n')[:-1]

metrics = dict()

for l in [lines1, lines5, lines10]:
    for line in l:
        x = line.split(':')
        y = x[0].split('/')
        seed = int(y[0][-1])
        model = y[-1]

        if model not in metrics.keys():
            metrics[model] = np.zeros([seeds,18]) - np.inf

        scores = [float(i) for i in x[-1].split('|')[:-1]]
        if args.mode == "Random":
            scores = scores[:6]
        elif args.mode == "Model":
            scores = scores[6:12]
        else:
            if "Contrastive" in model:
                scores = np.zeros(6) - np.inf
            else:
                scores = scores[12:]

        if l is lines1:
            metrics[model][seed-1][:6] = scores
        elif l is lines5:
            metrics[model][seed-1][6:12] = scores
        elif l is lines10:
            metrics[model][seed-1][12:18] = scores
        
for key in sorted(metrics.keys()):
    if args.objects is not None:
        if args.objects not in key:
            continue

    if args.loss is not None:
        if args.loss not in key:
            continue

    metrics[key] = metrics[key]

    means = np.around(np.mean(metrics[key], axis=0), decimals=2)
    stds = np.around(np.std(metrics[key], axis=0), decimals=2)

    string=' & ' + key.split('_')[0] + ' & '
    for i, (m,s) in enumerate(zip(means, stds)):
        if (i // 3) % 2 != 0:
            string += '\\g{'+str(m)+'}{'+str(s)+'} & '
    
    string=string[:-2] + '\\\\'

    print(string)
