# Ingredients for Causal Learning in model-based RL


## Generate data

### Chemistry Env
```
sh scripts/chem_data.sh num_obj num_color graph max_steps movement

graph: This can be either a name of the graph from the predefined graphs, or you can specify the structure like this: 1-\>2,2-\>3
max_steps: Always specify 10
movement: Static: The positions are fixed across all episodes.
          Dynamic: The positions are varying across all episodes. 
```
The following predefined graphs are available for specification:
```
'chain3':'0->1->2',
'fork3':'0->{1-2}',
'collider3':'{0-1}->2',
'collider4':'{0-2}->3',
'collider5':'{0-3}->4',
'collider6':'{0-4}->5',
'collider7':'{0-5}->6',
'collider8':'{0-6}->7',
'collider9':'{0-7}->8',
'collider10':'{0-8}->9',
'collider11':'{0-9}->10',
'collider12':'{0-10}->11',
'collider13':'{0-11}->12',
'collider14':'{0-12}->13',
'collider15':'{0-13}->14',
'confounder3':'{0-2}->{0-2}',
'chain4':'0->1->2->3',
'chain5':'0->1->2->3->4',
'chain6':'0->1->2->3->4->5',
'chain7':'0->1->2->3->4->5->6',
'chain8':'0->1->2->3->4->5->6->7',
'chain9':'0->1->2->3->4->5->6->7->8',
'chain10':'0->1->2->3->4->5->6->7->8->9',
'chain11':'0->1->2->3->4->5->6->7->8->9->10',
'chain12':'0->1->2->3->4->5->6->7->8->9->10->11',
'chain13':'0->1->2->3->4->5->6->7->8->9->10->11->12',
'chain14':'0->1->2->3->4->5->6->7->8->9->10->11->12->13',
'chain15':'0->1->2->3->4->5->6->7->8->9->10->11->12->13->14',
'full3':'{0-2}->{0-2}',
'full4':'{0-3}->{0-3}',
'full5':'{0-4}->{0-4}',
'full6':'{0-5}->{0-5}',
'full7':'{0-6}->{0-6}',
'full8':'{0-7}->{0-7}',
'full9':'{0-8}->{0-8}',
'full10':'{0-9}->{0-9}',
'full11':'{0-10}->{0-10}',
'full12':'{0-11}->{0-11}',
'full13':'{0-12}->{0-12}',
'full14':'{0-13}->{0-13}',
'full15':'{0-14}->{0-14}',
'tree9':'0->1->3->7,0->2->6,1->4,3->8,2->5',
'tree10':'0->1->3->7,0->2->6,1->4->9,3->8,2->5',
'tree11':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5',
'tree12':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11',
'tree13':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12',
'tree14':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
'tree15':'0->1->3->7,0->2->6->14,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
'jungle3':'0->{1-2}',
'jungle4':'0->1->3,0->2,0->3',
'jungle5':'0->1->3,1->4,0->2,0->3,0->4',
'jungle6':'0->1->3,1->4,0->2->5,0->3,0->4,0->5',
'jungle7':'0->1->3,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6',
'jungle8':'0->1->3->7,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7',
'jungle9':'0->1->3->7,3->8,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8',
'jungle10':'0->1->3->7,3->8,1->4->9,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9',
'jungle11':'0->1->3->7,3->8,1->4->9,4->10,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10',
'jungle12':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11',
'jungle13':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12',
'jungle14':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13',
'jungle15':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,6->14,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13,2->14',
'bidiag3':'{0-2}->{0-2}',
'bidiag4':'{0-1}->{1-2}->{2-3}',
'bidiag5':'{0-1}->{1-2}->{2-3}->{3-4}',
'bidiag6':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}',
'bidiag7':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}',
'bidiag8':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}',
'bidiag9':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}',
'bidiag10':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}',
'bidiag11':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}',
'bidiag12':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}',
'bidiag13':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}',
'bidiag14':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}',
'bidiag15':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}->{13-14}',

```
### Physics Env 

```
python data_gen/env.py --env_id WShapesObserved-Train-<num_obj>-<cmap>-v0 --fname data/wshapes_observed_<num_obj>_<cmap>_train.h5  --num_episodes 1000 --episode-length 100 --seed 1
```

Note: Refer to gen_observed.sh and gen_unobserved.sh to generate data for expts. usage: ./gen_observed.sh <num_obj> <cmap>   ./gen_unobserved.sh <num_obj> <cmap>


## Experiments

### Chemistry
The experiments can be run using the following files:

- scripts/ae_chem.sh
- scripts/vae_chem.sh
- scripts/modular_chem.sh
- scripts/rim_chem.sh
- scrips/scoff_chem.sh

Note that before running the RIM and SCOFF based expts please go into the `modular_dynamics` folder and run `pip install -e .`

The arguments for all the above files are same as follows:
``` 
sh scripts/<model>_chem.sh num_obj num_color graph contrastive_loss max_steps movement

graph: name of the graph(chain5, full5 etc)
contrastive_loss: True or False
max_steps: Always specify 10.
movement: Static or Dynamic.
```
Each run produces 3 folders:

- model-num_obj-num_colors-max_steps-movement-seed-contrastive_loss-1
- model-num_obj-num_colors-max_steps-movement-seed-contrastive_loss-2
- model-num_obj-num_colors-max_steps-movement-seed-contrastive_loss-3

The 3 different folders are for 3 different seeds(1,2,3) for which the models has been run. Note that this seed and the seed provided for argments to the shell is different.


For AE, VAE, Modular (default is AE)
```
python train_causal_baselines.py --dataset <train_file> --encoder <size> --name  <folder_name> --eval-dataset <test_file> \
--num-graphs 10  --embedding-dim <emb-dim --num-objects <num_obj> --action-dim <num_actions>   --valid-dataset <validation_file> --save-folder <save_folder> --batch-size <batch_size> --seed 5 \
--epochs 100 --pretrain-epochs 100 --predict-diff [--vae] [--modular]
```
For LSTM, RIM (default is LSTM)
```
python train_causal_lstm.py --dataset <train_file> --encoder <size> --name  <folder_name> --eval-dataset <test_file> \
--num-graphs 10  --embedding-dim <emb-dim --num-objects <num_obj> --action-dim <num_actions>   --valid-dataset <validation_file> --save-folder <save_folder> --batch-size <batch_size> --seed 5 \
--epochs 100 --pretrain-epochs 100 --predict-diff [--rim]
