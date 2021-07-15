# Systematic Evaluation of Causal Discovery in Visual Model Based Reinforcement Learning

We introduce a novel suite of RL environments as a platform for investigating inductive  biases, causal representations, and learning algorithms. The goal is to disentangle distinct aspects of causal learning by allowing the user to choose and modulate different properties of the ground truth causal graph, such as the structure, the size of the graph and the sparsity of the graph. ). We also provide evaluation criteria for measuring causal induction in MBRL that we argue help measure progress and facilitate further research in these directions. The paper is available at https://arxiv.org/abs/2107.00848.

## Table of Contents

- [Physics Environment](#physics-environment)
    * [Data Generation](#data-generation)
    * [Model Based Experiments](#model-based-experiments)
    * [Reinforcement Learning Experiments](#reinforcement-learning-experiments)
    * [To Reproduce Physics Environment Experiments from the paper](#to-reproduce-physics-environment-experiments-from-the-paper)
- [Chemistry Environment](#chemistry-environment)
    * [Data Generation](#data-generation-1)
    * [Model Based  Experiments](#model-based--experiments)
    * [Reinforcement Learning Experiments](#reinforcement-learning-experiments-1)
    * [To Reproduce Chemistry Environment Experiments from the Paper](#to-reproduce-chemistry-environment-experiments-from-the-paper)
- [Dataset Metadata](#dataset-metadata)

## Physics Environment

### Data Generation
- Observed Physics Environment
 - `bash scrips/gen_observed.sh num_obj Blues`
- Unobserved Physics Environment
 - `bash scripts/gen_unobserved.sh num_obj Sets`
- FixedUnobserved Physics Environment
 - `bash scripts/gen_unobserved_fixed.sh num_obj Sets`
In our experiments we use `num_obj = {3,5}`

### Model Based Experiments

**Observed Physics Environment**

```
bash scripts/run_observed.sh num_obj model_name encoder batch_size cmap seed loss emb_dim
bash scripts/eval_observed.sh num_obj model_name encoder cmap seed loss mode emb_dim

num_obj = number of objects {3,5}
model_name = AE, VAE, Modular, GNN
encoder = medium
batch_size = 512
cmap = Blues
loss = NLL or Contrastive
emb_dim = 128
mode = test-v0
```

**Unobserved Physics Environment**
```
bash scripts/run_unobserved.sh num_obj model_name encoder batch_size cmap seed loss emb_dim
bash scripts/eval_unobserved.sh num_obj model_name encoder cmap seed loss mode emb_dim

num_obj = number of objects {3,5}
model_name = AE, VAE, Modular, GNN
encoder = medium
batch_size = 512
cmap = Sets
loss = NLL or Contrastive
emb_dim = 128
mode = test
```

**FixedUnobserved Physics Environment**
```
bash scripts/run_fixed_unobserved.sh num_obj model_name encoder batch_size cmap seed loss emb_dim
bash scripts/eval_fixed_unobserved.sh num_obj model_name encoder cmap seed loss mode emb_dim

num_obj = number of objects {3,5}
model_name = AE, VAE, Modular, GNN
encoder = medium
batch_size = 512
cmap = Sets
loss = NLL or Contrastive
emb_dim = 128
mode = test
```



### Reinforcement Learning Experiments

The below scripts run the reinforcement learning experiments for the above trained models. 

**Observed Physics Environment**
```
# This scripts will automatically load the pre-trained model with above arguments. 
bash scripts/run_reward_observed.sh num_obj model_name encoder batch_size cmap seed loss emb_dim
bash scripts/eval_rl_observed.sh num_obj model_name encoder cmap seed loss mode emb_dim steps

num_obj = {3,5}
model_name = AE, VAE, Modular, GNN
batch_size = 32
cmap = Blues
loss = NLL or Contrastive
emb_dim = 128
mode = test-v0
steps = {1,5,10}

```


**Unobserved Physics Environment**
```
# This scripts will automatically load the pre-trained model with above arguments. 
bash scripts/run_reward_unobserved.sh num_obj model_name encoder batch_size cmap seed loss emb_dim
bash scripts/eval_rl_unobserved.sh num_obj model_name encoder cmap seed loss mode emb_dim steps

num_obj = {3,5}
model_name = AE, VAE, Modular, GNN
batch_size = 32
cmap = Sets
loss = NLL or Contrastive
emb_dim = 128
mode = test
steps = {1,5,10}

```

**FixedUnobserved Physics Environment**
```
# This scripts will automatically load the pre-trained model with above arguments. 
bash scripts/run_reward_fixed_unobserved.sh num_obj model_name encoder batch_size cmap seed loss emb_dim
bash scripts/eval_rl_fixed_unobserved.sh num_obj model_name encoder cmap seed loss mode emb_dim steps

num_obj = {3,5}
model_name = AE, VAE, Modular, GNN
batch_size = 32
cmap = Sets
loss = NLL or Contrastive
emb_dim = 128
mode = test
steps = {1,5,10}

```


### To Reproduce Physics Environment Experiments from the paper 
```
# Generate Date
bash scripts/gen_observed.sh 3 Blues
bash scripts/gen_observed.sh 5 Blues

bash scripts/gen_unobserved.sh 3 Sets
bash scripts/gen_unobserved.sh 5 Sets

bash scripts/gen_unobserved_fixed.sh 3 Sets
bash scripts/gen_unobserved_fixed.sh 5 Sets


# Model Based Experiments
## Observed Physics Environment
### These 8 experiments are run for model_name = AE, VAE, Modular, GNN
bash scripts/run_observed.sh 3 AE medium 512 Blues 0 NLL 128
bash scripts/eval_observed.sh 3 AE medium Blues 0 NLL test-v0 128

bash scripts/run_observed.sh 3 AE medium 512 Blues 0 Contrastive 128
bash scripts/eval_observed.sh 3 AE medium Blues 0 Contrastive test-v0 128


bash scripts/run_observed.sh 5 AE medium 512 Blues 0 NLL 128
bash scripts/eval_observed.sh 5 AE medium Blues 0 NLL test-v0 128

bash scripts/run_observed.sh 5 AE medium 512 Blues 0 Contrastive 128
bash scripts/eval_observed.sh 5 AE medium Blues 0 Contrastive test-v0 128

## Unobserved Physics Environment
### These 8 experiments are run for model_name = AE, VAE, Modular, GNN
bash scripts/run_unobserved.sh 3 AE medium 512 Sets 0 NLL 128
bash scripts/eval_unobserved.sh 3 AE medium Sets 0 NLL test 128

bash scripts/run_unobserved.sh 3 AE medium 512 Sets 0 Contrastive 128
bash scripts/eval_unobserved.sh 3 AE medium Sets 0 Contrastive test 128


bash scripts/run_unobserved.sh 5 AE medium 512 Sets 0 NLL 128
bash scripts/eval_unobserved.sh 5 AE medium Sets 0 NLL test 128

bash scripts/run_unobserved.sh 5 AE medium 512 Sets 0 Contrastive 128
bash scripts/eval_unobserved.sh 5 AE medium Sets 0 Contrastive test 128


## FixedUnobserved Physics Environment
### These 8 experiments are run for model_name = AE, VAE, Modular, GNN
bash scripts/run__fixed_unobserved.sh 3 AE medium 512 Sets 0 NLL 128
bash scripts/eval_fixded_unobserved.sh 3 AE medium Sets 0 NLL test 128

bash scripts/run_fixed_unobserved.sh 3 AE medium 512 Sets 0 Contrastive 128
bash scripts/eval_fixed_unobserved.sh 3 AE medium Sets 0 Contrastive test 128


bash scripts/run_fixed_unobserved.sh 5 AE medium 512 Sets 0 NLL 128
bash scripts/eval_fixed_unobserved.sh 5 AE medium Sets 0 NLL test 128

bash scripts/run_fixed_unobserved.sh 5 AE medium 512 Sets 0 Contrastive 128
bash scripts/eval_fixed_unobserved.sh 5 AE medium Sets 0 Contrastive test 128


# Reinforcement Learning 
## The below experiments can be repeated for model_name = {AE, VAE. Modular, GNN}, loss = {NLL, Contrastive}, num_obj = {3,5}, environments = {Observed, Unobserved, FixedUnobserved}
bash scripts/run_reward_observed.sh 3 AE medium 512 Blues 0 NLL 128
bash scripts/eval_rl_observed.sh 3 AE medium Blues 0 NLL Train 128 1
bash scripts/eval_rl_observed.sh 3 AE medium Blues 0 NLL Train 128 5
bash scripts/eval_rl_observed.sh 3 AE medium Blues 0 NLL Train 128 10
```


## Chemistry Environment

### Data Generation
```
bash scripts/chem_data.sh num_obj num_color graph max_steps movement

num_obj = 5
num_color = 5
graph = chain<num_obj>, full<num_obj>, collider<num_obj>. For example: chain5, full5, collider5
max_steps = 10
movement = Static = The positions are fixed across episodes.
          Dynamic = The positions are varying across episodes. 
```

### Model Based  Experiments 
```
bash scripts/run_chem.sh num_obj model_name encoder batch_size num_colors max_steps movement graph seed loss emb_dim
bash scripts/eval_chem.sh num_obj model_name encoder num_colors max_steps movement graph seed loss mode emb_dim


num_obj = 5
model_name = AE, VAE, Modular, GNN
encoder = medium
batch_size = 512
num_colors = 5
max_steps = 10
movement = {Static, Dynamic}
graph = chain<num_obj>, full<num_obj>, collider<num_obj>. For example: chain5, full5, collider5
loss = {NLL, Contrastive}
emb_dim = 128
mode = test
```

### Reinforcement Learning Experiments
```
bash scripts/run_chem_reward.sh num_obj model_name encoder batch_size num_colors max_steps movement graph seed loss emb_dim
bash scripts/eval_rl_chem.sh num_obj model_name encoder num_colors max_steps movement graph seed loss mode emb_dim steps


num_obj = 5
model_name = AE, VAE, Modular, GNN
encoder = medium
batch_size = 512
num_colors = 5
max_steps = 10
movement = {Static, Dynamic}
graph = chain<num_obj>, full<num_obj>, collider<num_obj>. For example: chain5, full5, collider5
loss = {NLL, Contrastive}
emb_dim = 128
mode = test
steps = {1, 5, 10}

```

### To Reproduce Chemistry Environment Experiments from the Paper
```
# Generate Data
bash scripts/chem_data.sh 5 5 chain5 10 Static
bash scripts/chem_data.sh 5 5 full5 10 Static
bash scripts/chem_data.sh 5 5 collider5 10 Static

bash scripts/chem_data.sh 5 5 chain5 10 Dynamic
bash scripts/chem_data.sh 5 5 full5 10 Dynamic
bash scripts/chem_data.sh 5 5 collider5 10 Dynamic


# Model Based Experiments
## Repeat the below experiments for model_name = {AE, VAE, Modular, GNN}
bash scripts/run_chem.sh 5 AE medium 512 5 10 Dynamic chain5 0 NLL 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Dynamic chain5 0 NLL test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Dynamic full5 0 NLL 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Dynamic full5 0 NLL test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Dynamic collider5 0 NLL 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Dynamic collider5 0 NLL test 128


bash scripts/run_chem.sh 5 AE medium 512 5 10 Dynamic chain5 0 Contrastive 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Dynamic chain5 0 Contrastive test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Dynamic full5 0 Contrastive 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Dynamic full5 0 Contrastive test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Dynamic collider5 0 Contrastive 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Dynamic collider5 0 Contrastive test 128


bash scripts/run_chem.sh 5 AE medium 512 5 10 Static chain5 0 NLL 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Static chain5 0 NLL test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Static full5 0 NLL 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Static full5 0 NLL test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Static collider5 0 NLL 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Static collider5 0 NLL test 128


bash scripts/run_chem.sh 5 AE medium 512 5 10 Static chain5 0 Contrastive 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Static chain5 0 Contrastive test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Static full5 0 Contrastive 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Static full5 0 Contrastive test 128

bash scripts/run_chem.sh 5 AE medium 512 5 10 Static collider5 0 Contrastive 128
bash scripts/eval_chem.sh 5 AE medium 5 10 Static collider5 0 Contrastive test 128


# Reinforcement Learning Experiments
## Repeat the below experiments for model_name = {AE, VAE, Modular, GNN}
bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Dynamic chain5 0 NLL 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic chain5 0 NLL Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic chain5 0 NLL Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic chain5 0 NLL Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Dynamic full5 0 NLL 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic full5 0 NLL Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic full5 0 NLL Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic full5 0 NLL Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Dynamic collider5 0 NLL 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic collider5 0 NLL Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic collider5 0 NLL Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic collider5 0 NLL Train 128 10


bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Dynamic chain5 0 Contrastive 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic chain5 0 Contrastive Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic chain5 0 Contrastive Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic chain5 0 Contrastive Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Dynamic full5 0 Contrastive 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic full5 0 Contrastive Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic full5 0 Contrastive Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic full5 0 Contrastive Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Dynamic collider5 0 Contrastive 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic collider5 0 Contrastive Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic collider5 0 Contrastive Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Dynamic collider5 0 Contrastive Train 128 10


bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Static chain5 0 NLL 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static chain5 0 NLL Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static chain5 0 NLL Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static chain5 0 NLL Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Static full5 0 NLL 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static full5 0 NLL Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static full5 0 NLL Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static full5 0 NLL Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Static collider5 0 NLL 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static collider5 0 NLL Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static collider5 0 NLL Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static collider5 0 NLL Train 128 10


bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Static chain5 0 Contrastive 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static chain5 0 Contrastive Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static chain5 0 Contrastive Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static chain5 0 Contrastive Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Static full5 0 Contrastive 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static full5 0 Contrastive Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static full5 0 Contrastive Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static full5 0 Contrastive Train 128 10

bash scripts/run_chem_reward.sh 5 AE medium 512 5 10 Static collider5 0 Contrastive 128
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static collider5 0 Contrastive Train 128 1
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static collider5 0 Contrastive Train 128 5
bash scripts/eval_rl_chem.sh 5 AE medium 5 10 Static collider5 0 Contrastive Train 128 10

```

## Dataset Metadata
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">CausalMBRL</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">CMBRL</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/dido1998/CausalMBRL</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">The CausalMBRL dataset is created to test the causal learning abilities of model-based
      reinforcement learning agents. It contains 2 environments: The Physics Environment and The Chemistry Environment. 
      Both the environments consist of blocks of various colors, shapes, and weights placed in a grid. For both the environments,
      there exists a ground-truth causal graph which dictates how the blocks interact. This graph is unknown to the model and should
      be discovered by it through interactions with the environment.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">MILA, University of Montreal</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://mila.quebec/en/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  
</table>
</div>

