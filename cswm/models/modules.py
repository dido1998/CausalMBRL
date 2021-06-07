from collections import OrderedDict
from itertools import chain

import numpy as np
import torch
from torch import nn

from .. import utils
from . import (
    View,
    EncoderCNNSmall, EncoderCNNMedium, EncoderCNNLarge,
    DecoderCNNSmall, DecoderCNNMedium, DecoderCNNLarge)

#from modularity import RIM, SCOFF


class CausalTransitionModel(nn.Module):
    """Main module for a Causal transition model.
    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim_per_object, input_dims, hidden_dim, 
                 action_dim, num_objects, input_shape=[3, 50, 50],
                 predict_diff=True, encoder='large', modular=False, 
                 vae=False, gnn=False, learn_edges = False, num_graphs = 10, multiplier=1, ignore_action=False,
                 causal=False, graph=None, copy_action=False):
        super(CausalTransitionModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.input_shape = input_shape
        self.modular = modular
        self.predict_diff = predict_diff
        self.vae = vae
        self.gnn = gnn
        self.graph = graph
        self.causal = causal
        self.num_graphs = num_graphs
        self.learn_edges = learn_edges
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        # gamma is structural parameter for causal edges
        gamma = torch.zeros(self.num_objects, self.num_objects)
        adj_mask = torch.eye(
                self.num_objects, self.num_objects).byte()
        gamma.masked_fill_(adj_mask, (float("-inf")))
        self.register_parameter("gamma", nn.Parameter(gamma))

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if self.modular or self.gnn:
            self.embedding_dim = embedding_dim_per_object
            flat = False
        else:
            self.embedding_dim = embedding_dim_per_object * num_objects
            flat = True

        if encoder == 'small':
            obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            width_height = np.array(width_height)
            width_height = width_height // 10
            self.decoder = DecoderCNNSmall(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        elif encoder == 'medium':
            obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            width_height = np.array(width_height)
            width_height = width_height // 5
            self.decoder = DecoderCNNMedium(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        elif encoder == 'large':
            obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            self.decoder = DecoderCNNLarge(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        if self.modular or self.gnn:
            obj_encoder = EncoderMLP(
                input_dim=np.prod(width_height),
                hidden_dim=hidden_dim,
                output_dim=self.embedding_dim,
                num_objects=num_objects)
        else:
            if self.vae:
                obj_encoder = EncoderMLP(
                    input_dim=np.product(width_height)*self.num_objects,
                    output_dim=self.embedding_dim * 2,
                    hidden_dim=hidden_dim, num_objects=num_objects,
                    flatten_input=True)
            else:
                obj_encoder = EncoderMLP(
                    input_dim=np.product(width_height)*self.num_objects,
                    output_dim=self.embedding_dim,
                    hidden_dim=hidden_dim, num_objects=num_objects,
                    flatten_input=True)

        if self.modular:
            self.transition_nets = nn.ModuleList()

            for i in range(self.num_objects):
                net = MLPTransition(
                    state_dim=self.embedding_dim*self.num_objects,
                    num_actions=self.num_objects * self.action_dim,
                    hidden_dim=self.hidden_dim//self.num_objects * multiplier, #change this to overparameterize transition model
                    output_dim=self.embedding_dim
                    )
                self.transition_nets.append(net)
        elif self.gnn:
            self.transition_nets = TransitionGNN(
                input_dim=self.embedding_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_objects=num_objects,
                ignore_action=ignore_action,
                copy_action=copy_action)
        else:
            self.transition_nets = MLPTransition(
                state_dim=self.embedding_dim, output_dim=self.embedding_dim,
                num_actions=self.num_objects * self.action_dim, hidden_dim=self.hidden_dim)

        self.encoder = nn.Sequential(OrderedDict(
            obj_extractor=obj_extractor,
            obj_encoder=obj_encoder))

        self.width = width_height[0]
        self.height = width_height[1]

    def transition_parameters(self):
        parameters = []
        if isinstance(self.transition_nets, list):
            for net in self.transition_nets:
                parameters = chain(parameters, net.parameters())
        else:
            return self.transition_nets.parameters()

        return parameters

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()

    def structural_parameters(self):
        return iter([self.gamma])

    def func_parameters(self):
        s = set(self.structural_parameters())
        return (p for p in super().parameters() if p not in s)

    def graph_sample(self, batch_size):
        # need to sample a batch
        gammaexp_batch = []
        if self.learn_edges:
            for batch_itr in range(batch_size):
                with torch.no_grad():
                    gammaexp = self.gamma.sigmoid()
                    gammaexp = torch.empty_like(gammaexp).uniform_().lt_(gammaexp)
                    gammaexp += torch.eye(self.num_objects).cuda()
                    gammaexp_batch.append(gammaexp)
            return torch.stack(gammaexp_batch).cuda()
        else:
            gammaexp = torch.ones([self.num_objects,self.num_objects])
            #gammaexp.diagonal().ones_()
            #gammaexp = torch.eye(self.num_objects)
            gammaexp = gammaexp.unsqueeze(0)
            gammaexp = gammaexp.repeat(batch_size, 1, 1)
            return gammaexp.cuda()

    def causal_model(self, state, action):
        # iterate through all M MLPs
        # if no learned edges, then no need to learn gamma
        gamma_itr = 1
        if self.learn_edges:
            gamma_itr = self.num_graphs
            gammaexp = self.graph_sample(gamma_itr)
            gammagrads = []
            gamma_losses = []
        else:
            gammaexp = self.graph + torch.eye(self.graph.shape[0])  
            gammaexp = gammaexp.cuda()
        loss = []
        

        # TODO: if not learning edges: then only iterate once!
        pred_next_state_ = []

        for gamma_i in range(gamma_itr):
            pred_next_state = []
            for i in range(self.num_objects):
                #print(gammaexp[gamma_i,:,i].view(1, -1, 1).size())
                #print(state.size())
                if gamma_itr > 1:
                    gamma = gammaexp[gamma_i,:,i]
                else:
                    gamma = gammaexp[i]
                ins = gamma.view(1, -1, 1) * state #.view(state.size(0), self.num_objects, -1)
                ins = ins.reshape(ins.shape[0], -1)
                #print(ins.shape)
                #ins = state.reshape(state.shape[0], -1)
                pred_ = self.transition_nets[i](ins, action)
                #print('transition;' + str(pred_.size()))
                pred_next_state.append(pred_)

            pred_next_state = torch.stack(pred_next_state)
            pred_next_state = pred_next_state.permute(1, 0, 2)
            
            #print(pred_next_state.size())

            if self.predict_diff:
                pred_next_state += state
            pred_next_state_.append(pred_next_state)

            #if next_state is not None:
            #    mse_loss = self.mse_loss(pred_next_state, next_state)
            #    mse_loss = torch.stack(tuple(mse_loss.mean(-1).mean(0)))
            #    loss.append(mse_loss)

            #if self.learn_edges:
            #   gammagrads.append(self.gamma.sigmoid() - gammaexp[gamma_i])

        #if next_state is None:
        #    loss = None
        #else:
        #    loss = torch.stack(loss)
        #dRdgamma = torch.zeros([self.num_objects, self.num_objects])

        #if self.learn_edges and next_state is not None:
        #    gammagrads = torch.stack(gammagrads)
        #    norm_loss = loss.softmax(0)
        #    dRdgamma = torch.einsum("kij,ki->ij", gammagrads, norm_loss)

        return pred_next_state, pred_next_state_, gammaexp

    def encode(self, obs):
        enc = self.encoder(obs)
        if self.vae:
            mu = enc[:, :self.embedding_dim]
            logvar = enc[:, self.embedding_dim:]
            if self.training:
                sigma = torch.exp(0.5 * logvar)
                eps = torch.randn_like(sigma)
                z = mu + eps * sigma
            else:
                z = mu
            return z, (mu, logvar)
        else:
            return enc, None

    def modular_transition(self, state, action):
        pred_next_state = []
        for i in range(self.num_objects):
            ins = state.view(state.shape[0], -1)
            pred_ = self.transition_nets[i](ins, action)
            pred_next_state.append(pred_)

        pred_next_state = torch.stack(pred_next_state)
        pred_next_state = pred_next_state.permute(1, 0, 2).contiguous()

        return pred_next_state

    def transition(self, state, action):
        if self.causal:
            pred_next_state, pred_next_states, gamma_exp = self.causal_model(state, action)
            if self.learn_edges:
                return pred_next_state, pred_next_states, gamma_exp
            else:
                return pred_next_state
        if self.modular:
            pred_next_state = self.modular_transition(
                state, action)
        elif self.gnn:
            action = torch.argmax(action, dim=1)
            pred_next_state = self.transition_nets(state, action)
        else:
            pred_next_state = self.transition_nets(state=state, action=action)

        if self.predict_diff:
            pred_next_state += state

        return pred_next_state

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))

class CausalTransitionModelLSTM(nn.Module):
    """Main module for a Recurrent Causal transition model.
    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
        rim: If False uses LSTM else RIMs (goyal et al)
    """
    def __init__(self, embedding_dim_per_object, input_dims, hidden_dim, action_dim,
                 num_objects, input_shape=[3, 50, 50], predict_diff=True, encoder='large',
                 graph=None, modular=False, vae=False, rim = False, scoff = False,
                 multiplier=1):
        super(CausalTransitionModelLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.input_shape = input_shape
        self.modular = modular
        self.predict_diff = predict_diff
        self.vae = vae
        self.graph = graph

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if self.modular:
            self.embedding_dim = embedding_dim_per_object
            flat = False
        else:
            self.embedding_dim = embedding_dim_per_object * num_objects
            flat = True

        if encoder == 'small':
            obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            width_height = np.array(width_height)
            width_height = width_height // 10
            self.decoder = DecoderCNNSmall(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        elif encoder == 'medium':
            obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            width_height = np.array(width_height)
            width_height = width_height // 5
            self.decoder = DecoderCNNMedium(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        elif encoder == 'large':
            obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            self.decoder = DecoderCNNLarge(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        if self.modular:
            obj_encoder = EncoderMLP(
                input_dim=np.prod(width_height),
                hidden_dim=hidden_dim,
                output_dim=self.embedding_dim,
                num_objects=num_objects)
        else:
            if self.vae:
                obj_encoder = EncoderMLP(
                    input_dim=np.product(width_height)*self.num_objects,
                    output_dim=self.embedding_dim * 2,
                    hidden_dim=hidden_dim, num_objects=num_objects,
                    flatten_input=True)
            else:
                obj_encoder = EncoderMLP(
                    input_dim=np.product(width_height)*self.num_objects,
                    output_dim=self.embedding_dim,
                    hidden_dim=hidden_dim, num_objects=num_objects,
                    flatten_input=True)
            
            self.action_encoder = nn.Linear(self.num_objects * self.action_dim, self.hidden_dim)
            self.obs_encoder = nn.Linear(self.embedding_dim, self.hidden_dim)
            self.lstm = False
            if rim == True:
                self.rim = True
                self.transition_nets =  RIM('cuda', 2 * self.hidden_dim, 600, 6, 4, rnn_cell = 'GRU', n_layers = 1, bidirectional = False)
                self.transition_linear = nn.Linear(600, self.embedding_dim)
            elif scoff == True:
                self.transition_nets = SCOFF('cuda', 2 * self.hidden_dim, 600, 4, 3, num_templates = 2, rnn_cell = 'GRU', n_layers = 1, bidirectional = False, version=1)
                self.transition_linear = nn.Linear(600, self.embedding_dim)
            else:
                self.lstm = True
                self.transition_nets = nn.LSTM(2 * self.hidden_dim, 600)
                self.transition_linear = nn.Linear(600, self.embedding_dim)

        self.encoder = nn.Sequential(OrderedDict(
            obj_extractor=obj_extractor,
            obj_encoder=obj_encoder))

        self.width = width_height[0]
        self.height = width_height[1]

    def transition_parameters(self):
        parameters = []
        if isinstance(self.transition_nets, list):
            for net in self.transition_nets:
                parameters = chain(parameters, net.parameters())
        else:
            return list(self.transition_nets.parameters()) + list(self.transition_linear.parameters())

        return parameters

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()

    def encode(self, obs):
        enc = self.encoder(obs)
        if self.vae:
            mu = enc[:, :self.embedding_dim]
            logvar = enc[:, self.embedding_dim:]
            if self.training:
                sigma = torch.exp(0.5 * logvar)
                eps = torch.randn_like(sigma)
                z = mu + eps * sigma
            else:
                z = mu
            return z, (mu, logvar)
        else:
            return enc, None

    def transition(self, state, action, hidden):
        encoded_action = self.action_encoder(action)
        encoded_state = self.obs_encoder(state)
        
        x = torch.cat((encoded_state, encoded_action), dim = 1)
        x = x.unsqueeze(0)
        if not self.lstm:
            x, hidden, _ = self.transition_nets(x, hidden)
        else:
            x, hidden = self.transition_nets(x, hidden)

        x = self.transition_linear(x)
        x = x.squeeze(0)

        if self.predict_diff:
            x = state + x

        return x, hidden

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))

class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects,
                 ignore_action=False, copy_action=False, act_fn='relu'):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr)

        if not self.ignore_action:

            if self.copy_action:
                action_vec = utils.to_one_hot(
                    action, self.action_dim).repeat(1, self.num_objects)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                action_vec = utils.to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)

class MLPTransition(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, output_dim):
        super().__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.action_encoder = nn.Linear(num_actions, hidden_dim)
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.output_dim = output_dim
        self.decoder = nn.Sequential(OrderedDict(
            activation0=nn.ReLU(inplace=True),
            layer1=nn.Linear(2 * hidden_dim, hidden_dim),
            norm1=nn.LayerNorm(hidden_dim),
            activation1=nn.ReLU(inplace=True),
            layer2=nn.Linear(hidden_dim, hidden_dim * 2),
            norm2=nn.LayerNorm(hidden_dim * 2),
            activation2=nn.ReLU(inplace=True),
            out_layer=nn.Linear(hidden_dim * 2, output_dim)
        ))

    def forward(self, state, action, params=None):
        return self.decoder(torch.cat([
            self.action_encoder(action),
            self.state_encoder(state)
        ], dim=1))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 flatten_input=False, act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim
        self.flatten_input = flatten_input

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        if self.flatten_input:
            h_flat = ins.view(ins.shape[0], -1)
        else:
            h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)

class RewardPredictor(nn.Module):

    def __init__(self, embedding_dim):
        super(RewardPredictor, self).__init__()

        self.embedding_dim = embedding_dim

        self.model = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)
