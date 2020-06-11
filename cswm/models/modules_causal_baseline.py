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
from .rnn_models_wiki import RNNModel



class CausalTransitionModel(nn.Module):
    """Main module for a Causal transition model.
    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, input_shape=[3, 50, 50],
                 predict_diff=True, encoder='large', num_graphs=10,
                 modular=False, learn_edges=False, vae=False,
                 multiplier=1):

        super(CausalTransitionModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.input_shape = input_shape
        self.modular = modular
        self.learn_edges = learn_edges
        self.predict_diff = predict_diff
        self.num_graphs = num_graphs
        self.vae = vae

        self.mse_loss = torch.nn.MSELoss(reduction='none')

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if self.modular:
            self.embedding_dim = embedding_dim // num_objects
            flat = False
        else:
            self.embedding_dim = embedding_dim
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

            self.transition_nets = nn.ModuleList()

            for i in range(self.num_objects):
                net = MLPTransition(
                    state_dim=self.embedding_dim*self.num_objects,
                    num_actions=self.num_objects * self.action_dim,
                    hidden_dim=self.hidden_dim//self.num_objects * multiplier, #change this to overparameterize transition model
                    output_dim=self.embedding_dim
                    )
                self.transition_nets.append(net)
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

    def func_parameters(self):
        s = set(self.structural_parameters())
        return (p for p in super().parameters() if p not in s)

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
            return z, -0.5 * torch.sum(1 + logvar - \
                mu.pow(2) - logvar.exp())
        else:
            return enc, 0.0

    def modular_transition(self, state, action, next_state=None):
        loss = []

        pred_next_state = []
        for i in range(self.num_objects):
            ins = state.view(state.shape[0], -1)
            pred_ = self.transition_nets[i](ins, action)
            pred_next_state.append(pred_)

        pred_next_state = torch.stack(pred_next_state)
        pred_next_state = pred_next_state.permute(1, 0, 2).contiguous()

        return pred_next_state

    def transition(self, state, action, next_state=None):
        if self.modular:
            pred_next_state = self.modular_transition(
                state, action, next_state)
        else:
            pred_next_state = self.transition_nets(state=state, action=action)

        if self.predict_diff:
            pred_next_state += state
        if next_state is None:
            loss = None
        else:
            loss = self.mse_loss(pred_next_state, next_state)

        if next_state is None:
            return pred_next_state
        else:
            return loss.mean(), None, pred_next_state

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class CausalTransition(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim,
                 num_objects, num_actions,
                 action_embed_dim=128, act_fn='relu'):
        super(CausalTransition, self).__init__()
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.input_dim = input_dim + self.action_embed_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.num_objects = num_objects
        self.act_fc1 = nn.Linear(
                    self.num_actions, self.action_embed_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins, action):
        h_flat = ins.view(ins.shape[0], -1)
        act_embed = self.act_fc1(action)
        h_flat = torch.cat((h_flat, act_embed), -1)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)
        return h


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
            activation0=nn.ReLU(),
            layer1=nn.Linear(2 * hidden_dim, hidden_dim),
            norm1=nn.LayerNorm(hidden_dim),
            activation1=nn.ReLU(),
            layer2=nn.Linear(hidden_dim, hidden_dim * 2),
            norm2=nn.LayerNorm(hidden_dim * 2),
            activation2=nn.ReLU(),
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


class DecoderMLP(nn.Module):
    """MLP decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.output_size = output_size

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        obj_ids = torch.arange(self.num_objects)
        obj_ids = utils.to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h = torch.cat((ins, obj_ids), -1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1],
                      self.output_size[2])


class DecoderCNNSmall1(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu', flat_state=False):
        super(DecoderCNNSmall1, self).__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        if flat_state:
            output_dim = num_objects * width * height
        else:
            output_dim = width * height

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.network = nn.Sequential(OrderedDict(
            fc1=nn.Linear(input_dim, hidden_dim),
            act1=utils.get_act_fn(act_fn),
            fc2=nn.Linear(hidden_dim, hidden_dim),
            ln=nn.LayerNorm(hidden_dim),
            act2=utils.get_act_fn(act_fn),
            fc3=nn.Linear(hidden_dim, output_dim),
            view=View(self.num_objects, width, height),
            deconv1=nn.ConvTranspose2d(num_objects, hidden_dim,
                                       kernel_size=1, stride=1),
            act3=utils.get_act_fn(act_fn),
            deconv2=nn.ConvTranspose2d(hidden_dim, output_size[0],
                                       kernel_size=10, stride=10)
        ))

    def forward(self, ins):
        return self.network(ins)


class CausalLSTM(nn.Module):
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, state_dim=32, input_shape=[3, 50, 50],
                 predict_diff=True, encoder='large', num_graphs=10,
                 modular=False, learn_edges=False, vae=False):
        super().__init__()
        model = CausalTransitionModel(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim,
            input_dims=input_shape,
            input_shape=input_shape,
            num_graphs=args.num_graphs,
            modular=args.modular,
            predict_diff=args.predict_diff,
            learn_edges=args.learn_edges,
            vae=args.vae,
            num_objects=args.num_objects,
            encoder=args.encoder).cuda()
        if modular == False:
            input_size = 32
        else:
            input_size = 25
        self.lstm = nn.LSTM(input_size, 300)




class CausalTransitionModelLSTM(nn.Module):
    """Main module for a Causal transition model.

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, state_dim=32, input_shape=[3, 50, 50],
                 predict_diff=True, encoder='large', num_graphs=10,
                 modular=False, learn_edges=False, vae=False, rim = False, rules = False):

        super(CausalTransitionModelLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.input_shape = input_shape
        self.modular = modular
        self.learn_edges = learn_edges
        self.predict_diff = predict_diff
        self.num_graphs = num_graphs
        self.vae = vae


        self.mse_loss = torch.nn.MSELoss(reduction='none')

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if self.modular:
            self.embedding_dim = embedding_dim // num_objects
            flat = False
        else:
            self.embedding_dim = embedding_dim
            flat = True
        print(self.embedding_dim)

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

            self.transition_nets = nn.ModuleList()

            for i in range(self.num_objects):
                net = MLPTransition(
                    state_dim=self.embedding_dim*self.num_objects,
                    num_actions=self.num_objects * self.action_dim,
                    hidden_dim=self.hidden_dim//self.num_objects * multiplier, #change this to overparameterize transition model
                    output_dim=self.embedding_dim
                    )
                self.transition_nets.append(net)
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
            self.rim = False
            if rim == True:
                self.rim = True
                self.transition_nets = RNNModel('LSTM', self.embedding_dim + self.num_objects * self.action_dim, self.embedding_dim + self.num_objects * self.action_dim, [400], 1,  num_blocks = [5], topk = [3], use_rules = rules)
                self.transition_linear = nn.Linear(400, self.embedding_dim)
            else:
                self.transition_nets = nn.LSTMCell(self.embedding_dim + self.num_objects * self.action_dim, 512)
                print('LSTM')
                self.transition_linear = nn.Linear(512, self.embedding_dim)

        self.encoder = nn.Sequential(OrderedDict(
            obj_extractor=obj_extractor,
            obj_encoder=obj_encoder))

        self.width = width_height[0]
        self.height = width_height[1]

    def structural_parameters(self):
        return iter([self.gamma])

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

    def func_parameters(self):
        s = set(self.structural_parameters())
        return (p for p in super().parameters() if p not in s)

    def encode(self, obs):
        enc = self.encoder(obs)
        if self.vae:
            mu = enc[:, :self.state_dim]
            logvar = enc[:, self.state_dim:]
            if self.training:
                sigma = torch.exp(0.5 * logvar)
                eps = torch.randn_like(sigma)
                z = mu + eps * sigma
            else:
                z = mu
            return z, -0.5 * torch.sum(1 + logvar - \
                mu.pow(2) - logvar.exp())
        else:
            return enc, 0.0

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

    def causal_model(self, state, action, next_state=None):
        # iterate through all M MLPs
        # if no learned edges, then no need to learn gamma

        gamma_itr = 1
        if self.learn_edges:
            gamma_itr = self.num_graphs
        
        gammaexp = self.graph_sample(gamma_itr)
        gammagrads = []
        loss = []
        gamma_losses = []
        # TODO: if not learning edges: then only iterate once!

        for gamma_i in range(gamma_itr):
            pred_next_state = []
            for i in range(self.num_objects):
                ins = gammaexp[gamma_i,:,i].view(1, -1, 1) * state
                ins = ins.reshape(ins.shape[0], -1)
                #ins = state.reshape(state.shape[0], -1)
                pred_ = self.transition_nets[i](ins, action)
                pred_next_state.append(pred_)

            pred_next_state = torch.stack(pred_next_state)
            pred_next_state = pred_next_state.permute(1, 0, 2)

            if self.predict_diff:
                pred_next_state += state

            if next_state is not None:
                mse_loss = self.mse_loss(pred_next_state, next_state)
                mse_loss = torch.stack(tuple(mse_loss.mean(-1).mean(0)))
                loss.append(mse_loss)

            if self.learn_edges:
                gammagrads.append(self.gamma.sigmoid() - gammaexp[gamma_i])

        if next_state is None:
            loss = None
        else:
            loss = torch.stack(loss)
        dRdgamma = torch.zeros([self.num_objects, self.num_objects])

        if self.learn_edges and next_state is not None:
            gammagrads = torch.stack(gammagrads)
            norm_loss = loss.softmax(0)
            dRdgamma = torch.einsum("kij,ki->ij", gammagrads, norm_loss)

        return loss, dRdgamma, pred_next_state

    def transition(self, state, action, hidden, next_state=None):
        """if self.modular:
            # learned causal model
            loss, dRdgamma, pred_next_state = self.causal_model(
                state, action, next_state)
        if not self.modular and not self.learn_edges:
            # baseline MLP
            pred_next_state = self.transition_nets(state=state, action=action)
            if self.predict_diff:
                pred_next_state += state
            dRdgamma = None
            if next_state is None:
                loss = None
            else:
                loss = self.mse_loss(pred_next_state, next_state)

        if next_state is None:
            return pred_next_state
        else:
            return loss.mean(), dRdgamma, pred_next_state"""
        x_orig = state
        x = torch.cat((state, action), dim =1)
        if self.rim:
            x = x.unsqueeze(0)

            x, hidden, _,_,_,_,_ = self.transition_nets(x, hidden.detach())
            x = x.squeeze(0)
            x = self.transition_linear(x)
        else:
            h, c = self.transition_nets(x, hidden.detach())
            x = self.transition_linear(h)
        if self.predict_diff:
            x = x + x_orig
        if next_state is not None:
            loss = self.mse_loss(x, next_state)
            return x, hidden, loss.mean()
        else:
            return x, hidden





    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))