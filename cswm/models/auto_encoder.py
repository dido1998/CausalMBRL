from collections import OrderedDict

import torch
from enum import Enum
from torch import nn

from . import (
    DecoderCNNSmall, DecoderCNNMedium, DecoderCNNLarge,
    EncoderMLP,
    EncoderCNNSmall, EncoderCNNMedium, EncoderCNNLarge)


class Encoders(Enum):
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'


class Decoders(Enum):
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'


class GridworldAutoEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=256, state_dim=32,
                 out_features=(3, 50, 50), *,
                 dropout=False, encoder_type=Encoders.SMALL,
                 decoder_type=Decoders.SMALL):
        super().__init__()

        encoder_out_size = state_dim
        if dropout:
            state_dim = state_dim * 2

        encoder = {
            Encoders.SMALL: EncoderCNNSmall,
            Encoders.MEDIUM: EncoderCNNMedium,
            Encoders.LARGE: EncoderCNNLarge}[encoder_type](
                in_channels, hidden_size, encoder_out_size)

        mlp_input_size = {
            Encoders.SMALL: state_dim,
            Encoders.MEDIUM: state_dim * 4,
            Encoders.LARGE: state_dim * 100,
        }[encoder_type]

        self.encoder = nn.Sequential(OrderedDict(
            cnn_encoder=encoder,
            object_encoder=EncoderMLP(
                input_dim=mlp_input_size, output_dim=state_dim,
                hidden_dim=125, num_objects=25,
                flatten_input=True,
                dropout=dropout),
        ))

        self.decoder = {Decoders.SMALL: DecoderCNNSmall,
                        Decoders.MEDIUM: DecoderCNNMedium,
                        Decoders.LARGE: DecoderCNNLarge}[decoder_type](
                input_dim=state_dim, hidden_dim=hidden_size,
                num_objects=5, output_size=out_features,
                flat_state=True)

        self.transition = MLPTransition(
            state_dim=state_dim, num_actions=20, hidden_dim=512)

    def forward(self, obs, action):
        state = self.encoder(obs)
        rec = torch.sigmoid(self.decoder(state))

        next_state_pred = self.transition(state=state, action=action)
        next_rec = torch.sigmoid(self.decoder(next_state_pred))
        return state, rec, next_state_pred, next_rec


class MLPTransition(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim):
        super().__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_encoder = nn.Linear(num_actions, hidden_dim)
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.decoder = nn.Sequential(OrderedDict(
            activation0=nn.ReLU(),
            layer1=nn.Linear(2 * hidden_dim, hidden_dim),
            norm1=nn.LayerNorm(hidden_dim),
            activation1=nn.ReLU(),
            layer2=nn.Linear(hidden_dim, hidden_dim * 2),
            norm2=nn.LayerNorm(hidden_dim * 2),
            activation2=nn.ReLU(),
            out_layer=nn.Linear(hidden_dim * 2, state_dim)
        ))

    def forward(self, state, action, params=None):
        return state + self.decoder(torch.cat([
            self.action_encoder(action),
            self.state_encoder(state)
        ], dim=1))
