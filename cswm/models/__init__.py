from collections import OrderedDict

import torch
from torch import nn

from .. import utils


__all__ = ("View", "DecoderCNNSmall", "DecoderCNNMedium", "EncoderMLP",
           "EncoderCNNSmall", "EncoderCNNMedium", "EncoderCNNLarge")


class View(nn.Module):
    def __init__(self, *shape):
        self.shape = shape
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class EncoderMLP(nn.Sequential):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu', flatten_input=False, dropout=False):

        self.num_objects = num_objects
        self.input_dim = input_dim

        if flatten_input:
            view = View(self.num_objects * self.input_dim)
            linear_input_dim = self.num_objects * self.input_dim
        else:
            view = View(self.num_objects, self.input_dim)
            linear_input_dim = self.input_dim

        maybe_dropout = dict(dropout=nn.Dropout(0.5)) if dropout else {}

        super().__init__(OrderedDict(
            view=view,
            fc1=nn.Linear(linear_input_dim, hidden_dim),
            act1=utils.get_act_fn(act_fn),
            fc2=nn.Linear(hidden_dim, hidden_dim),
            ln=nn.LayerNorm(hidden_dim),
            act2=utils.get_act_fn(act_fn),
            fc3=nn.Linear(hidden_dim, output_dim),
            **maybe_dropout
        ))

class EncoderCNNSmall(nn.Sequential):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects,
                 act_fn='sigmoid', act_fn_hid='relu'):
        super().__init__(OrderedDict(
            cnn1=nn.Conv2d(
                input_dim, hidden_dim, (10, 10), stride=10),
            ln1=nn.BatchNorm2d(hidden_dim),
            act1=utils.get_act_fn(act_fn_hid),
            cnn2=nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1),
            act2=utils.get_act_fn(act_fn),
        ))


class EncoderCNNMedium(nn.Sequential):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects,
                 act_fn='sigmoid', act_fn_hid='leaky_relu'):
        super().__init__(OrderedDict(
            cnn1=nn.Conv2d(
                input_dim, hidden_dim, (9, 9), padding=4),
            ln1=nn.BatchNorm2d(hidden_dim),
            act1=utils.get_act_fn(act_fn_hid),
            cnn2=nn.Conv2d(
                hidden_dim, num_objects, (5, 5), stride=5),
            act2=utils.get_act_fn(act_fn)
        ))


class EncoderCNNLarge(nn.Sequential):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects,
                 act_fn='sigmoid',
                 act_fn_hid='relu'):
        super().__init__(OrderedDict(
            cnn1=nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1),
            ln1=nn.BatchNorm2d(hidden_dim),
            act1=utils.get_act_fn(act_fn_hid),
            cnn2=nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1),
            ln2=nn.BatchNorm2d(hidden_dim),
            act2=utils.get_act_fn(act_fn_hid),
            cnn3=nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1),
            ln3=nn.BatchNorm2d(hidden_dim),
            act3=utils.get_act_fn(act_fn_hid),
            cnn4=nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1),
            act4=utils.get_act_fn(act_fn),
        ))


class DecoderCNNSmall(nn.Sequential):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu', flat_state=False):

        width, height = output_size[1] // 10, output_size[2] // 10

        if flat_state:
            output_dim = num_objects * width * height
        else:
            output_dim = width * height

        self.input_dim = input_dim
        self.num_objects = num_objects

        super().__init__(OrderedDict(
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

        #torch.nn.init.constant_(self.network.deconv2.bias, -1)


class DecoderCNNMedium(nn.Sequential):
    """CNN decoder, maps latent state to image."""
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu', flat_state=False):

        width, height = output_size[1] // 5, output_size[2] // 5

        if flat_state:
            output_dim = num_objects * width * height
        else:
            output_dim = width * height

        self.input_dim = input_dim
        self.num_objects = num_objects

        super().__init__(OrderedDict(
            fc1=nn.Linear(input_dim, hidden_dim),
            act1=utils.get_act_fn(act_fn),
            fc2=nn.Linear(hidden_dim, hidden_dim),
            ln=nn.LayerNorm(hidden_dim),
            act2=utils.get_act_fn(act_fn),
            fc3=nn.Linear(hidden_dim, output_dim),
            view=View(self.num_objects, width, height),
            deconv1=nn.ConvTranspose2d(num_objects, hidden_dim,
                                       kernel_size=5, stride=5),
            ln1=nn.BatchNorm2d(hidden_dim),
            act3=utils.get_act_fn(act_fn),
            deconv2=nn.ConvTranspose2d(hidden_dim, output_size[0],
                                       kernel_size=9, padding=4),
        ))


class DecoderCNNLarge(nn.Sequential):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu', flat_state=False):
        self.input_dim = input_dim
        self.num_objects = num_objects

        num_channels, width, height = output_size

        if flat_state:
            output_dim = num_objects * width * height
        else:
            output_dim = width * height

        super().__init__(OrderedDict(
            fc1=nn.Linear(input_dim, hidden_dim),
            act1=utils.get_act_fn(act_fn),
            fc2=nn.Linear(hidden_dim, hidden_dim),
            ln=nn.LayerNorm(hidden_dim),
            act2=utils.get_act_fn(act_fn),
            fc3=nn.Linear(hidden_dim, output_dim),
            view=View(self.num_objects, width, height),
            deconv1=nn.ConvTranspose2d(num_objects, hidden_dim,
                                       kernel_size=3, padding=1),
            ln1=nn.BatchNorm2d(hidden_dim),
            act3=utils.get_act_fn(act_fn),
            deconv2=nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                       kernel_size=3, padding=1),
            ln2=nn.BatchNorm2d(hidden_dim),
            act4=utils.get_act_fn(act_fn),
            deconv3=nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                       kernel_size=3, padding=1),
            ln3=nn.BatchNorm2d(hidden_dim),
            act5=utils.get_act_fn(act_fn),
            deconv4=nn.ConvTranspose2d(hidden_dim, num_channels,
                                       kernel_size=3, padding=1),
        ))
