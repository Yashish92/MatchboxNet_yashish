"""
The _pw_group and _shuffle functions were taken from https://github.com/minhto2802/keras-shufflenet and adapted for
1D convolutions

The mentioned repository tries to implement the https://arxiv.org/pdf/1707.01083.pdf (shufflenet) paper which introduced the
group convolution and shuffling across channels when computing point-wise convolutions. The original paper and the repository
is implemented for 2D convolutions

"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Lambda, Conv1D, Flatten, BatchNormalization, MaxPool1D, ReLU, MaxPool2D, Conv2D, DepthwiseConv1D
import keras.backend as K

def _pw_group(tensor, nb_groups, in_channels, out_channels):
    """Pointwise grouped convolution."""
    nb_chan_per_grp = in_channels // nb_groups

    pw_convs = []
    for grp in range(nb_groups):
        x = Lambda(lambda x: x[:, :, nb_chan_per_grp * grp: nb_chan_per_grp * (grp + 1)])(tensor)
        grp_out_chan = int(out_channels / nb_groups + 0.5)

        pw_convs.append(
            Conv1D(grp_out_chan,
                   kernel_size=1,
                   padding='same',
                   use_bias=False,
                   strides=1)(x)
        )

    return Concatenate(axis=-1)(pw_convs)   # check axis


def _shuffle(x, nb_groups):
    """Perform shuffling between channels in groups"""
    def shuffle_layer(x):
        _, h, n = K.int_shape(x)
        nb_chan_per_grp = n // nb_groups

        x = K.reshape(x, (-1, h, nb_chan_per_grp, nb_groups))
        x = K.permute_dimensions(x, (0, 1, 2, 3)) # Transpose only grps and chs
        x = K.reshape(x, (-1, h, n))

        return x

    return Lambda(shuffle_layer)(x)


def pointwise_conv(input, channel_in, channel_out=64):
    """
    Compute pointwise convolutions
    :param input: input tensor
    :param channel_in: No of channels in input
    :param channel_out: No of channels in the output
    :return: final output of pointwise convolution layer
    """
    group_conv = _pw_group(input, nb_groups=2, in_channels=channel_in, out_channels=channel_out)
    shuff = _shuffle(group_conv, nb_groups=2)

    return shuff
