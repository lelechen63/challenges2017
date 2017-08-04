import argparse
import os
from time import strftime
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, Dropout, Flatten, Input, concatenate, Reshape, Lambda, Permute
from keras.layers.recurrent import LSTM
from nibabel import load as load_nii
from utils import color_codes, nfold_cross_validation, get_biggest_region
from itertools import izip
from data_creation import load_patch_batch_train, get_cnn_centers
from data_creation import load_patch_batch_generator_test
from data_manipulation.generate_features import get_mask_voxels
from data_manipulation.metrics import dsc_seg
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.layers.pooling import AveragePooling3D
# from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

from subpixel import SubPixelUpscaling
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu, 3x3x3 Conv3D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''

    concat_axis = 1 

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_uniform', padding='same', data_format='channels_first',  use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer='he_uniform', padding='same', data_format='channels_first', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __transition_block(ip, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = 1 

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_uniform', padding='same',data_format='channels_first', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling3D((2, 2, 2), data_format='channels_first', strides=(2, 2, 2))(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 
    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        # x = concatenate(x_list, axis=concat_axis)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_up_block(ip, nb_filters, type='upsampling', weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    if type == 'upsampling':
        x = UpSampling3D()(ip)
    elif type == 'subpixel':
        x = Conv3D(nb_filters, (3, 3,  3), activation='relu', padding='same',data_format='channels_first', W_regularizer=l2(weight_decay),
                   use_bias=False, kernel_initializer='he_uniform')(ip)
        x = SubPixelUpscaling(scale_factor=2)(x)
        x = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same',data_format='channels_first', W_regularizer=l2(weight_decay),
                   use_bias=False, kernel_initializer='he_uniform')(x)
    else:
        x = Conv3DTranspose(nb_filters, (3, 3, 3), activation='relu', padding='same',data_format='channels_first', strides=(2, 2, 2),
                            kernel_initializer='he_uniform')(ip)

    return x

def create_densenet(nb_classes, img_input, include_top= False, depth=40, nb_dense_block=3, growth_rate=0, nb_filter=4,
                       nb_layers_per_block=2, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-4,
                       activation='softmax'):
    concat_axis = 1
    assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4'
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'
            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            count = int((depth - 4) / 3)
            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    if bottleneck:
        nb_layers = [int(layer // 2) for layer in nb_layers]
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Conv3D(8, (3, 3, 3), kernel_initializer='he_uniform', padding='same',data_format='channels_first',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], 16*(block_idx + 1), growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        print x.shape
        print '----'
        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)
        print x.shape
        print '----'
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, 2, 64, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)
    print x.shape
    print '----'
 
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    x = Activation('relu')(x)
    # x = GlobalAveragePooling3D()(x)

    # if include_top:
    #     x = Dense(nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    return x
def test():
	a = np.zeros((None,13,13,13,1))
	print a.shape
	net = create_densenet(2,a)
	print net.summary()
test()