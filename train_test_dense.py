# from __future__ import print_function
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Brats17CBICA/')
    parser.add_argument('-F', '--n-fold', dest='folds', type=int, default=5)
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=13)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=5)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=2048)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=500)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=6)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=10)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential', default=False)
    parser.add_argument('--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('--padding', action='store', dest='padding', default='valid')
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--no-t1', action='store_false', dest='use_t1', default=True)
    parser.add_argument('--no-t1ce', action='store_false', dest='use_t1ce', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    parser.add_argument('-m', '--multi-channel', action='store_true', dest='multi', default=False)
    return vars(parser.parse_args())


def list_directories(path):
    return filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])


def get_names_from_path(options):
    path = options['dir_name']

    patients = sorted(list_directories(path))

    # Prepare the names
    flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair'])
                   for p in patients] if options['use_flair'] else None
    t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2'])
                for p in patients] if options['use_t2'] else None
    t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1'])
                for p in patients] if options['use_t1'] else None
    t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce'])
                  for p in patients] if options['use_t1ce'] else None
    label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


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
        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, 2, 64, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)
 
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    x = Activation('relu')(x)
    # x = GlobalAveragePooling3D()(x)

    # if include_top:
    #     x = Dense(nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    return x


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    sequential = options['sequential']
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    num_classes = 5
    epochs = options['epochs']
    padding = options['padding']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks
    balanced = options['balanced']
    # Data loading parameters
    preload = options['preload']
    queue = options['queue']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_name']
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    s_s = '.s' if sequential else '.f'
    ub_s = '.ub' if not balanced else ''
    params_s = (ub_s, dfactor, s_s, patch_width, conv_s, filters_s, dense_size, epochs, padding)
    sufix = '%s.D%d%s.p%d.c%s.n%s.d%d.e%d.pad_%s.' % params_s
    n_channels = np.count_nonzero([
        options['use_flair'],
        options['use_t2'],
        options['use_t1'],
        options['use_t1ce']]
    )

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting cross-validation' + c['nc'])
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    data_names, label_names = get_names_from_path(options)
    folds = options['folds']
    fold_generator = izip(nfold_cross_validation(data_names, label_names, n=folds, val_data=0.25), xrange(folds))
    dsc_results = list()
    for (train_data, train_labels, val_data, val_labels, test_data, test_labels), i in fold_generator:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Fold %d/%d: ' % (i+1, folds) + c['g'] +
              'Number of training/validation/testing images (%d=%d/%d=%d/%d)'
              % (len(train_data), len(train_labels), len(val_data), len(val_labels), len(test_data)) + c['nc'])
        # Prepare the data relevant to the leave-one-out (subtract the patient from the dataset and set the path)
        # Also, prepare the network
        net_name = os.path.join(path, 'baseline-brats2017.fold%d' % i + sufix + 'mdl')

        # First we check that we did not train for that patient, in order to save time
        try:
            # net_name_before =  os.path.join(path,'baseline-brats2017.fold0.D500.f.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e1.pad_valid.mdl')
            net = keras.models.load_model(net_name)
        except IOError:
            print '==============================================================='
            # NET definition using Keras
            train_centers = get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced)
            val_centers = get_cnn_centers(val_data[:, 0], val_labels, balanced=balanced)
            train_samples = len(train_centers)/dfactor
            val_samples = len(val_centers) / dfactor
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Creating and compiling the model ' +
                  c['b'] + '(%d samples)' % train_samples + c['nc'])
            train_steps_per_epoch = -(-train_samples/batch_size)
            val_steps_per_epoch = -(-val_samples / batch_size)
            input_shape = (n_channels,) + patch_size
            
            # This architecture is based on the functional Keras API to introduce 3 output paths:
            # - Whole tumor segmentation
            # - Core segmentation (including whole tumor)
            # - Whole segmentation (tumor, core and enhancing parts)
            # The idea is to let the network work on the three parts to improve the multiclass segmentation.
            merged_inputs = Input(shape=(4,) + patch_size, name='merged_inputs')
            flair = Reshape((1,) + patch_size)(
              Lambda(
                  lambda l: l[:, 0, :, :, :],
                  output_shape=(1,) + patch_size)(merged_inputs),
            )
            t2 = Reshape((1,) + patch_size)(
              Lambda(lambda l: l[:, 1, :, :, :], output_shape=(1,) + patch_size)(merged_inputs)
            )
            t1 = Lambda(lambda l: l[:, 2:, :, :, :], output_shape=(2,) + patch_size)(merged_inputs)
            flair = create_densenet(2,flair)
            # t2 = create_densenet(3, t2)
            # t1 = create_densenet(5,t1)
                      
            # flair = Conv3D(8,(3,3,3),activation= 'relu',data_format = 'channels_first')(flair)
            # # flair = Dropout(0.5)(flair)
            # flair = Conv3D(16,(3,3,3),activation= 'relu',data_format = 'channels_first')(flair)
            # # flair = Dropout(0.5)(flair)
            # flair = Conv3D(16,(3,3,3),activation= 'relu',data_format = 'channels_first')(flair)
            # # flair = Dropout(0.5)(flair)
            # flair = Conv3D(32,(3,3,3),activation= 'relu',data_format = 'channels_first')(flair)
            # # flair = Dropout(0.5)(flair)
            # flair = Conv3D(32,(3,3,3),activation= 'relu',data_format = 'channels_first')(flair)
            # flair = Dropout(0.5)(flair)
            # t2 = Conv3D(8,(3,3,3),activation= 'relu',data_format = 'channels_first')(t2)
            # # t2 = Dropout(0.5)(t2)
            # t2 = Conv3D(16,(3,3,3),activation= 'relu',data_format = 'channels_first')(t2)
            # # t2 = Dropout(0.5)(t2)
            # t2 = Conv3D(16,(3,3,3),activation= 'relu',data_format = 'channels_first')(t2)
            # # t2 = Dropout(0.5)(t2)
            # t2 = Conv3D(32,(3,3,3),activation= 'relu',data_format = 'channels_first')(t2)
            # # t2 = Dropout(0.5)(t2)
            # t2 = Conv3D(32,(3,3,3),activation= 'relu',data_format = 'channels_first')(t2)
            # # t2 = Dropout(0.5)(t2)
            # t1 = Conv3D(8,(3,3,3),activation= 'relu',data_format = 'channels_first')(t1)
            # # t1 = Dropout(0.5)(t1)
            # t1 = Conv3D(16,(3,3,3),activation= 'relu',data_format = 'channels_first')(t1)
            # # t1 = Dropout(0.5)(t1)
            # t1 = Conv3D(16,(3,3,3),activation= 'relu',data_format = 'channels_first')(t1)
            # # t1 = Dropout(0.5)(t1)
            # t1 = Conv3D(32,(3,3,3),activation= 'relu',data_format = 'channels_first')(t1) 
            # # t1 = Dropout(0.5)(t1)
            # t1 = Conv3D(32,(3,3,3),activation= 'relu',data_format = 'channels_first')(t1)
            # t1 = Dropout(0.5)(t1)
            # for filters, kernel_size in zip(filters_list, kernel_size_list):
            #   flair = Conv3D(filters,
            #                  kernel_size=kernel_size,
            #                  activation='relu',
            #                  data_format='channels_first'
            #                  )(flair)
            #   t2 = Conv3D(filters,
            #               kernel_size=kernel_size,
            #               activation='relu',
            #               data_format='channels_first'
            #               )(t2)
            #   t1 = Conv3D(filters,
            #               kernel_size=kernel_size,
            #               activation='relu',
            #               data_format='channels_first'
            #               )(t1)
            #   flair = Dropout(0.5)(flair)
            #   t2 = Dropout(0.5)(t2)
            #   t1 = Dropout(0.5)(t1)
            flair = Flatten()(flair)
            # t2 = Flatten()(t2)
            # t1 = Flatten()(t1)
            flair = Dense(dense_size, activation='relu')(flair)
            flair = Dropout(0.5)(flair)
            # t2 = concatenate([flair, t2])
            # t2 = Dense(dense_size, activation='relu')(t2)
            # t2 = Dropout(0.5)(t2)
            # t1 = concatenate([t2, t1])
            # t1 = Dense(dense_size, activation='relu')(t1)
            # t1 = Dropout(0.5)(t1)

            tumor = Dense(2, activation='softmax', name='tumor')(flair)
            # core = Dense(3, activation='softmax', name='core')(t2)
            # enhancing = Dense(num_classes, activation='softmax', name='enhancing')(t1)
            # net = Model(inputs=merged_inputs, outputs=[tumor, core, enhancing])
            net = Model(inputs=merged_inputs, outputs=[tumor])
            



            # net_name_before =  os.path.join(path,'baseline-brats2017.fold0.D500.f.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e1.pad_valid.mdl')
            # net = keras.models.load_model(net_name_before)
            net.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Training the model with a generator for ' +
                  c['b'] + '(%d parameters)' % net.count_params() + c['nc'])
            print(net.summary())
       
            net.fit_generator(
                generator=load_patch_batch_train(
                    image_names=train_data,
                    label_names=train_labels,
                    centers=train_centers,
                    batch_size=batch_size,
                    size=patch_size,
                    # fc_shape = patch_size,
                    nlabels=num_classes,
                    dfactor=dfactor,
                    preload=preload,
                    split=not sequential,
                    datatype=np.float32
                ),
                validation_data=load_patch_batch_train(
                    image_names=val_data,
                    label_names=val_labels,
                    centers=val_centers,
                    batch_size=batch_size,
                    size=patch_size,
                    # fc_shape = patch_size,
                    nlabels=num_classes,
                    dfactor=dfactor,
                    preload=preload,
                    split=not sequential,
                    datatype=np.float32
                ),
                steps_per_epoch=train_steps_per_epoch,
                validation_steps=val_steps_per_epoch,
                max_q_size=queue,
                epochs=epochs
            )
            net.save(net_name)

        # Then we test the net.
        for p, gt_name in zip(test_data, test_labels):
            p_name = p[0].rsplit('/')[-2]
            patient_path = '/'.join(p[0].rsplit('/')[:-1])
            outputname = os.path.join(patient_path, 'deep-brats17' + sufix + 'test.nii.gz')
            gt_nii = load_nii(gt_name)
            gt = np.copy(gt_nii.get_data()).astype(dtype=np.uint8)
            try:
                load_nii(outputname)
            except IOError:
                roi_nii = load_nii(p[0])
                roi = roi_nii.get_data().astype(dtype=np.bool)
                centers = get_mask_voxels(roi)
                test_samples = np.count_nonzero(roi)
                image = np.zeros_like(roi).astype(dtype=np.uint8)
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      '<Creating the probability map ' + c['b'] + p_name + c['nc'] + c['g'] +
                      ' (%d samples)>' % test_samples + c['nc'])
                test_steps_per_epoch = -(-test_samples / batch_size)
                y_pr_pred = net.predict_generator(
                    generator=load_patch_batch_generator_test(
                        image_names=p,
                        centers=centers,
                        batch_size=batch_size,
                        size=patch_size,
                        preload=preload,
                    ),
                    steps=test_steps_per_epoch,
                    max_q_size=queue
                )
                [x, y, z] = np.stack(centers, axis=1)

                if not sequential:
                    tumor = np.argmax(y_pr_pred[0], axis=1)
                    y_pr_pred = y_pr_pred[-1]
                    roi = np.zeros_like(roi).astype(dtype=np.uint8)
                    roi[x, y, z] = tumor
                    roi_nii.get_data()[:] = roi
                    roiname = os.path.join(patient_path, 'deep-brats17' + sufix + 'test.roi.nii.gz')
                    roi_nii.to_filename(roiname)

                y_pred = np.argmax(y_pr_pred, axis=1)

                image[x, y, z] = y_pred
                # Post-processing (Basically keep the biggest connected region)
                image = get_biggest_region(image)
                labels = np.unique(gt.flatten())
                results = (p_name,) + tuple([dsc_seg(gt == l, image == l) for l in labels[1:]])
                text = 'Subject %s DSC: ' + '/'.join(['%f' for _ in labels[1:]])
                print(text % results)
                dsc_results.append(results)

                print(c['g'] + '                   -- Saving image ' + c['b'] + outputname + c['nc'])
                roi_nii.get_data()[:] = image
                roi_nii.to_filename(outputname)


if __name__ == '__main__':
    main()








