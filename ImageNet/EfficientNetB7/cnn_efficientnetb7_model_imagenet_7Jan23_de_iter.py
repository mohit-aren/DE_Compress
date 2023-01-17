from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import string
import collections

import numpy as np
import cv2
from keras import applications
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend, layers, models
from keras import utils as keras_utils
from keras.models import model_from_json
import tensorflow as tf

from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras.engine import training
#from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

from six.moves import xrange
from keras_applications.imagenet_utils import _obtain_input_shape
#from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
#from keras.applications.efficientnet import preprocess_input as _preprocess_input
def preprocess_input(x, data_format, mode):
  """Preprocesses a Numpy array encoding a batch of images.
  Arguments:
      x: Input array, 3D or 4D.
      data_format: Data format of the image array.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
  Returns:
      Preprocessed Numpy array.
  """
  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x

  if mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if x.ndim == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  # Zero-center by mean pixel
  if data_format == 'channels_first':
    if x.ndim == 3:
      x[0, :, :] -= mean[0]
      x[1, :, :] -= mean[1]
      x[2, :, :] -= mean[2]
      if std is not None:
        x[0, :, :] /= std[0]
        x[1, :, :] /= std[1]
        x[2, :, :] /= std[2]
    else:
      x[:, 0, :, :] -= mean[0]
      x[:, 1, :, :] -= mean[1]
      x[:, 2, :, :] -= mean[2]
      if std is not None:
        x[:, 0, :, :] /= std[0]
        x[:, 1, :, :] /= std[1]
        x[:, 2, :, :] /= std[2]
  else:
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
      x[..., 0] /= std[0]
      x[..., 1] /= std[1]
      x[..., 2] /= std[2]
  return x

import keras_applications as ka
#from .__version__ import __version__


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', ka._KERAS_BACKEND)
    layers = kwargs.get('layers', ka._KERAS_LAYERS)
    models = kwargs.get('models', ka._KERAS_MODELS)
    utils = kwargs.get('utils', ka._KERAS_UTILS)
    return backend, layers, models, utils

#from . import get_submodules_from_kwargs
from weights import IMAGENET_WEIGHTS_PATH, IMAGENET_WEIGHTS_HASHES, NS_WEIGHTS_HASHES, NS_WEIGHTS_PATH

'''
backend = None
layers = None
models = None
keras_utils = None
'''

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def preprocess_input(x, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in ['backend', 'layers', 'models', 'utils']}
    return _preprocess_input(x, mode='torch', **kwargs)


def get_swish(**kwargs):
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """

        if backend.backend() == 'tensorflow':
            try:
                # The native TF implementation has a more
                # memory-efficient gradient implementation
                return backend.tf.nn.swish(x)
            except AttributeError:
                pass

        return x * backend.sigmoid(x)

    return swish


def get_dropout(**kwargs):
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.

    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', ):
    """Mobile Inverted Residual Bottleneck."""

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # workaround over non working dropout with None in noise_shape in tf.keras
    '''
    Dropout = get_dropout(
        backend=backend,
        layers=layers,
        models=models,
        utils=keras_utils
    )
    '''
    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
        if backend.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            pattern = ([True, True, True, False] if backend.image_data_format() == 'channels_last'
                       else [True, False, True, True])
            se_tensor = layers.Lambda(
                lambda x: backend.pattern_broadcast(x, pattern),
                name=prefix + 'se_broadcast')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            '''
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=prefix + 'drop')(x)
            '''
            x = layers.Dropout(0.3)(x)
        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    #global backend, layers, models, keras_utils
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', 'noisy-student', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_resolution,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.python.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    activation = get_swish(**kwargs)

    # Build stem
    x = img_input
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(round_repeats(block_args.num_repeat,
                                         depth_coefficient) for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    # Build top
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':

        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(
            file_name,
            IMAGENET_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash,
        )
        model.load_weights(weights_path)

    elif weights == 'noisy-student':

        if include_top:
            file_name = "{}_{}.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = "{}_{}_notop.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(
            file_name,
            NS_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash,
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model

def EfficientNet_compress(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    #global backend, layers, models, keras_utils
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', 'noisy-student', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_resolution,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.python.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    activation = 'relu'

    # Build stem
    x = img_input
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(round_repeats(block_args.num_repeat,
                                         depth_coefficient) for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    # Build top
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':

        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(
            file_name,
            IMAGENET_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash,
        )
        model.load_weights(weights_path)

    elif weights == 'noisy-student':

        if include_top:
            file_name = "{}_{}.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = "{}_{}_notop.h5".format(model_name, weights)
            file_hash = NS_WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(
            file_name,
            NS_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash,
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.0, 224, 0.2,
        model_name='efficientnet-b0',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB1(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.1, 240, 0.2,
        model_name='efficientnet-b1',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.1, 1.2, 260, 0.3,
        model_name='efficientnet-b2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.2, 1.4, 300, 0.3,
        model_name='efficientnet-b3',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB4(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.4, 1.8, 380, 0.4,
        model_name='efficientnet-b4',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB5(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.6, 2.2, 456, 0.4,
        model_name='efficientnet-b5',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB6(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.8, 2.6, 528, 0.5,
        model_name='efficientnet-b6',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB7(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        2.0, 3.1, 600, 0.5,
        model_name='efficientnet-b7',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )

def EfficientNetB7_c(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet_compress(
        2.0, 3.1, 600, 0.5,
        model_name='efficientnet-b7',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetL2(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        4.3, 5.3, 800, 0.5,
        model_name='efficientnet-l2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetL2, '__doc__', EfficientNet.__doc__)

img_width, img_height = 48, 48        # Resolution of inputs
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3
#model=applications.VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

model = EfficientNetB7_c(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
model.summary()
#initialise top model
"""
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = get_file('vgg16_weights.h5', WEIGHTS_PATH_NO_TOP)

vgg_model.load_weights(weights_path)

# add the model on top of the convolutional base

model_final = Model(input= vgg_model.input, output= top_model(vgg_model.output))
"""
# Freeze first 15 layers
for layer in model.layers[:45]:
	layer.trainable = False
for layer in model.layers[45:]:
   layer.trainable = True


x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(output_dim = 1000, activation=tf.nn.softmax)(x) # 4-way softmax classifier at the end

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

#model_final.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical' )
label_map = (training_set.class_indices)

print(label_map)

itr = test_set = test_datagen.flow_from_directory(
        'val',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical')
nb_train_samples = 1300000                # Number of train samples
nb_validation_samples = 50000            # Number of validation samples

X, y = itr.next()


#block7b_expand_conv
#model_final.fit(x_train,onehot_encoded,batch_size=64,epochs=10)
#x_test = x_test[0:100]
#y_test = y_test[0:100]
model_final.load_weights('keras_efficientnet_main.h5')


#arr = model_final.evaluate(X,y)

#print(arr)

import random 
from keras.models import model_from_json
layer0_b = 2304
layer1_b = 3840

layer2_b = 1024
layer3_b = 1024

layer0_a = 2304
layer1_a = 3840

layer2_a = 1024
layer3_a = 1024

lyr = 0


#model1.summary()

model1 = model_final

for withtrain in range(0,10):
    lyr = 0
    if(withtrain > 0):
        model_final = model1
        layer0_b = layer0_a
        layer1_b = layer1_a
        layer2_b = layer2_a
        layer3_b = layer3_a
        #model_final.load_weights('EfficientNet_pruned_weights' + str(withtrain) + '.h5')
        
    while lyr < len(model_final.layers)-2:
    #for layer in model_final.layers[0:]:
        layer = model_final.layers[lyr]
        if(layer.name.find('block6b_expand_conv') >= 0):
            break
        
        #model1.add(layer)
        lyr += 1
    
    
    filters = model_final.layers[lyr].get_weights()
    filters = np.array(filters)
    print(filters.shape)
    #biases1 = np.copy(biases)
    shp = filters.shape
    A= []
    Acc = []
    
    def ensure_bounds(par):
        new_par = []
        for index in range(0, len(par)):
            if(par[index] >= 0.5):
                new_par.append(1)
            else:
                new_par.append(0)
                
        return new_par
    
    popsize = 20
    mutate = 0.5
    recombination = 0.7
    population = []
    for i in range(0,popsize):
        indv = []
        for k in range(0, layer0_b):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,2+1):
        print ('GENERATION:',i)
    
        gen_scores = [] # score keeping
    
        # cycle through each individual in the population
        for j in range(0, popsize):
    
            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0,popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)
    
            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]     # target individual
    
            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    
            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor)
    
            #--- RECOMBINATION (step #3.B) ----------------+
    
            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])
    
                else:
                    v_trial.append(x_t[k])
                    
            #--- GREEDY SELECTION (step #3.C) -------------+
    
            filters1 = np.copy(filters)
            #biases1 = np.copy(biases)
            
            for i in range(0,layer0_b):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[lyr].set_weights([filters1.reshape(shp[1], shp[2], shp[3], shp[4])])
            arr = model_final.evaluate(X,y)
            score_trial = arr[1]
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer0_b):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:, :,:,j] = 0
                if(x_t[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[lyr].set_weights([filters1.reshape(shp[1], shp[2], shp[3], shp[4])])
            arr = model_final.evaluate(X,y)
            score_target = arr[1]
    
            #score_target = cost_func(x_t)
    
            if score_trial > score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                #print '   >',score_trial, v_trial
    
            else:
                #print '   >',score_target, x_t
                gen_scores.append(score_target)
    
        #--- SCORE KEEPING --------------------------------+
    
        gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
        gen_best = max(gen_scores) 
        print(gen_best)                                 # fitness of best individual
        par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual
    
    A0 = np.copy(par1)       
    new_num = np.sum(par1)
    layer0_a = new_num
    
    layer0_a = layer0_b//2
           
    print(layer0_a)
    
    while lyr < len(model_final.layers)-2:
    #for layer in model_final.layers[0:]:
        layer = model_final.layers[lyr]
        if(layer.name.find('block7b_expand_conv') >= 0):
            break
        
        #model1.add(layer)
        lyr += 1
    
    
    filters = model_final.layers[lyr].get_weights()
    filters = np.array(filters)
    print(filters.shape)
    #biases1 = np.copy(biases)
    shp = filters.shape
    A= []
    Acc = []
    
    def ensure_bounds(par):
        new_par = []
        for index in range(0, len(par)):
            if(par[index] >= 0.5):
                new_par.append(1)
            else:
                new_par.append(0)
                
        return new_par
    
    popsize = 20
    mutate = 0.5
    recombination = 0.7
    population = []
    for i in range(0,popsize):
        indv = []
        for k in range(0, layer1_b):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,2+1):
        print ('GENERATION:',i)
    
        gen_scores = [] # score keeping
    
        # cycle through each individual in the population
        for j in range(0, popsize):
    
            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0,popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)
    
            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]     # target individual
    
            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    
            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor)
    
            #--- RECOMBINATION (step #3.B) ----------------+
    
            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])
    
                else:
                    v_trial.append(x_t[k])
                    
            #--- GREEDY SELECTION (step #3.C) -------------+
    
            filters1 = np.copy(filters)
            #biases1 = np.copy(biases)
            
            for i in range(0,layer1_b):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[lyr].set_weights([filters1.reshape(shp[1], shp[2], shp[3], shp[4])])
            arr = model_final.evaluate(X,y)
            score_trial = arr[1]
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer1_b):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:, :,:,j] = 0
                if(x_t[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[lyr].set_weights([filters1.reshape(shp[1], shp[2], shp[3], shp[4])])
            arr = model_final.evaluate(X,y)
            score_target = arr[1]
    
            #score_target = cost_func(x_t)
    
            if score_trial > score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                #print '   >',score_trial, v_trial
    
            else:
                #print '   >',score_target, x_t
                gen_scores.append(score_target)
    
        #--- SCORE KEEPING --------------------------------+
    
        gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
        gen_best = max(gen_scores) 
        print(gen_best)                                 # fitness of best individual
        par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual
    
    A1 = np.copy(par1)       
    new_num = np.sum(par1)
    layer1_a = new_num
    
    layer1_a = layer1_b //2
           
    print(layer1_a)
    
    while lyr < len(model_final.layers)-2:
        layer = model_final.layers[lyr]
        if(layer.name.find('dense_') >= 0):
            break
        lyr += 1
    
    ####################### 1st dense layer with 1024 filters
    print('1st dense layer with 1024 filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(X,y)
    print(arr)
    
    filters, biases = model_final.layers[lyr].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    def ensure_bounds(par):
        new_par = []
        for index in range(0, len(par)):
            if(par[index] >= 0.5):
                new_par.append(1)
            else:
                new_par.append(0)
                
        return new_par
    
    popsize = 20
    mutate = 0.5
    recombination = 0.7
    population = []
    for i in range(0,popsize):
        indv = []
        for k in range(0, layer2_b):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,2+1):
        print ('GENERATION:',i)
    
        gen_scores = [] # score keeping
    
        # cycle through each individual in the population
        for j in range(0, popsize):
    
            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0,popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)
    
            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]     # target individual
    
            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    
            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor)
    
            #--- RECOMBINATION (step #3.B) ----------------+
    
            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])
    
                else:
                    v_trial.append(x_t[k])
                    
            #--- GREEDY SELECTION (step #3.C) -------------+
    
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer2_b):
                f = filters[ :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:, :,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[lyr].set_weights([filters1, biases1])
            arr = model_final.evaluate(X,y)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer2_b):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:, :,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[lyr].set_weights([filters1, biases1])
            arr = model_final.evaluate(X,y)
            score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)
    
    
            #score_target = cost_func(x_t)
    
            if score_trial > score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                #print '   >',score_trial, v_trial
    
            else:
                #print '   >',score_target, x_t
                gen_scores.append(score_target)
    
        #--- SCORE KEEPING --------------------------------+
    
        gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
        gen_best = max(gen_scores) 
        print(gen_best)                                 # fitness of best individual
        par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual
    
    
    A2 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    layer2_a = new_num
    
    lyr += 1
    while lyr < len(model_final.layers)-2:
        layer = model_final.layers[lyr]
        if(layer.name.find('dense_') >= 0):
            break
        lyr += 1
    
    ####################### 1st dense layer with 1024 filters
    print('2nd dense layer with 1024 filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(X,y)
    print(arr)
    
    filters, biases = model_final.layers[lyr].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    def ensure_bounds(par):
        new_par = []
        for index in range(0, len(par)):
            if(par[index] >= 0.5):
                new_par.append(1)
            else:
                new_par.append(0)
                
        return new_par
    
    popsize = 20
    mutate = 0.5
    recombination = 0.7
    population = []
    for i in range(0,popsize):
        indv = []
        for k in range(0, layer3_b):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,2+1):
        print ('GENERATION:',i)
    
        gen_scores = [] # score keeping
    
        # cycle through each individual in the population
        for j in range(0, popsize):
    
            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0,popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)
    
            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]     # target individual
    
            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    
            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor)
    
            #--- RECOMBINATION (step #3.B) ----------------+
    
            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])
    
                else:
                    v_trial.append(x_t[k])
                    
            #--- GREEDY SELECTION (step #3.C) -------------+
    
            filters1 = np.copy(filters)
            #biases1 = np.copy(biases)
            
            for i in range(0,layer3_b):
                f = filters[ :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:, :,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[ :, i] = 0
            
            model_final.layers[lyr].set_weights([filters1, biases1])
            arr = model_final.evaluate(X,y)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer3_b):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:, :,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[lyr].set_weights([filters1, biases1])
            arr = model_final.evaluate(X,y)
            score_target = 0.5*arr[1]+0.5*len(x_t)/np.sum(x_t)
    
    
            #score_target = cost_func(x_t)
    
            if score_trial > score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                #print '   >',score_trial, v_trial
    
            else:
                #print '   >',score_target, x_t
                gen_scores.append(score_target)
    
        #--- SCORE KEEPING --------------------------------+
    
        gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
        gen_best = max(gen_scores) 
        print(gen_best)                                 # fitness of best individual
        par1 = population[gen_scores.index(max(gen_scores))]     # solution of best individual
    
    
    A3 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    layer3_a = new_num
    
    
    model = model_final
    
    #model1 = model_final
    model2 = EfficientNetB7_c(weights=None, include_top=False, input_shape=(img_width, img_height, 3))
    #model.summary()
    # Freeze first 15 layers
    for layer in model2.layers[:45]:
    	layer.trainable = False
    for layer in model2.layers[45:]:
       layer.trainable = True
    
    
    x = model2.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_dim = 1000, activation=tf.nn.softmax)(x) # 4-way softmax classifier at the end
    
    model1 = Model(input=model2.input, output=predictions)
    
    lyr = 0
    
    while lyr < len(model1.layers)-2:
    #for layer in model.layers[0:]:
        layer = model1.layers[lyr]
        if(layer.name.find('block6a_project_conv') >= 0):
            break
        
        #model1.add(layer)
        lyr += 1
        
    dns = 0    
    while lyr < len(model1.layers)-2:
        #layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        layer = model1.layers[lyr]
        print(layer.name)
        layer = model1.layers[lyr]
        print(layer.name, lyr)
        if(layer.name.find('block7a_project_conv') >= 0):
            break
        
        if(layer.name.find('conv') >= 0):
            filters1 = layer.get_weights()
            print(model1.layers[lyr].filters)
            if(model1.layers[lyr].filters == 2304):
                model1.layers[lyr].filters = layer0_a
            if(model1.layers[lyr].filters == layer1_b//6):
                #model1.layers[lyr].filters = layer1_a//6
                pass
            if(model1.layers[lyr].filters == (layer1_b//6)*4):
                pass
                #model1.layers[lyr].filters = (layer1_a//6)*4
        elif(layer.name.find('dense_') >= 0 and dns == 0):
            #print('Dense', layer2_a)
            filters, biases = layer.get_weights()
            filters1 = np.array(filters)
            print(filters1.shape)
            dns = 1
            model1.layers[lyr].units = layer2_a
        elif(layer.name.find('dense_') >= 0 and dns == 1):
            #filters1, biases = layer.get_weights()
            model1.layers[lyr].units = layer3_a
            dns = 2
        
        lyr += 1
    
    while lyr < len(model1.layers)-2:
    #for layer in model.layers[0:]:
        layer = model1.layers[lyr]
        if(layer.name.find('block7a_project_conv') >= 0):
            break
        
        #model1.add(layer)
        lyr += 1
        
    dns = 0    
    while lyr < len(model1.layers)-2:
        #layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        layer = model1.layers[lyr]
        print(layer.name)
        layer = model1.layers[lyr]
        print(layer.name, lyr)
        if(layer.name == 'block7d_add'):
            #print('Err')
            lyr += 4
            continue
        if(layer.name.find('conv') >= 0):
            filters1 = layer.get_weights()
            print(model1.layers[lyr].filters)
            if(model1.layers[lyr].filters == 3840):
                model1.layers[lyr].filters = layer1_a
            if(model1.layers[lyr].filters == layer1_b//6):
                #model1.layers[lyr].filters = layer1_a//6
                pass
            if(model1.layers[lyr].filters == (layer1_b//6)*4):
                pass
                #model1.layers[lyr].filters = (layer1_a//6)*4
        elif(layer.name.find('dense_') >= 0 and dns == 0):
            #print('Dense', layer2_a)
            filters, biases = layer.get_weights()
            filters1 = np.array(filters)
            print(filters1.shape)
            dns = 1
            model1.layers[lyr].units = layer2_a
        elif(layer.name.find('dense_') >= 0 and dns == 1):
            #filters1, biases = layer.get_weights()
            model1.layers[lyr].units = layer3_a
            dns = 2
        
        lyr += 1
    
    #model1.summary()
    json = model1.to_json().replace(str(3840), str(layer1_a))
    json = json.replace(str(2304),str(layer0_a))
    model1 = model_from_json(json)
    model1.summary()
    model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])
    
    model1.save_weights('EfficientNet_pruned_weights' + str(withtrain+1) + '.h5')
    model1.save('Eff_Pruned_FLOPs.h5')
    lyr = 0
    while lyr < len(model1.layers)-2:
    #for layer in model.layers[0:]:
        layer = model1.layers[lyr]
        if(layer.name.find('block6a_project_conv') >= 0):
            break
        
        layerr = model_final.layers[lyr].get_weights()
        model1.layers[lyr].set_weights(layerr)
    
        #model1.add(layer)
        lyr += 1
    
    filters = model_final.layers[lyr].get_weights()
    filters1 = model1.layers[lyr].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    print('Shape', np.array(filters).shape)
    print('Shape', np.array(filters1).shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    
    index1 = 0
    # plot each channel separately
    for j in range(layer0_b):
        if(A0[j] == 1) :
            filters1[:, :, index1, :] = filters[:, :, :, j, :]
            #print(index1,i)
            #biases1[index1] = biases[index1]
            index1 += 1
    #print(biases1, biases)        
    model1.layers[lyr].set_weights([filters1])
    """
    while lyr < len(model1.layers)-2:
    #for layer in model_final.layers[0:]:
        layer = model1.layers[lyr]
        if(layer.name.find('block7a_project_conv') >= 0):
            break
        
        #layerr = model_final.layers[lyr].get_weights()
        #model1.layers[lyr].set_weights(layerr)
    
        #model1.add(layer)
        lyr += 1
    
    filters = model_final.layers[lyr].get_weights()
    filters1 = model1.layers[lyr].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    print('Shape', np.array(filters).shape)
    print('Shape', np.array(filters1).shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    
    index1 = 0
    # plot each channel separately
    for j in range(layer1_b):
        if(A1[j] == 1) :
            filters1[:, :, index1, :] = filters[:, :, :, j, :]
            #print(index1,i)
            #biases1[index1] = biases[index1]
            index1 += 1
    #print(biases1, biases)        
    model1.layers[lyr].set_weights([filters1])
    """
    
    while lyr < len(model1.layers)-2:
    #for layer in model_final.layers[0:]:
        layer = model1.layers[lyr]
        if(layer.name.find('dense_') >= 0):
            break
        #model1.add(layer)
        lyr += 1
    
    
    
    
    ######################## 1st convolution layer with 128 filters
    filters, biases = model_final.layers[lyr].get_weights()
    filters1, biases1 = model1.layers[lyr].get_weights()
    print('Shape', np.array(filters).shape)
    print('Shape', np.array(filters1).shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer2_b):
        if(A2[j] == 1) :
            filters1[:, index1] = filters[:, j]
            #print(index1,i)
            biases1[index1] = biases[index1]
            index1 += 1
    #print(biases1, biases)        
    model1.layers[lyr].set_weights([filters1, biases1])
    lyr += 1
    while lyr < len(model1.layers)-2:
    #for layer in model_final.layers[0:]:
        layer = model1.layers[lyr]
        if(layer.name.find('dense_') >= 0):
            break
        
        #model1.add(layer)
        lyr += 1
    
    ######################## 2nd dense layer with 1024 filters
    filters, biases = model_final.layers[lyr].get_weights()
    filters1, biases1 = model1.layers[lyr].get_weights()
    print('Shape', np.array(filters).shape)
    print('Shape', np.array(filters1).shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer3_b):
        if(A3[j] == 1) :
            index2 = 0
            for l in range(layer2_b):
                if(A2[l] == 1):
                    filters1[index2, index1] = filters[l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[lyr].set_weights([filters1, biases1])
    
    
    arr = model1.evaluate(X,y)
    print(arr)
    
    #y_train = y_train.reshape(len(y_train), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_train)
    model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])
    
    
    model1.fit_generator(
        training_set,
        samples_per_epoch=nb_train_samples, 
        epochs=1,
        validation_data=test_set,
        validation_steps=nb_validation_samples/32)
    
    model1.summary()
    model1.save_weights('EfficientNet_pruned_weights' + str(withtrain+1) + '.h5')
    model1.save('Eff_Pruned_FLOPs' + str(withtrain+1) +'.h5')
    #y_test = y_test.reshape(len(y_test), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_test)
    arr = model1.evaluate(X,y)
    
    
    print(arr)