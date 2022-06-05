# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018

@author: mohit123
"""
import numpy as np
import cv2
from keras import applications
from keras.models import Sequential
from keras.layers import  Convolution2D, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file
from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Activation
import random

img_width, img_height = 48, 48        # Resolution of inputs
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape

import tensorflow as tf

from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras.engine import training
#from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
#from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/mobilenet/"
)
#layers = None


def MobileNet(
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=1e-3,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    """Instantiates the MobileNet architecture.
    Reference:
    - [MobileNets: Efficient Convolutional Neural Networks
       for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)
    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.
    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).
    Note: each Keras Application expects a specific kind of input preprocessing.
    For MobileNet, call `tf.keras.applications.mobilenet.preprocess_input`
    on your inputs before passing them to the model.
    `mobilenet.preprocess_input` will scale input pixels between -1 and 1.
    Args:
      input_shape: Optional shape tuple, only to be specified if `include_top`
        is False (otherwise the input shape has to be `(224, 224, 3)` (with
        `channels_last` data format) or (3, 224, 224) (with `channels_first`
        data format). It should have exactly 3 inputs channels, and width and
        height should be no smaller than 32. E.g. `(200, 200, 3)` would be one
        valid value. Default to `None`.
        `input_shape` will be ignored if the `input_tensor` is provided.
      alpha: Controls the width of the network. This is known as the width
        multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
        decreases the number of filters in each layer. - If `alpha` > 1.0,
        proportionally increases the number of filters in each layer. - If
        `alpha` = 1, default number of filters from the paper are used at each
        layer. Default to 1.0.
      depth_multiplier: Depth multiplier for depthwise convolution. This is
        called the resolution multiplier in the MobileNet paper. Default to 1.0.
      dropout: Dropout rate. Default to 0.001.
      include_top: Boolean, whether to include the fully-connected layer at the
        top of the network. Default to `True`.
      weights: One of `None` (random initialization), 'imagenet' (pre-training
        on ImageNet), or the path to the weights file to be loaded. Default to
        `imagenet`.
      input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model. `input_tensor` is useful for sharing
        inputs between multiple different networks. Default to None.
      pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` (default) means that the output of the model will be
            the 4D tensor output of the last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: Optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified. Defaults to 1000.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.
    """
    
    '''
    global layers
    
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()

    if kwargs:
        raise ValueError("Unknown argument(s): {(kwargs,)}")
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            "Received weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            "Received classes={classes}"
        )
    '''
    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape =_obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if depth_multiplier != 1:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "depth multiplier must be 1.  "
                "Received depth_multiplier={depth_multiplier}"
            )

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one o"
                "`0.25`, `0.50`, `0.75` or `1.0` only.  "
                "Received alpha={alpha}"
            )

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            logging.warning(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [128, 160, 192, 224]. "
                "Weights for input shape (224, 224) will be "
                "loaded as the default."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2
    )
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4
    )
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6
    )
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12
    )
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        x = layers.Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
        x = layers.Reshape((classes,), name="reshape_2")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(
            activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name="mobilenet_%0.2f_%s" % (alpha, rows))

    # Load weights.
    if weights == "imagenet":
        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        if include_top:
            model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def MobileNet_compress(layer1_filt, layer2_filt, layer3_filt, 
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=1e-3,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):

    
    '''
    global layers
    
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()

    if kwargs:
        raise ValueError("Unknown argument(s): {(kwargs,)}")
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            "Received weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            "Received classes={classes}"
        )
    '''
    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape =_obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if depth_multiplier != 1:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "depth multiplier must be 1.  "
                "Received depth_multiplier={depth_multiplier}"
            )

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one o"
                "`0.25`, `0.50`, `0.75` or `1.0` only.  "
                "Received alpha={alpha}"
            )

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            logging.warning(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [128, 160, 192, 224]. "
                "Weights for input shape (224, 224) will be "
                "loaded as the default."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2
    )
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(
        x, layer1_filt, alpha, depth_multiplier, strides=(2, 2), block_id=4
    )
    x = _depthwise_conv_block(x, layer1_filt, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(
        x, layer2_filt, alpha, depth_multiplier, strides=(2, 2), block_id=6
    )
    x = _depthwise_conv_block(x, layer2_filt, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, layer2_filt, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, layer2_filt, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, layer2_filt, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, layer2_filt, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(
        x, layer3_filt, alpha, depth_multiplier, strides=(2, 2), block_id=12
    )
    x = _depthwise_conv_block(x, layer3_filt, alpha, depth_multiplier, block_id=13)

    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        x = layers.Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
        x = layers.Reshape((classes,), name="reshape_2")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(
            activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name="mobilenet_%0.2f_%s" % (alpha, rows))

    # Load weights.
    if weights == "imagenet":
        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        if include_top:
            model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).
    Args:
      inputs: Input tensor of shape `(rows, cols, 3)` (with `channels_last`
        data format) or (3, rows, cols) (with `channels_first` data format).
        It should have exactly 3 inputs channels, and width and height should
        be no smaller than 32. E.g. `(224, 224, 3)` would be one valid value.
      filters: Integer, the dimensionality of the output space (i.e. the
        number of output filters in the convolution).
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      kernel: An integer or tuple/list of 2 integers, specifying the width and
        height of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any `dilation_rate`
        value != 1. # Input shape
      4D tensor with shape: `(samples, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, rows, cols, channels)` if
        data_format='channels_last'. # Output shape
      4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to stride.
    Returns:
      Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    filters = int(filters * alpha)
    x = layers.Conv2D(
        filters,
        kernel,
        padding="same",
        use_bias=False,
        strides=strides,
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name="conv1_bn")(x)
    #return layers.ReLU(6.0, name="conv1_relu")(x)
    return layers.Activation('relu')(x)


def _depthwise_conv_block(
    inputs,
    pointwise_conv_filters,
    alpha,
    depth_multiplier=1,
    strides=(1, 1),
    block_id=1,
):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    Args:
      inputs: Input tensor of shape `(rows, cols, channels)` (with
        `channels_last` data format) or (channels, rows, cols) (with
        `channels_first` data format).
      pointwise_conv_filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the pointwise convolution).
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel. The total number of depthwise convolution
        output channels will be equal to `filters_in * depth_multiplier`.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any `dilation_rate`
        value != 1.
      block_id: Integer, a unique identification designating the block number.
        # Input shape
      4D tensor with shape: `(batch, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(batch, rows, cols, channels)` if
        data_format='channels_last'. # Output shape
      4D tensor with shape: `(batch, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to stride.
    Returns:
      Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(
            ((0, 1), (0, 1)), name="conv_pad_%d" % block_id
        )(inputs)
    x = layers.DepthwiseConv2D(
        (3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name="conv_dw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="conv_dw_%d_bn" % block_id
    )(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name="conv_pw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="conv_pw_%d_bn" % block_id
    )(x)
    return layers.Activation('relu')(x)



img_width, img_height = 48, 48        # Resolution of inputs
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3
#model=applications.VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

model = MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
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
predictions = Dense(output_dim = 10, activation=tf.nn.softmax)(x) # 4-way softmax classifier at the end

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = [cv2.resize(i, (48,48)) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')

x_test = [cv2.resize(i, (48,48)) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

onehot_encoder = OneHotEncoder(sparse=False)

y_train = y_train.reshape(len(y_train), 1)

onehot_encoded = onehot_encoder.fit_transform(y_train)


#model_final.fit(x_train,onehot_encoded,batch_size=64,epochs=1)

#model_final.save_weights('vgg16_1epoch.h5')
model_final.summary()
model_final.load_weights('mobilenet_100epoch.h5')


#model1.load_weights('vgg16_1epoch.h5')
layer1_b = 256
layer2_b = 512
layer3_b = 1024
layer4_b = 1024
layer5_b = 1024


layer1_a = 256
layer2_a = 512
layer3_a = 1024
layer4_a = 1024
layer5_a = 1024

model1 = None

for withtrain in range(0,5):

    wt = "mobilenet_pruned"+ str(withtrain) + ".h5"
    
    if(withtrain > 0):
        model_final = model1
    
    model_final.load_weights(wt)
    layer1_b = layer1_a
    layer2_b = layer2_a
    layer3_b = layer3_a
    layer4_b = layer4_a
    layer5_b = layer5_a


    model1 = MobileNet_compress(layer1_a,layer2_a,layer3_a,weights=None, include_top=False, input_shape=(img_width, img_height, 3))
    model1.summary()
    
    x = model1.output
    x = Flatten()(x)
    x = Dense(layer4_a, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(layer5_a, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_dim = 10, activation=tf.nn.softmax)(x) # 4-way softmax classifier at the end
    
    model1 = Model(input=model1.input, output=predictions)
    
    model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])
    
    
    y_test = y_test.reshape(len(y_test), 1)
    
    onehot_encoded = onehot_encoder.fit_transform(y_test)
   
    ####################### 1st convolution layer with 128 filters
    print('1st convolution layer with 256 filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(x_test,onehot_encoded)
    print(arr)
    
    filters = model_final.layers[27].get_weights()
    filters = np.array(filters)
    filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    shp = np.array(filters).shape
    print(shp)
    print('Shape', np.array(filters).shape)
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
        for k in range(0, layer1_a):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,10+1):
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
            
            for i in range(0,layer1_a):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[27].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer1_a):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[27].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
            arr = model_final.evaluate(x_test,onehot_encoded)
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
    
    
    A1 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    layer1_a = new_num
    
    ####################### 1st convolution layer with 512 filters
    print('1st convolution layer with 512 filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(x_test,onehot_encoded)
    print(arr)
    
    filters = model_final.layers[40].get_weights()
    filters = np.array(filters)
    filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    shp = np.array(filters).shape
    print(shp)
    
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
        for k in range(0, layer2_a):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,10+1):
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
            
            for i in range(0,layer2_a):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[40].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer2_a):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[40].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
            arr = model_final.evaluate(x_test,onehot_encoded)
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
    
    ####################### 1st convolution layer with 512 filters
    print('1st convolution layer with 512 filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(x_test,onehot_encoded)
    print(arr)
    
    filters = model_final.layers[77].get_weights()
    filters = np.array(filters)
    filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    shp = np.array(filters).shape
    print(shp)
    
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
        for k in range(0, layer3_a):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,10+1):
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
            
            for i in range(0,layer3_a):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[77].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer3_a):
                f = filters[:, :, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    #biases1[i] = 0
                    filters1[:, :, :, :, i] = 0
            
            model_final.layers[77].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
            arr = model_final.evaluate(x_test,onehot_encoded)
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

    
    print('1st dense layer with 1024 filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(x_test,onehot_encoded)
    print(arr)
    
    filters, biases = model_final.layers[87].get_weights()
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
        for k in range(0, layer4_a):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,10+1):
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
            
            for i in range(0,layer4_a):
                f = filters[ :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[87].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer4_a):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[87].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
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
    
    
    A4 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    layer4_a = new_num
    
    
    ####################### 2nd dense layer with 1024 filters
    print('2nd dense layer with 1024 filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(x_test,onehot_encoded)
    print(arr)
    
    filters, biases = model_final.layers[89].get_weights()
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
        for k in range(0, layer5_a):
            indv.append(random.randint(0,1))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+
    
    # cycle through each generation (step #2)
    for i in range(1,10+1):
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
            
            for i in range(0,layer5_a):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[89].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,layer5_a):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[89].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
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
    
    
    A5 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    layer5_a = new_num
    
    model1 = MobileNet_compress(layer1_a,layer2_a,layer3_a,weights=None, include_top=False, input_shape=(img_width, img_height, 3))
    model1.summary()
    
    x = model1.output
    x = Flatten()(x)
    x = Dense(layer4_a, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(layer5_a, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_dim = 10, activation=tf.nn.softmax)(x) # 4-way softmax classifier at the end
    
    model1 = Model(input=model1.input, output=predictions)
    
    model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])
    
    model1.summary()
    
    model = model_final
    
    for layer_idx in range(1,27):
        #print(layer_idx)
        layerr = model_final.layers[layer_idx].get_weights()
        if(layerr == None or len(layerr) == 0):
            continue
        model1.layers[layer_idx].set_weights(layerr)
    
    ######################## 1st convolution layer with 128 filters
    filters = model.layers[27].get_weights()
    filters1 = model1.layers[27].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 128, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer1_b):
        if(A1[j] == 1) :
            """
            for i1 in range (0,3):
                for j1 in range(0,3):
                    filters1[:, :, :, index1][:,:,j][i1][j1] = filters[:, :, :, i][:,:,j][i1][j1]
            """
            filters1[:, :, :, :, index1] = filters[:, :, :, :, j]
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[27].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    ######################## 2nd convolution layer with 128 filters
    filters = model.layers[33].get_weights()
    filters1 = model1.layers[33].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 128, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer1_b):
        if(A1[j] == 1) :
            index2 = 0
            for l in range(layer1_b):
                if(A1[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[33].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    ######################## 1st convolution layer with 512 filters
    filters = model.layers[40].get_weights()
    filters1 = model1.layers[40].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer2_b):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(layer1_b):
                if(A1[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[40].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    ######################## 2nd convolution layer with 512 filters
    filters = model.layers[46].get_weights()
    filters1 = model1.layers[46].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer2_b):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(layer2_b):
                if(A2[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[46].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    ######################## 3rd convolution layer with 512 filters
    filters = model.layers[52].get_weights()
    filters1 = model1.layers[52].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer2_b):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(layer2_b):
                if(A2[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[52].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    
    ######################## 4th convolution layer with 512 filters
    filters = model.layers[58].get_weights()
    filters1 = model1.layers[58].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer2_b):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(layer2_b):
                if(A2[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[58].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    ######################## 5th convolution layer with 512 filters
    filters = model.layers[64].get_weights()
    filters1 = model1.layers[64].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer2_b):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(layer2_b):
                if(A2[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[64].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    ######################## 6th convolution layer with 512 filters
    filters = model.layers[70].get_weights()
    filters1 = model1.layers[70].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer2_b):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(layer2_b):
                if(A2[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[70].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    ######################## 1st convolution layer with 1024 filters
    filters = model.layers[77].get_weights()
    filters1 = model1.layers[77].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer3_b):
        if(A3[j] == 1) :
            index2 = 0
            for l in range(layer2_b):
                if(A2[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[77].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    
    ######################## 1st convolution layer with 1024 filters
    filters = model.layers[83].get_weights()
    filters1 = model1.layers[83].get_weights()
    filters = np.array(filters)
    filters1 = np.array(filters1)
    shp = np.array(filters1).shape
    print(shp)
    #print('Shape', filters.shape)
    #print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer3_b):
        if(A3[j] == 1) :
            index2 = 0
            for l in range(layer3_b):
                if(A3[l] == 1):
                    filters1[:, :, :, index2, index1] = filters[:, :, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            #biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[83].set_weights(filters1.reshape(shp[0], shp[1], shp[2], shp[3], shp[4]))
    
    ######################## 1st Dense layer with 1024 filters
    filters, biases = model.layers[87].get_weights()
    filters1, biases1 = model1.layers[87].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer4_b):
        if(A4[j] == 1) :
            index2 = 0
            for l in range(layer3_b):
                if(A3[l] == 1):
                    filters1[index2, index1] = filters[l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[87].set_weights([filters1, biases1])
    
    
    ######################## 2nd Dense layer with 1024 filters
    
    filters, biases = model.layers[89].get_weights()
    filters1, biases1 = model1.layers[89].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(layer5_b):
        if(A5[j] == 1) :
            index2 = 0
            for l in range(layer4_b):
                if(A4[l] == 1):
                    filters1[index2, index1] = filters[l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[89].set_weights([filters1, biases1])

    
    
    arr = model1.evaluate(x_test,onehot_encoded)
    print(arr)
    
    #y_train = y_train.reshape(len(y_train), 1)
    
    onehot_encoded = onehot_encoder.fit_transform(y_train)
    
    
    model1.fit(x_train,onehot_encoded,batch_size=64,epochs=50)
    model1.summary()
    
    wt = "mobilenet_pruned"+ str(withtrain+1) + ".h5"
    model1.save_weights(wt)
    
    #y_test = y_test.reshape(len(y_test), 1)
    
    onehot_encoded = onehot_encoder.fit_transform(y_test)
    arr = model1.evaluate(x_test,onehot_encoded)
    
    
    print(arr)