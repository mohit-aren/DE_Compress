# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018
@author: mohit123
"""
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
#from sklearn.preprocessing import OneHotEncoder
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



img_width, img_height = 256, 256        # Resolution of inputs
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
predictions = Dense(output_dim = 1000, activation=tf.nn.softmax)(x) # 4-way softmax classifier at the end

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical' )
label_map = (training_set.class_indices)

print(label_map)

itr = test_set = test_datagen.flow_from_directory(
        'val',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

#model_final.save_weights('keras_resnet50_main.h5')

nb_train_samples = 1300000                # Number of train samples
nb_validation_samples = 50000            # Number of validation samples


model_final.fit_generator(
        training_set,
        samples_per_epoch=nb_train_samples, 
        epochs=1,
        validation_data=test_set,
        validation_steps=nb_validation_samples/64)

model_final.save_weights('mobilenet_100epoch.h5')

X, y = itr.next()
arr = model_final.evaluate(X,y);
print(arr)

x_test = X
onehot_encoded = y

arr = model_final.evaluate(x_test,onehot_encoded)

print(arr)