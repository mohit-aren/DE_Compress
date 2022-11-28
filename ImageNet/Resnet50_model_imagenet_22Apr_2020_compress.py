# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018

@author: mohit123
"""
import numpy as np
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
import random

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import time

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
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
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=224,
                                      data_format=K.image_data_format(),
					require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((1, 1))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def ResNet50_compress(conv1_filt, conv2_filt, conv3_filt, conv4_filt,include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
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
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=224,
                                      data_format=K.image_data_format(),
					require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((1, 1))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, conv1_filt], stage=2, block='a', strides=(1, 1))
    #x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    #x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, conv2_filt], stage=3, block='a')
    #x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    #x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    #x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, conv3_filt], stage=4, block='a')
    #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, conv4_filt], stage=5, block='a')
    #x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    #x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

model=ResNet50(include_top=True, weights=None,
             input_tensor=None, input_shape=(256, 256, 3),
             pooling=None,
             classes=1000)

#ResNet50(include_top=False, input_shape=(197, 197, 3))

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
predictions = Dense(output_dim = 10, activation="softmax")(x) # 4-way softmax classifier at the end

model_final = Model(input=model.input, output=predictions)
"""
model_final = model
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model_final.summary()

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

"""
model_final.fit_generator(
        training_set,
        samples_per_epoch=nb_train_samples, 
        epochs=1,
        validation_data=test_set,
        validation_steps=nb_validation_samples/64)


model_final.save_weights('keras_resnet50_main.h5')
"""
model_final.load_weights('keras_resnet50_main.h5')

X, y = itr.next()
arr = model_final.evaluate(X,y);
print(arr)

x_test = X
onehot_encoded = y

conv_layer1 = 256
conv_layer2 = 512
conv_layer3 = 1024
conv_layer4 = 2048
conv_layer5 = 1024
conv_layer6 = 1024

####################### 1st convolution layer with 256 filters
print('1st convolution layer with 256 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[12].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

for index in range(0,10):
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = len(filters), 1
    B = []
    for j in range(0,256):
        x = random.randint(0,1)
        #print(x)
        B.append(x)
    
    for i in range(n_filters):
        f = filters[:, i]
        if(B[i] == 0):
            filters1[:,:,:, i] = 0
            biases1[i] = 0


    model_final.layers[12].set_weights([filters1, biases1])
    
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)

    arr = model_final.evaluate(x_test,onehot_encoded)
    print (arr)
    if(arr[1] > 0.10):
        A.append(B) 
        Acc.append(arr[1])
        print(index, 'added')
    

max1 = 0
max_index1 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max1):
        max1 = Acc[index]
        max_index1 = index
    index += 1
        
Acc[max_index1] = -1
max2 = 0
max_index2 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max2):
        max2 = Acc[index]
        max_index2 = index
    index += 1
        

par1 = np.copy(A[max_index1])
par2 = np.copy(A[max_index2])
print(max1, max2)
temp_sat = 0
for index_ga in range(0, 0):
    new_max = 0
    child = []
    temp_index = 0
    while (new_max < max1 and new_max < max2 and temp_index < 10):
        k = random.randint(10,200)
        
        child = np.copy(par1)
        #Crossover
        for index in range(k, 256):
            child[index] = par2[index]
            
        #Mutation
        
        for l in range(0,5):
            temp_mut = random.randint(0,255)
            child[temp_mut] = 1-child[temp_mut]
            
            
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,256):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(child[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
        
        model_final.layers[12].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        new_max = arr[1]
        print (arr)
        temp_index += 1
    if(new_max >= max1 and new_max-max1 < 0.000001):
        temp_sat += 1
    if(temp_sat >0):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,256):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(par1[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(par1[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
                
        model_final.layers[12].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        print(par1)
        print('Filters left', np.sum(par1))
        print (arr)
        break
    print('new max', new_max)
    if(new_max > max1):
        max2 = max1
        par2 = np.copy(par1)
        par1 = np.copy(child)
        max1 = new_max
        print('max1', max1)
        print(par1)
    elif(new_max > max2):
        par2 = np.copy(child)
        max2 = new_max
        print('max2', max2)

A1 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
conv_layer1 = new_num

####################### 1st convolution layer with 512 filters
print('1st convolution layer with 512 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[44].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

for index in range(0,10):
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = len(filters), 1
    B = []
    for j in range(0,512):
        x = random.randint(0,1)
        #print(x)
        B.append(x)
    
    for i in range(n_filters):
        f = filters[:, i]
        if(B[i] == 0):
            filters1[:,:,:, i] = 0
            biases1[i] = 0


    model_final.layers[44].set_weights([filters1, biases1])
    
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)

    arr = model_final.evaluate(x_test,onehot_encoded)
    print (arr)
    if(arr[1] > 0.10):
        A.append(B) 
        Acc.append(arr[1])
        print(index, 'added')
    

max1 = 0
max_index1 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max1):
        max1 = Acc[index]
        max_index1 = index
    index += 1
        
Acc[max_index1] = -1
max2 = 0
max_index2 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max2):
        max2 = Acc[index]
        max_index2 = index
    index += 1
        

par1 = np.copy(A[max_index1])
par2 = np.copy(A[max_index2])
print(max1, max2)
temp_sat = 0
for index_ga in range(0, 0):
    new_max = 0
    child = []
    temp_index = 0
    while (new_max < max1 and new_max < max2 and temp_index < 10):
        k = random.randint(10,500)
        
        child = np.copy(par1)
        #Crossover
        for index in range(k, 512):
            child[index] = par2[index]
            
        #Mutation
        
        for l in range(0,5):
            temp_mut = random.randint(0,511)
            child[temp_mut] = 1-child[temp_mut]
            
            
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,512):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(child[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
        
        model_final.layers[44].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        new_max = arr[1]
        print (arr)
        temp_index += 1
    if(new_max >= max1 and new_max-max1 < 0.000001):
        temp_sat += 1
    if(temp_sat >0):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,512):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(par1[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(par1[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
                
        model_final.layers[44].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        print(par1)
        print('Filters left', np.sum(par1))
        print (arr)
        break
    print('new max', new_max)
    if(new_max > max1):
        max2 = max1
        par2 = np.copy(par1)
        par1 = np.copy(child)
        max1 = new_max
        print('max1', max1)
        print(par1)
    elif(new_max > max2):
        par2 = np.copy(child)
        max2 = new_max
        print('max2', max2)

A2 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
conv_layer2 = new_num

####################### 1st convolution layer with 1024 filters
print('1st convolution layer with 1024 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[86].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

for index in range(0,10):
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = len(filters), 1
    B = []
    for j in range(0,1024):
        x = random.randint(0,1)
        #print(x)
        B.append(x)
    
    for i in range(n_filters):
        f = filters[:, i]
        if(B[i] == 0):
            filters1[:,:,:, i] = 0
            biases1[i] = 0


    model_final.layers[86].set_weights([filters1, biases1])
    
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)

    arr = model_final.evaluate(x_test,onehot_encoded)
    print (arr)
    if(arr[1] > 0.10):
        A.append(B) 
        Acc.append(arr[1])
        print(index, 'added')
    

max1 = 0
max_index1 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max1):
        max1 = Acc[index]
        max_index1 = index
    index += 1
        
Acc[max_index1] = -1
max2 = 0
max_index2 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max2):
        max2 = Acc[index]
        max_index2 = index
    index += 1
        

par1 = np.copy(A[max_index1])
par2 = np.copy(A[max_index2])
print(max1, max2)
temp_sat = 0
for index_ga in range(0, 0):
    new_max = 0
    child = []
    temp_index = 0
    while (new_max < max1 and new_max < max2 and temp_index < 10):
        k = random.randint(10,1000)
        
        child = np.copy(par1)
        #Crossover
        for index in range(k, 1024):
            child[index] = par2[index]
            
        #Mutation
        
        for l in range(0,5):
            temp_mut = random.randint(0,1023)
            child[temp_mut] = 1-child[temp_mut]
            
            
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,1024):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(child[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
        
        model_final.layers[86].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        new_max = arr[1]
        print (arr)
        temp_index += 1
    if(new_max >= max1 and new_max-max1 < 0.000001):
        temp_sat += 1
    if(temp_sat >0):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,1024):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(par1[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(par1[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
                
        model_final.layers[86].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        print(par1)
        print('Filters left', np.sum(par1))
        print (arr)
        break
    print('new max', new_max)
    if(new_max > max1):
        max2 = max1
        par2 = np.copy(par1)
        par1 = np.copy(child)
        max1 = new_max
        print('max1', max1)
        print(par1)
    elif(new_max > max2):
        par2 = np.copy(child)
        max2 = new_max
        print('max2', max2)

A3 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
conv_layer3 = new_num

####################### 1st convolution layer with 2048 filters
print('1st convolution layer with 2048 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[148].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

for index in range(0,10):
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = len(filters), 1
    B = []
    for j in range(0,2048):
        x = random.randint(0,1)
        #print(x)
        B.append(x)
    
    for i in range(n_filters):
        f = filters[:, i]
        if(B[i] == 0):
            filters1[:,:,:, i] = 0
            biases1[i] = 0


    model_final.layers[148].set_weights([filters1, biases1])
    
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)

    arr = model_final.evaluate(x_test,onehot_encoded)
    print (arr)
    if(arr[1] > 0.10):
        A.append(B) 
        Acc.append(arr[1])
        print(index, 'added')
    

max1 = 0
max_index1 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max1):
        max1 = Acc[index]
        max_index1 = index
    index += 1
        
Acc[max_index1] = -1
max2 = 0
max_index2 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max2):
        max2 = Acc[index]
        max_index2 = index
    index += 1
        

par1 = np.copy(A[max_index1])
par2 = np.copy(A[max_index2])
print(max1, max2)
temp_sat = 0
for index_ga in range(0, 0):
    new_max = 0
    child = []
    temp_index = 0
    while (new_max < max1 and new_max < max2 and temp_index < 10):
        k = random.randint(10,2000)
        
        child = np.copy(par1)
        #Crossover
        for index in range(k, 2048):
            child[index] = par2[index]
            
        #Mutation
        
        for l in range(0,5):
            temp_mut = random.randint(0,2047)
            child[temp_mut] = 1-child[temp_mut]
            
            
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,2048):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(child[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
        
        model_final.layers[148].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        new_max = arr[1]
        print (arr)
        temp_index += 1
    if(new_max >= max1 and new_max-max1 < 0.000001):
        temp_sat += 1
    if(temp_sat >0):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,2048):
            f = filters[:,:,:, i]
            #for j in range(3):
                #if(par1[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(par1[i] == 0)        :
                biases1[i] = 0
                filters1[:,:,:, i] = 0
                
        model_final.layers[148].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        print(par1)
        print('Filters left', np.sum(par1))
        print (arr)
        break
    print('new max', new_max)
    if(new_max > max1):
        max2 = max1
        par2 = np.copy(par1)
        par1 = np.copy(child)
        max1 = new_max
        print('max1', max1)
        print(par1)
    elif(new_max > max2):
        par2 = np.copy(child)
        max2 = new_max
        print('max2', max2)

A4 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
conv_layer4 = new_num
"""
####################### 1st dense layer with 1024 filters
print('1st dense layer with 1024 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[176].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

for index in range(0,10):
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    B = []
    for j in range(0,1024):
        x = random.randint(0,1)
        #print(x)
        B.append(x)
    
    for i in range(n_filters):
        f = filters[:, i]
        if(B[i] == 0):
            filters1[:, i] = 0
            biases1[i] = 0


    model_final.layers[176].set_weights([filters1, biases1])
    
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)

    arr = model_final.evaluate(x_test,onehot_encoded)
    print (arr)
    if(arr[1] > 0.10):
        A.append(B) 
        Acc.append(arr[1])
        print(index, 'added')
    

max1 = 0
max_index1 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max1):
        max1 = Acc[index]
        max_index1 = index
    index += 1
        
Acc[max_index1] = -1
max2 = 0
max_index2 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max2):
        max2 = Acc[index]
        max_index2 = index
    index += 1
        

par1 = np.copy(A[max_index1])
par2 = np.copy(A[max_index2])
print(max1, max2)
temp_sat = 0
for index_ga in range(0, 0):
    new_max = 0
    child = []
    temp_index = 0
    while (new_max < max1 and new_max < max2 and temp_index < 10):
        k = random.randint(10,1000)
        
        child = np.copy(par1)
        #Crossover
        for index in range(k, 1024):
            child[index] = par2[index]
            
        #Mutation
        
        for l in range(0,5):
            temp_mut = random.randint(0,1023)
            child[temp_mut] = 1-child[temp_mut]
            
            
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(child[i] == 0)        :
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[176].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        new_max = arr[1]
        print (arr)
        temp_index += 1
    if(new_max >= max1 and new_max-max1 < 0.000001):
        temp_sat += 1
    if(temp_sat >0):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(par1[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(par1[i] == 0)        :
                biases1[i] = 0
                filters1[:, i] = 0
                
        model_final.layers[176].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        print(par1)
        print('Filters left', np.sum(par1))
        print (arr)
        break
    print('new max', new_max)
    if(new_max > max1):
        max2 = max1
        par2 = np.copy(par1)
        par1 = np.copy(child)
        max1 = new_max
        print('max1', max1)
        print(par1)
    elif(new_max > max2):
        par2 = np.copy(child)
        max2 = new_max
        print('max2', max2)

A5 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
conv_layer5 = new_num


####################### 2nd dense layer with 1024 filters
print('2nd dense layer with 1024 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[178].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

for index in range(0,10):
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 1024, 1
    B = []
    for j in range(0,1024):
        x = random.randint(0,1)
        #print(x)
        B.append(x)
    
    for i in range(n_filters):
        f = filters[:, i]
        if(B[i] == 0):
            filters1[:, i] = 0
            biases1[i] = 0


    model_final.layers[178].set_weights([filters1, biases1])
    
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)

    arr = model_final.evaluate(x_test,onehot_encoded)
    print (arr)
    if(arr[1] > 0.10):
        A.append(B) 
        Acc.append(arr[1])
        print(index, 'added')
    

max1 = 0
max_index1 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max1):
        max1 = Acc[index]
        max_index1 = index
    index += 1
        
Acc[max_index1] = -1
max2 = 0
max_index2 = 0
index = 0
while index < len(Acc):
    if(Acc[index]> max2):
        max2 = Acc[index]
        max_index2 = index
    index += 1
        

par1 = np.copy(A[max_index1])
par2 = np.copy(A[max_index2])
print(max1, max2)
temp_sat = 0
for index_ga in range(0, 0):
    new_max = 0
    child = []
    temp_index = 0
    while (new_max < max1 and new_max < max2 and temp_index < 10):
        k = random.randint(10,1000)
        
        child = np.copy(par1)
        #Crossover
        for index in range(k, 1024):
            child[index] = par2[index]
            
        #Mutation
        
        for l in range(0,5):
            temp_mut = random.randint(0,1023)
            child[temp_mut] = 1-child[temp_mut]
            
            
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(child[i] == 0)        :
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[178].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        new_max = arr[1]
        print (arr)
        temp_index += 1
    if(new_max >= max1 and new_max-max1 < 0.000001):
        temp_sat += 1
    if(temp_sat >0):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(par1[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(par1[i] == 0)        :
                biases1[i] = 0
                filters1[:, i] = 0
                
        model_final.layers[178].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        print(par1)
        print('Filters left', np.sum(par1))
        print (arr)
        break
    print('new max', new_max)
    if(new_max > max1):
        max2 = max1
        par2 = np.copy(par1)
        par1 = np.copy(child)
        max1 = new_max
        print('max1', max1)
        print(par1)
    elif(new_max > max2):
        par2 = np.copy(child)
        max2 = new_max
        print('max2', max2)

A6 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
conv_layer6 = new_num

"""

model_compress=ResNet50_compress(conv_layer1, conv_layer2, conv_layer3, conv_layer4, include_top=True, weights=None,
             input_tensor=None, input_shape=(256, 256, 3),
             pooling=None,
             classes=1000)

#ResNet50(include_top=False, input_shape=(197, 197, 3))

"""
# Freeze first 15 layers
for layer in model.layers[:45]:
	layer.trainable = False
for layer in model.layers[45:]:
   layer.trainable = True


x = model_compress.output
x = Flatten()(x)
x = Dense(conv_layer5, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(conv_layer6, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(output_dim = 1000, activation="softmax")(x) # 4-way softmax classifier at the end

model1 = Model(input=model_compress.input, output=predictions)
"""
model1 = model_compress
model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()
for layer_idx in range(0,12):
    if(len(model.layers[layer_idx].get_weights()) == 0):
        continue

    #print(layer_idx)
    layerr = model_final.layers[layer_idx].get_weights()
    model1.layers[layer_idx].set_weights(layerr)
for k in range(12,14):
    if(len(model.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)


for k in range(18,19):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+20].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)


for k in range(21,22):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+20].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)

for k in range(24,26):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+20].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)


for k in range(30,31):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+50].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)

for k in range(33,34):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+50].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)

for k in range(36,38):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+50].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)

for k in range(42,43):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+100].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)

for k in range(45,46):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+100].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)

for k in range(48,50):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+100].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)

"""
######################## 1st dense layer with 1024 filters
filters, biases = model_final.layers[176].get_weights()
filters1, biases1 = model1.layers[56].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)

######################## 2nd dense layer with 1024 filters
filters, biases = model_final.layers[178].get_weights()
filters1, biases1 = model1.layers[58].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
"""

arr = model1.evaluate(x_test,onehot_encoded)
print(arr)

#y_train = y_train.reshape(len(y_train), 1)


for layer_idx in range(0,12):
    #print(layer_idx)
    layerr = model_final.layers[layer_idx].get_weights()
    model1.layers[layer_idx].set_weights(layerr)
for k in range(12,14):
    if(len(model.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A1[j] == 1) :
            """
            for i1 in range (0,3):
                for j1 in range(0,3):
                    filters1[:, :, index1][:,:,j][i1][j1] = filters[:, :, i][:,:,j][i1][j1]
            """
            filters1[:, :, :, index1] = filters[:, :, :, j]
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(14,18):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A1[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(18,19):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+20].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 128, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(128):
        index2 = 0
        for l in range(256):
            if(A1[l] == 1) :
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(19,20):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+20].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(128):
        if(A1[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(21,22):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+20].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 128, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(128):
        index2 = 0
        for l in range(128):
            filters1[:, :, index2, index1] = filters[:, :, l, j]
            index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(22,23):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+20].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(128):
        if(A1[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(24,26):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+20].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 512, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    if(k == 24):
        for j in range(512):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(128):
                    if(1 == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)  
    else:
        index1 = 0
        # plot each channel separately
        for j in range(512):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(256):
                    if(A1[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1      
    model1.layers[k].set_weights([filters1, biases1])

for k in range(26,28):
    if(len(model_final.layers[k+20].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+20].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(A2[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(30,31):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+50].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(1 == 1) :
            index2 = 0
            for l in range(512):
                if(A2[l] == 1) :
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(31,32):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+50].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A2[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(33,34):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+50].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 256, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(256):
        index2 = 0
        for l in range(256):
            if(1 == 1) :
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
        index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(34,35):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+50].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A2[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(36,38):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+50].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
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
    if(k == 36):
        for j in range(1024):
            if(A3[j] == 1) :
                index2 = 0
                for l in range(256):
                    if(1 == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)       
    else:
        for j in range(1024):
            if(A3[j] == 1) :
                index2 = 0
                for l in range(512):
                    if(A2[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
 
    model1.layers[k].set_weights([filters1, biases1])

for k in range(38,40):
    if(len(model_final.layers[k+50].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+50].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(1024):
        if(A3[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(42,43):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+100].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 512, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(1 == 1):
            index2 = 0
            for l in range(1024):
                if(A3[l] == 1) :
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(43,44):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+100].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(A3[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(45,46):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+100].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 512, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(512):
        index2 = 0
        for l in range(512):
            filters1[:, :, index2, index1] = filters[:, :, l, j]
            index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(46,47):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+100].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(512):
        if(A3[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)

for k in range(48,50):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k+100].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 2048, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    if(k == 48):
        for j in range(2048):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(512):
                    if(1 == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
    else:
        for j in range(2048):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(1024):
                    if(A3[l] == 1) :
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1

    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(50,52):
    if(len(model_final.layers[k+100].get_weights()) == 0):
        continue
    
    A_1 = model_final.layers[k+100].get_weights()
    A_2 = model1.layers[k].get_weights()
    
    print('Len1', len(A_1))
    print('Len2', len(A_2))
    index1 = 0
    # plot each channel separately
    for j in range(2048):
        if(A4[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights(A_2)
"""
######################## 1st dense layer with 1024 filters
filters, biases = model_final.layers[176].get_weights()
filters1, biases1 = model1.layers[56].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 1024, 1

index1 = 0
# plot each channel separately
for j in range(1024):
    if(A5[j] == 1) :
        index2 = 0
        for l in range(2048):
            if(A4[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[56].set_weights([filters1, biases1])

######################## 2nd dense layer with 1024 filters
filters, biases = model_final.layers[178].get_weights()
filters1, biases1 = model1.layers[58].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 1024, 1

index1 = 0
# plot each channel separately
for j in range(1024):
    if(A6[j] == 1) :
        index2 = 0
        for l in range(1024):
            if(A5[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[58].set_weights([filters1, biases1])
"""

arr = model1.evaluate(x_test,onehot_encoded)
print(arr)

#y_train = y_train.reshape(len(y_train), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_train)

model1.fit_generator(
        training_set,
        samples_per_epoch=nb_train_samples, 
        epochs=50,
        validation_data=test_set,
        validation_steps=nb_validation_samples/64)


#model1.fit(x_train,onehot_encoded,batch_size=64,epochs=100)
model1.summary()
model1.save_weights('Resnet_pruned_weights.h5')

#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
arr = model1.evaluate(x_test,onehot_encoded)


print(arr)
