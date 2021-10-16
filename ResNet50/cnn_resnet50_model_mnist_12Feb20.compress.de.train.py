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
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import random 

img_width, img_height = 48, 48        # Resolution of inputs
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3

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
                                      min_size=48,
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

    x = ZeroPadding2D((78, 78))(img_input)
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
                                      min_size=48,
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

    x = ZeroPadding2D((78, 78))(img_input)
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
"""
model=ResNet50(include_top=False, weights=None,
             input_tensor=None, input_shape=(48, 48, 3),
             pooling=None,
             classes=1000)

#ResNet50(include_top=False, input_shape=(197, 197, 3))


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

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model_final.summary()


model_final.load_weights('resnet50_100epoch.h5')
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = [cv2.cvtColor(cv2.resize(i, (48,48)), cv2.COLOR_GRAY2BGR) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')

x_test = [cv2.cvtColor(cv2.resize(i, (48,48)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

onehot_encoder = OneHotEncoder(sparse=False)

y_train = y_train.reshape(len(y_train), 1)

onehot_encoded = onehot_encoder.fit_transform(y_train)


y_test = y_test.reshape(len(y_test), 1)

onehot_encoded = onehot_encoder.fit_transform(y_test)

#arr = model_final.evaluate(x_test,onehot_encoded)

#print(arr)

conv_layer1 = 63
conv_layer2 = 114
conv_layer3 = 229
conv_layer4 = 452
conv_layer5 = 232
conv_layer6 = 211
"""
####################### 1st convolution layer with 256 filters
print('1st convolution layer with 256 filters')
filters, biases = model_final.layers[12].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

A= []
Acc = []
#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
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
    for k in range(0, 256):
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
        
        for i in range(0,256):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model.layers[12].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,256):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[12].set_weights([filters1, biases1])
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
conv_layer1 = new_num

####################### 1st convolution layer with 512 filters
print('1st convolution layer with 512 filters')
filters, biases = model_final.layers[44].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

A= []
Acc = []
#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
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
    for k in range(0, 512):
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
        
        for i in range(0,512):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[44].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,512):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[44].set_weights([filters1, biases1])
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
conv_layer2 = new_num


####################### 1st convolution layer with 1024 filters
print('1st convolution layer with 1024 filters')
filters, biases = model_final.layers[86].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

A= []
Acc = []
#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
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
    for k in range(0, 1024):
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
        
        for i in range(0,1024):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[86].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,1024):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[86].set_weights([filters1, biases1])
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
conv_layer3 = new_num


####################### 1st convolution layer with 2048 filters
print('1st convolution layer with 2048 filters')
filters, biases = model_final.layers[148].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

A= []
Acc = []
#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
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
    for k in range(0, 2048):
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
        
        for i in range(0,2048):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[148].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,2048):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[148].set_weights([filters1, biases1])
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
conv_layer4 = new_num


####################### 1st dense layer with 1024 filters
print('1st dense layer with 1024 filters')
filters, biases = model_final.layers[176].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

A= []
Acc = []
#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
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
    for k in range(0, 1024):
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
        
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[ :, i] = 0
        
        model_final.layers[176].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[176].set_weights([filters1, biases1])
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
conv_layer5 = new_num

print('--------------- 2nd Dense layer 1024 layers ---')

filters, biases = model_final.layers[178].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

A= []
Acc = []
#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
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
    for k in range(0, 1024):
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
        
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[ :, i] = 0
        
        model_final.layers[178].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,1024):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[178].set_weights([filters1, biases1])
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

A6 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
conv_layer6 = new_num


model_compress=ResNet50_compress(conv_layer1, conv_layer2, conv_layer3, conv_layer4, include_top=False, weights=None,
             input_tensor=None, input_shape=(48, 48, 3),
             pooling=None,
             classes=1000)

#ResNet50(include_top=False, input_shape=(197, 197, 3))


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
predictions = Dense(output_dim = 10, activation="softmax")(x) # 4-way softmax classifier at the end

model1 = Model(input=model_compress.input, output=predictions)

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()
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
    
    index1 = 0
    # plot each channel separately
    for j in range(256):
        if(A1[j] == 1) :
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

arr = model1.evaluate(x_test,onehot_encoded)
print(arr)

#y_train = y_train.reshape(len(y_train), 1)

onehot_encoded = onehot_encoder.fit_transform(y_train)


model1.fit(x_train,onehot_encoded,batch_size=64,epochs=10)
model1.summary()
model1.save_weights('Resnet_pruned_weights.5.h5')

#y_test = y_test.reshape(len(y_test), 1)

onehot_encoded = onehot_encoder.fit_transform(y_test)
arr = model1.evaluate(x_test,onehot_encoded)


print(arr)
"""


for withtrain in range(1,5):
    
    ####################### 1st convolution layer with 256 filters
    model_compress=ResNet50_compress(conv_layer1, conv_layer2, conv_layer3, conv_layer4, include_top=False, weights=None,
                 input_tensor=None, input_shape=(48, 48, 3),
                 pooling=None,
                 classes=1000)
    
    #ResNet50(include_top=False, input_shape=(197, 197, 3))
    
    
    # Freeze first 15 layers
    """
    for layer in model.layers[:45]:
    	layer.trainable = False
    for layer in model.layers[45:]:
       layer.trainable = True
    """
    
    x = model_compress.output
    x = Flatten()(x)
    x = Dense(conv_layer5, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(conv_layer6, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_dim = 10, activation="softmax")(x) # 4-way softmax classifier at the end
    
    model_final = Model(input=model_compress.input, output=predictions)
    
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])
    
    model_final.summary()

    oconv_layer1 = conv_layer1
    oconv_layer2 = conv_layer2
    oconv_layer3 = conv_layer3
    oconv_layer4 = conv_layer4
    oconv_layer5 = conv_layer5
    oconv_layer6 = conv_layer6
    
    wt = "Resnet_pruned_weights_" + str(withtrain) + ".h5"

    model_final.load_weights(wt);
    print('1st convolution layer with 256 filters')
    filters, biases = model_final.layers[12].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    A= []
    Acc = []
    #y_test = y_test.reshape(len(y_test), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_test)
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
        for k in range(0, oconv_layer1):
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
            
            for i in range(0,oconv_layer1):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[12].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,oconv_layer1):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[12].set_weights([filters1, biases1])
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
    conv_layer1 = new_num
    
    ####################### 1st convolution layer with oconv_layer2 filters
    print('1st convolution layer with oconv_layer2 filters')
    filters, biases = model_final.layers[24].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    A= []
    Acc = []
    #y_test = y_test.reshape(len(y_test), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_test)
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
        for k in range(0, oconv_layer2):
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
            
            for i in range(0,oconv_layer2):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[24].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,oconv_layer2):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[24].set_weights([filters1, biases1])
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
    conv_layer2 = new_num
    
    
    ####################### 1st convolution layer with 1024 filters
    print('1st convolution layer with oconv_layer3 filters')
    filters, biases = model_final.layers[36].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    A= []
    Acc = []
    #y_test = y_test.reshape(len(y_test), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_test)
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
        for k in range(0, oconv_layer3):
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
            
            for i in range(0,oconv_layer3):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[36].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,oconv_layer3):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[36].set_weights([filters1, biases1])
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
    conv_layer3 = new_num
    
    
    ####################### 1st convolution layer with oconv_layer4 filters
    print('1st convolution layer with oconv_layer4 filters')
    filters, biases = model_final.layers[48].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    A= []
    Acc = []
    #y_test = y_test.reshape(len(y_test), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_test)
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
        for k in range(0, oconv_layer4):
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
            
            for i in range(0,oconv_layer4):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[48].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,oconv_layer4):
                f = filters[:, :, :, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, :, :, i] = 0
            
            model_final.layers[48].set_weights([filters1, biases1])
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
    conv_layer4 = new_num
    
    
    ####################### 1st dense layer with oconv_layer3 filters
    print('1st dense layer with oconv_layer5 filters')
    filters, biases = model_final.layers[56].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    A= []
    Acc = []
    #y_test = y_test.reshape(len(y_test), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_test)
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
        for k in range(0, oconv_layer5):
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
            
            for i in range(0,oconv_layer5):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[ :, i] = 0
            
            model_final.layers[56].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,oconv_layer5):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[56].set_weights([filters1, biases1])
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
    conv_layer5 = new_num
    
    print('--------------- 2nd Dense layer oconv_layer6 layers ---')
    
    filters, biases = model_final.layers[58].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    A= []
    Acc = []
    #y_test = y_test.reshape(len(y_test), 1)
    
    #onehot_encoded = onehot_encoder.fit_transform(y_test)
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
        for k in range(0, oconv_layer6):
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
            
            for i in range(0,oconv_layer6):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(v_trial[i] == 0):
                    biases1[i] = 0
                    filters1[ :, i] = 0
            
            model_final.layers[58].set_weights([filters1, biases1])
            arr = model_final.evaluate(x_test,onehot_encoded)
            score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
            
            #score_trial  = cost_func(v_trial)
            for i in range(0,oconv_layer6):
                f = filters[:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(x_t[i] == 0):
                    biases1[i] = 0
                    filters1[:, i] = 0
            
            model_final.layers[58].set_weights([filters1, biases1])
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
    
    A6 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    conv_layer6 = new_num
    
    
    model_compress=ResNet50_compress(conv_layer1, conv_layer2, conv_layer3, conv_layer4, include_top=False, weights=None,
                 input_tensor=None, input_shape=(48, 48, 3),
                 pooling=None,
                 classes=1000)
    
    #ResNet50(include_top=False, input_shape=(197, 197, 3))
    
    
    # Freeze first 15 layers
    """
    for layer in model.layers[:45]:
    	layer.trainable = False
    for layer in model.layers[45:]:
       layer.trainable = True
    """
    
    x = model_compress.output
    x = Flatten()(x)
    x = Dense(conv_layer5, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(conv_layer6, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_dim = 10, activation="softmax")(x) # 4-way softmax classifier at the end
    
    model1 = Model(input=model_compress.input, output=predictions)
    
    model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])
    
    model1.summary()
    
    """
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
        filters1, biases1 = model1.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = oconv_layer1, 1
        
        """
        for i in range(n_filters):
            f = filters[:, :, i]
        """
        index1 = 0
        # plot each channel separately
        for j in range(oconv_layer1):
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
        for j in range(oconv_layer1):
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
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
            for l in range(oconv_layer1):
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(128):
            #if(A1[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
        #print(biases1, biases)        
        model1.layers[k].set_weights(A_2)
    
    for k in range(21,22):
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(128):
            #if(A1[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
        #print(biases1, biases)        
        model1.layers[k].set_weights(A_2)
    
    for k in range(24,26):
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
        filters1, biases1 = model1.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = oconv_layer2, 1
        
        """
        for i in range(n_filters):
            f = filters[:, :, i]
        """
        index1 = 0
        # plot each channel separately
        if(k == 24):
            for j in range(oconv_layer2):
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
            for j in range(oconv_layer2):
                if(A2[j] == 1) :
                    index2 = 0
                    for l in range(oconv_layer1):
                        if(A1[l] == 1) :
                            filters1[:, :, index2, index1] = filters[:, :, l, j]
                            index2 += 1
                    #biases1[:, :, :, index1] = biases[:, :, :, i]
                    biases1[index1] = biases[j]
                    #print(index1,i)
                    index1 += 1      
        model1.layers[k].set_weights([filters1, biases1])
    
    for k in range(26,28):
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(oconv_layer2):
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
        if(len(model_final.layers[k].get_weights()) == 0):
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
            if(1 == 1) :
                index2 = 0
                for l in range(oconv_layer2):
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(256):
            #if(A2[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
        #print(biases1, biases)        
        model1.layers[k].set_weights(A_2)
    
    for k in range(33,34):
        if(len(model_final.layers[k].get_weights()) == 0):
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(256):
            #if(A2[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
        #print(biases1, biases)        
        model1.layers[k].set_weights(A_2)
    
    for k in range(36,38):
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
        filters1, biases1 = model1.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = oconv_layer3, 1
        
        """
        for i in range(n_filters):
            f = filters[:, :, i]
        """
        index1 = 0
        # plot each channel separately
        if(k == 36):
            for j in range(oconv_layer3):
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
            for j in range(oconv_layer3):
                if(A3[j] == 1) :
                    index2 = 0
                    for l in range(oconv_layer2):
                        if(A2[l] == 1) :
                            filters1[:, :, index2, index1] = filters[:, :, l, j]
                            index2 += 1
                    #biases1[:, :, :, index1] = biases[:, :, :, i]
                    biases1[index1] = biases[j]
                    #print(index1,i)
                    index1 += 1
     
        model1.layers[k].set_weights([filters1, biases1])
    
    for k in range(38,40):
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(oconv_layer3):
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
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
                for l in range(oconv_layer3):
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(512):
            #if(A3[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
        #print(biases1, biases)        
        model1.layers[k].set_weights(A_2)
    
    for k in range(45,46):
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(512):
            #if(A3[j] == 1) :
            A_2[0][index1] = A_1[0][j]
            A_2[1][index1] = A_1[1][j]
            A_2[2][index1] = A_1[2][j]
            A_2[3][index1] = A_1[3][j]
            #print(index1,i)
            index1 += 1
        #print(biases1, biases)        
        model1.layers[k].set_weights(A_2)
    
    for k in range(48,50):
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        filters, biases = model_final.layers[k].get_weights()
        filters1, biases1 = model1.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = oconv_layer4, 1
        
        """
        for i in range(n_filters):
            f = filters[:, :, i]
        """
        index1 = 0
        # plot each channel separately
        if(k == 48):
            for j in range(oconv_layer4):
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
            for j in range(oconv_layer4):
                if(A4[j] == 1) :
                    index2 = 0
                    for l in range(oconv_layer3):
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
        if(len(model_final.layers[k].get_weights()) == 0):
            continue
        
        A_1 = model_final.layers[k].get_weights()
        A_2 = model1.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(oconv_layer4):
            if(A4[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        model1.layers[k].set_weights(A_2)
    
    ######################## 1st dense layer with 1024 filters
    filters, biases = model_final.layers[56].get_weights()
    filters1, biases1 = model1.layers[56].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = oconv_layer5, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(oconv_layer5):
        if(A5[j] == 1) :
            index2 = 0
            for l in range(oconv_layer4):
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
    filters, biases = model_final.layers[58].get_weights()
    filters1, biases1 = model1.layers[58].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = oconv_layer6, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(oconv_layer6):
        if(A6[j] == 1) :
            index2 = 0
            for l in range(oconv_layer5):
                if(A5[l] == 1):
                    filters1[index2, index1] = filters[l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[58].set_weights([filters1, biases1])
    
    arr = model1.evaluate(x_test,onehot_encoded)
    print(arr)
    
    #y_train = y_train.reshape(len(y_train), 1)
    
    onehot_encoded = onehot_encoder.fit_transform(y_train)
    
    
    model1.fit(x_train,onehot_encoded,batch_size=64,epochs=10)
    model1.summary()
    wt1 = "Resnet_pruned_weights_" + str(withtrain+1) + ".h5"
    model1.save_weights(wt1)
    
    #y_test = y_test.reshape(len(y_test), 1)
    
    onehot_encoded = onehot_encoder.fit_transform(y_test)
    arr = model1.evaluate(x_test,onehot_encoded)
    
    
    print(arr)
