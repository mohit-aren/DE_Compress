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


from keras.models import Model
from keras.layers import Add, Activation, Concatenate, Conv2D, Dropout 
from keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D
import keras.backend as K

__version__ = '0.0.1'


def SqueezeNet(input_shape, nb_classes, use_bypass=False, dropout_rate=None, compression=1.0):
    """
    Creating a SqueezeNet of version 1.0
    
    Arguments:
        input_shape  : shape of the input images e.g. (224,224,3)
        nb_classes   : number of classes
        use_bypass   : if true, bypass connections will be created at fire module 3, 5, 7, and 9 (default: False)
        dropout_rate : defines the dropout rate that is accomplished after last fire module (default: None)
        compression  : reduce the number of feature-maps (default: 1.0)
        
    Returns:
        Model        : Keras model instance
    """
    
    input_img = Input(shape=input_shape)
    x = ZeroPadding2D((90,90), data_format="channels_last")(input_img)

    x = Conv2D(int(96*compression), (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_fire_module(x, int(16*compression), name='fire2')
    x = create_fire_module(x, int(16*compression), name='fire3', use_bypass=use_bypass)
    x = create_fire_module(x, int(32*compression), name='fire4')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)
    
    x = create_fire_module(x, int(32*compression), name='fire5', use_bypass=use_bypass)
    x = create_fire_module(x, int(48*compression), name='fire6')
    x = create_fire_module(x, int(48*compression), name='fire7', use_bypass=use_bypass)
    x = create_fire_module(x, int(64*compression), name='fire8')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool8')(x)
    
    x = create_fire_module(x, int(64*compression), name='fire9', use_bypass=use_bypass)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
        
    x = output(x, nb_classes)

    return Model(inputs=input_img, outputs=x)


def SqueezeNet_compress(fire6_filters, fire7_filters, fire8_filters, fire9_filters, input_shape, nb_classes, use_bypass=False, dropout_rate=None, compression=1.0):
    """
    Creating a SqueezeNet of version 1.0
    
    Arguments:
        input_shape  : shape of the input images e.g. (224,224,3)
        nb_classes   : number of classes
        use_bypass   : if true, bypass connections will be created at fire module 3, 5, 7, and 9 (default: False)
        dropout_rate : defines the dropout rate that is accomplished after last fire module (default: None)
        compression  : reduce the number of feature-maps (default: 1.0)
        
    Returns:
        Model        : Keras model instance
    """
    
    input_img = Input(shape=input_shape)
    x = ZeroPadding2D((90,90), data_format="channels_last")(input_img)

    x = Conv2D(int(96*compression), (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_fire_module(x, int(16*compression), name='fire2')
    x = create_fire_module(x, int(16*compression), name='fire3', use_bypass=use_bypass)
    x = create_fire_module(x, int(32*compression), name='fire4')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)
    
    x = create_fire_module(x, int(32*compression), name='fire5', use_bypass=use_bypass)

    print('fire6_filters',fire6_filters)
    x = create_fire_module_compress(fire6_filters, x, int(48*compression), name='fire6')
    x = create_fire_module_compress(fire7_filters, x, int(48*compression), name='fire7', use_bypass=use_bypass)
    x = create_fire_module_compress(fire8_filters, x, int(64*compression), name='fire8')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool8')(x)
    
    x = create_fire_module_compress(fire9_filters, x, int(64*compression), name='fire9', use_bypass=use_bypass)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
        
    x = output(x, nb_classes)

    return Model(inputs=input_img, outputs=x)


def SqueezeNet_11(input_shape, nb_classes, dropout_rate=None, compression=1.0):
    """
    Creating a SqueezeNet of version 1.1
    
    2.4x less computation over SqueezeNet 1.0 implemented above.
    
    Arguments:
        input_shape  : shape of the input images e.g. (224,224,3)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after last fire module (default: None)
        compression  : reduce the number of feature-maps
        
    Returns:
        Model        : Keras model instance
    """
    
    input_img = Input(shape=input_shape)

    x = Conv2D(int(64*compression), (3,3), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_fire_module(x, int(16*compression), name='fire2')
    x = create_fire_module(x, int(16*compression), name='fire3')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool3')(x)
    
    x = create_fire_module(x, int(32*compression), name='fire4')
    x = create_fire_module(x, int(32*compression), name='fire5')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool5')(x)
    
    x = create_fire_module(x, int(48*compression), name='fire6')
    x = create_fire_module(x, int(48*compression), name='fire7')
    x = create_fire_module(x, int(64*compression), name='fire8')
    x = create_fire_module(x, int(64*compression), name='fire9')

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    # Creating last conv10
    x = output(x, nb_classes)

    return Model(inputs=input_img, outputs=x)


def output(x, nb_classes):
    x = Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
    x = GlobalAveragePooling2D(name='avgpool10')(x)
    x = Activation("softmax", name='softmax')(x)
    return x


def create_fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    """
    Creates a fire module
    
    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123
    
    Returns:
        x                 : returns a fire module
    """
    
    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    
    axis = get_axis()
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret


def create_fire_module_compress(fire_filters, x, nb_squeeze_filter, name, use_bypass=False):
    """
    Creates a fire module
    
    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123
    
    Returns:
        x                 : returns a fire module
    """
    
    #nb_expand_filter = 4 * nb_squeeze_filter
    nb_expand_filter = fire_filters
    print('fire_filters', fire_filters)
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    expand_1x1 = Conv2D(fire_filters, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(fire_filters, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    
    axis = get_axis()
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret


def get_axis():
    axis = -1 if K.image_data_format() == 'channels_last' else 1
    return axis
model_final=SqueezeNet((48,48,3), 10)

#ResNet50(include_top=False, input_shape=(197, 197, 3))

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model_final.summary()

layer1_filters = 192
layer2_filters = 192
layer3_filters = 256
layer4_filters = 256

model1=SqueezeNet_compress(layer1_filters,layer2_filters,layer3_filters,layer4_filters, (48,48,3), 10)

#ResNet50(include_top=False, input_shape=(197, 197, 3))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()


model_final.load_weights('sqeezenet_100epoch.h5')

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

arr = model_final.evaluate(x_test,onehot_encoded)

print(arr)

####################### 1st convolution layer with 192 filters
print('1st convolution layer with 192 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[23].get_weights()
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
    for k in range(0, 192):
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
        
        for i in range(0,192):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[23].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,192):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[23].set_weights([filters1, biases1])
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
layer1_filters = new_num


####################### 2nd convolution layer with 192 filters
print('2nd convolution layer with 192 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[27].get_weights()
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
    for k in range(0, 192):
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
        
        for i in range(0,192):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[27].set_weights([filters1, biases1])
        arr = model_final.evaluate(x_test,onehot_encoded)
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,192):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[27].set_weights([filters1, biases1])
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
layer2_filters = new_num


####################### 1st convolution layer with 256 filters
print('1st convolution layer with 256 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[31].get_weights()
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
        
        model_final.layers[31].set_weights([filters1, biases1])
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
        
        model_final.layers[31].set_weights([filters1, biases1])
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
layer3_filters = new_num


####################### 2nd convolution layer with 256 filters
print('2nd convolution layer with 256 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[35].get_weights()
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
        
        model_final.layers[35].set_weights([filters1, biases1])
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
        
        model_final.layers[35].set_weights([filters1, biases1])
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
layer4_filters = new_num



model1=SqueezeNet_compress(layer1_filters,layer2_filters,layer3_filters,layer4_filters, (48,48,3), 10)

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])


model1.summary()
for layer_idx in range(0,22):
    #print(layer_idx)
    layerr = model_final.layers[layer_idx].get_weights()
    model1.layers[layer_idx].set_weights(layerr)
for k in range(23,26):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)


for k in range(27,30):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)



for k in range(31,34):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)


for k in range(36,39):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)



arr = model1.evaluate(x_test,onehot_encoded)
print(arr)

#y_train = y_train.reshape(len(y_train), 1)


for k in range(23,25):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 192, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(192):
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

for k in range(25,26):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 192, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(48):
        if(1 == 1) :
            index2  = 0
            for l in range(192):
                if(A1[l] == 1):
                    filters1[:, :, index2*2, index1] = filters[:, :, l*2, j]
                    filters1[:, :, index2*2+1, index1] = filters[:, :, l*2+1, j]
                    index2 += 1
            biases1[index1] = biases[j]
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(27,29):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 192, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(192):
        if(A2[j] == 1) :
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

for k in range(29,30):
    if(len(model_final.layers[k].get_weights()) == 0):
        continue
    filters, biases = model_final.layers[k].get_weights()
    filters1, biases1 = model1.layers[k].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = 192, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(64):
        if(1 == 1) :
            index2  = 0
            for l in range(192):
                if(A2[l] == 1):
                    filters1[:, :, index2*2, index1] = filters[:, :, l*2, j]
                    filters1[:, :, index2*2+1, index1] = filters[:, :, l*2+1, j]
                    index2 += 1
            biases1[index1] = biases[j]
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])

for k in range(30,32):
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
        if(A3[j] == 1) :
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

for k in range(32,35):
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
    for j in range(64):
        if(1 == 1) :
            index2  = 0
            for l in range(256):
                if(A3[l] == 1):
                    filters1[:, :, index2*2, index1] = filters[:, :, l*2, j]
                    filters1[:, :, index2*2+1, index1] = filters[:, :, l*2+1, j]
                    index2 += 1
            biases1[index1] = biases[j]
            index1 += 1
    #print(biases1, biases)        
    model1.layers[k].set_weights([filters1, biases1])


for k in range(35,37):
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
        if(A4[j] == 1) :
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

arr = model1.evaluate(x_test,onehot_encoded)
print(arr)

#y_train = y_train.reshape(len(y_train), 1)

onehot_encoded = onehot_encoder.fit_transform(y_train)


model1.fit(x_train,onehot_encoded,batch_size=64,epochs=10)
model1.summary()
model1.save_weights('sqeezenet_pruned_weights.4.h5')

#y_test = y_test.reshape(len(y_test), 1)

onehot_encoded = onehot_encoder.fit_transform(y_test)
arr = model1.evaluate(x_test,onehot_encoded)


print(arr)
