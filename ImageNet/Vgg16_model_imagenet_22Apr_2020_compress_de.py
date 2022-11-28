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

import numpy as np
import warnings

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
from keras.datasets import mnist
#from sklearn.preprocessing import OneHotEncoder
from keras.layers import Activation
import random
import time

img_width, img_height = 224, 224        # Resolution of inputs
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3
model=applications.VGG16(weights=None, include_top=True, input_shape=(img_width, img_height, 3), classes=1000)

#initialise top model
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
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical' )
label_map = (training_set.class_indices)

print(label_map)

itr = test_set = test_datagen.flow_from_directory(
        'val',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')
nb_train_samples = 1300000                # Number of train samples
nb_validation_samples = 50000            # Number of validation samples


"""
model_final.fit_generator(
        training_set,
        samples_per_epoch=nb_train_samples, 
        epochs=1,
        validation_data=test_set,
        validation_steps=nb_validation_samples/64)
"""

model_final.load_weights('keras_vgg16_main.h5')


X, y = itr.next()
arr = model_final.evaluate(X,y);
print(arr)

#model1.load_weights('vgg16_1epoch.h5')
layer1_b = 128
layer2_b = 256
layer3_b = 512
layer4_b = 512

layer5_b = 4096
layer6_b = 4096

layer1_a = 128
layer2_a = 256
layer3_a = 512
layer4_a = 512

layer5_a = 4096
layer6_a = 4096

model1 = Sequential()
model1.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#model1.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer1_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer2_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer3_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer4_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model1.add(Flatten())
model1.add(Dense(units=layer5_a,activation="relu"))
model1.add(Dense(units=layer6_a,activation="relu"))
model1.add(Dense(1000))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)

####################### 1st convolution layer with 128 filters
print('1st convolution layer with 128 filters')
A = []
Acc = []

arr = model_final.evaluate(X,y);
print(arr)

filters, biases = model_final.layers[4].get_weights()
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
    for k in range(0, 128):
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
        
        for i in range(0,128):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[4].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,128):
            f = filters[:, :, :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[4].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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

####################### 1st convolution layer with 256 filters
print('1st convolution layer with 256 filters')
A = []
Acc = []

arr = model_final.evaluate(X,y);
print(arr)

filters, biases = model_final.layers[7].get_weights()
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
        
        model_final.layers[7].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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
        
        model_final.layers[7].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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

arr = model_final.evaluate(X,y);
print(arr)

filters, biases = model_final.layers[11].get_weights()
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
        
        model_final.layers[11].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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
        
        model_final.layers[11].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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

####################### 2nd convolution layer with 512 filters
print('2nd convolution layer with 512 filters')
A = []
Acc = []

arr = model_final.evaluate(X,y);
print(arr)

filters, biases = model_final.layers[15].get_weights()
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
        
        model_final.layers[15].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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
        
        model_final.layers[15].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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

####################### 1st dense layer with 4096 filters
print('1st dense layer with 4096 filters')
A = []
Acc = []

arr = model_final.evaluate(X,y);
print(arr)

filters, biases = model_final.layers[20].get_weights()
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
    for k in range(0, 4096):
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
        
        for i in range(0,4096):
            f = filters[ :, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[20].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,4096):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[20].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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


####################### 2nd dense layer with 4096 filters
print('2nd dense layer with 4096 filters')
A = []
Acc = []

arr = model_final.evaluate(X,y);
print(arr)

filters, biases = model_final.layers[21].get_weights()
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
    for k in range(0, 4096):
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
        
        for i in range(0,4096):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(v_trial[i] == 0):
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[21].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
        score_trial = 0.5*arr[1]+0.5*len(v_trial)/np.sum(v_trial)
        
        #score_trial  = cost_func(v_trial)
        for i in range(0,4096):
            f = filters[:, i]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(x_t[i] == 0):
                biases1[i] = 0
                filters1[:, i] = 0
        
        model_final.layers[21].set_weights([filters1, biases1])
        arr = model_final.evaluate(X,y);
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
layer6_a = new_num


model1 = Sequential()
model1.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#model1.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer1_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer2_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer3_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer4_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model1.add(Flatten())
model1.add(Dense(units=layer5_a,activation="relu"))
model1.add(Dense(units=layer6_a,activation="relu"))
model1.add(Dense(1000))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

layerr = model_final.layers[1].get_weights()
model1.layers[0].set_weights(layerr)

model = model_final
######################## 1st convolution layer with 128 filters
filters, biases = model.layers[4].get_weights()
filters1, biases1 = model1.layers[2].get_weights()
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
model1.layers[2].set_weights([filters1, biases1])

######################## 1st convolution layer with 256 filters
filters, biases = model.layers[7].get_weights()
filters1, biases1 = model1.layers[4].get_weights()
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
    if(A2[j] == 1) :
        index2 = 0
        for l in range(128):
            if(A1[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[4].set_weights([filters1, biases1])

######################## 1st convolution layer with 512 filters
filters, biases = model.layers[11].get_weights()
filters1, biases1 = model1.layers[6].get_weights()
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
    if(A3[j] == 1) :
        index2 = 0
        for l in range(256):
            if(A2[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[6].set_weights([filters1, biases1])

######################## 2nd convolution layer with 512 filters
filters, biases = model.layers[15].get_weights()
filters1, biases1 = model1.layers[8].get_weights()
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
    if(A4[j] == 1) :
        index2 = 0
        for l in range(512):
            if(A3[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[8].set_weights([filters1, biases1])

######################## 1st dense layer with 4096 filters
filters, biases = model.layers[20].get_weights()
filters1, biases1 = model1.layers[11].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 4096, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(4096):
    if(A5[j] == 1) :
        index2 = 0
        for l in range(512):
            if(A4[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[11].set_weights([filters1, biases1])

######################## 2nd dense layer with 4096 filters
filters, biases = model.layers[21].get_weights()
filters1, biases1 = model1.layers[12].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 4096, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(4096):
    if(A6[j] == 1) :
        index2 = 0
        for l in range(4096):
            if(A5[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[12].set_weights([filters1, biases1])


arr = model1.evaluate(X,y);
print(arr)

#y_train = y_train.reshape(len(y_train), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_train)

model1.fit_generator(
        training_set,
        samples_per_epoch=nb_train_samples, 
        epochs=10,
        validation_data=test_set,
        validation_steps=nb_validation_samples/64)

#model1.fit(x_train,onehot_encoded,batch_size=64,epochs=100)
model1.summary()
model1.save_weights('VGG16_pruned_weights.h5')

#y_test = y_test.reshape(len(y_test), 1)

#onehot_encoded = onehot_encoder.fit_transform(y_test)
arr = model1.evaluate(X,y);


print(arr)