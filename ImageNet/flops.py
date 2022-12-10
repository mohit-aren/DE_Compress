# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 19:51:29 2022

@author: Administrator
"""


import tensorflow as tf
import keras

def get_flops(model_h5_path):
    session = tf.Session()
    graph = tf.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = keras.models.load_model(model_h5_path)

            run_meta = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops
#The above function takes the path of a saved model in h5 format. You can save your model and use the function this way:

#model.save('Model.h5'
tf.reset_default_graph()
print(get_flops('Pruned.h5'))