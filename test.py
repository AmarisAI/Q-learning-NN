# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 04:02:26 2015

@author: ka
"""
import lasagne


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
# Prepare Theano variables for inputs and targets
input_var = T.matrix('inputs')
target_var = T.icol('targets')
# Create neural network model
network = build_mlp(input_var)
                                         
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()                             
                                         
                                         
                                         
                                         
                                         
                                         
                                         