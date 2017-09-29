# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:27:45 2017

@author: Martin Lampel
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network.multilayer_perceptron import MLPRegressor

import matplotlib.pyplot as plt

from iknowutil import load_data, create_data_set

neurons = 40
alpha = 10**-3
seed = 6

if __name__ == '__main__':

    train_entries, energies = load_data('new_training_set.xml')
    test_entries, _ = load_data('new_validation_set.xml')
  
    reduced_set = create_data_set(train_entries)
    test_set = create_data_set(test_entries)
    
    train_set, val_set, train_energy, val_energy = train_test_split(reduced_set, 
                                        energies, test_size=0.25, random_state=0) 
    
    mlpregressor = MLPRegressor(hidden_layer_sizes=(neurons,), activation='relu',
                                    solver='lbfgs', alpha=alpha, 
                                    max_iter=1000, random_state=seed)
    mlpregressor.fit(train_set, train_energy)  

    y_predict_training = mlpregressor.predict(train_set)
    y_predict_val = mlpregressor.predict(val_set)
    y_predict_set = mlpregressor.predict(test_set)
    
    mse_train = (train_energy - y_predict_training)**2/2
    mse_val = (val_energy - y_predict_val)**2/2
    print(mean_squared_error(train_energy, y_predict_training), 
          mean_squared_error(val_energy, y_predict_val))
    print(y_predict_set)
              
    plt.figure()
    plt.plot(train_energy, 'ro')
    plt.plot(y_predict_training, 'bx')
    plt.show()
    plt.figure()
    plt.plot(val_energy, 'go')
    plt.plot(y_predict_val, 'kx')
    plt.show()
    
    plt.figure()
    plt.plot(mse_train, 'ro')
    plt.plot(mse_val, 'go')
    plt.show()
    
    