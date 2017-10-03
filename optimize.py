# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 00:24:46 2017

@author: Martin Lampel
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from iknowutil import load_data, create_data_set

def plot_mse_vs_iterations(train_mses, test_mses, n_iterations, hidden_neuron_list):
    """
    Plot the mean squared errors as a function of n_iterations
    :param train_mses: Array of training MSE of shape (len(hidden_neuron_list),n_iterations)
    :param test_mses: Array of testing MSE of shape (len(hidden_neuron_list),n_iterations)
    :param n_iterations: List of number of iterations that produced the above MSEs
    :param hidden_neuron_list: The number of hidden neurons used for the above experiment (Used only for the title of the plot)
    :return:
    """
    plt.figure()
    plt.title("Variation of MSE across iterations".format(hidden_neuron_list))

    color = ['blue','orange','red','green','purple']

    for k_hid,n_hid in enumerate(hidden_neuron_list):
        for data,name,ls in zip([train_mses[k_hid],test_mses[k_hid,]],['Train','Test'],['dashed','solid']):
            plt.plot(range(n_iterations), data, label=name + ' n_h = {}'.format(n_hid), linestyle=ls, color=color[k_hid])

    plt.xlim([0,n_iterations])

    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE")
    plt.minorticks_on()
    plt.show()


def neurons(train_set, train_energies, val_set, val_energies):
   
    neurons = (1,2,3,4,6,8,12,20,30,40,50,80,150)
    
    train_mses = np.empty((0,10))
    val_mses = np.empty((0,10))
    
    for neuron in neurons:       
        mse_train = []
        mse_val = []
        print('neurons: ', neuron)
        for i in range(10):            
            mlpregressor = MLPRegressor(hidden_layer_sizes=(neuron,), 
                                        activation='relu', solver='lbfgs', 
                                        alpha=0, max_iter=200, random_state=i)
            mlpregressor.fit(train_set, train_energies)  
            y_predict_training = mlpregressor.predict(train_set)
            mse = mean_squared_error(train_energies, y_predict_training)
            mse_train.append(mse)
            y_predict_val = mlpregressor.predict(val_set)
            mse = mean_squared_error(val_energies, y_predict_val)
            mse_val.append(mse)
       
        train_mses = np.vstack((train_mses, mse_train))
        val_mses = np.vstack((val_mses, mse_val))
        
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training MSE with number of neurons in the hidden layer")
    
    for data,name,color in zip([train_mses, val_mses],["Training MSE","Validation MSE"],['orange','blue']):
        m = data.mean(axis=1)
        s = data.std(axis=1)

        plt.plot(neurons, m, 'o', linestyle='-', label=name, color=color)
        plt.fill_between(neurons, m-s,m+s,color=color,alpha=.2)

    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("MSE")
    # plt.semilogx()
    plt.legend()
    plt.show()#
    
            
def alpha(train_set, train_energies, val_set, val_energies):
    
    alphas = [10**-8 , 10**-7, 10**-6 , 10**-5 , 10**-4, 10**-3 , 10**-2 , 10**-1, 1, 10, 100]
    
    neuron = 200
    train_mses = np.empty((0,10))
    val_mses = np.empty((0,10))
    
    for alpha in alphas:
        mse_train = []
        mse_val = []
        for i in range(10):            
            mlpregressor = MLPRegressor(hidden_layer_sizes=(neuron,), 
                                        activation='relu', solver='lbfgs', 
                                        alpha=alpha, max_iter=200, random_state=i)
            mlpregressor.fit(train_set, train_energies)  
            y_predict_training = mlpregressor.predict(train_set)
            mse = mean_squared_error(train_energies, y_predict_training)
            mse_train.append(mse)
            y_predict_val = mlpregressor.predict(val_set)
            mse = mean_squared_error(val_energies, y_predict_val)
            mse_val.append(mse)
       
        train_mses = np.vstack((train_mses, mse_train))
        val_mses = np.vstack((val_mses, mse_val))
    
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training MSE with regularization parameter")

    for data,name,color in zip([train_mses,val_mses],["Training MSE","Validation MSE"],['orange','blue']):
        m = data.mean(axis=1)
        s = data.std(axis=1)

        plt.plot(alphas, m, 'o', linestyle='-', label=name, color=color)
        plt.fill_between(alphas, m-s,m+s,color=color,alpha=.2)

    plt.semilogx()
    plt.xlabel("Alphas")
    plt.ylabel("MSE")
    # plt.semilogx()
    plt.legend()
    plt.show()
    
def optimal_solver(train_set, train_energies, val_set, val_energies):

    neurons = (2,8,20)
    iterations = 30
    
    solvers = ['lbfgs', 'sgd', 'adam']
    train_mse_result = {}
    test_mse_result = {}
    
    for solver in solvers:       
          
        train_mses = np.empty((0,iterations))
        test_mses = np.empty((0,iterations))
        
        for neuron in neurons:

            regressor = MLPRegressor(hidden_layer_sizes=(neuron, ), 
                                     activation='relu', solver=solver, 
                                     alpha=0, max_iter=1, warm_start=True)
            mse_train= []
            mse_val = []
            for i in range(iterations):
                regressor.fit(train_set, train_energies)   
                y_predict_training = regressor.predict(train_set)
                mse = mean_squared_error(train_energies, y_predict_training)
                mse_train.append(mse)
                y_predict_val = regressor.predict(val_set)
                mse = mean_squared_error(val_energies, y_predict_val)
                mse_val.append(mse)
                
            train_mses = np.vstack((train_mses, np.array(mse_train)))
            test_mses = np.vstack((test_mses, np.array(mse_val)))
            
        train_mse_result[solver] = train_mses
        test_mse_result[solver] = test_mses
                       
    plot_mse_vs_iterations(train_mse_result['lbfgs'], test_mse_result['lbfgs'], 
                            iterations, neurons)
    plot_mse_vs_iterations(train_mse_result['sgd'], test_mse_result['sgd'], 
                           iterations, neurons)
    plot_mse_vs_iterations(train_mse_result['adam'], test_mse_result['adam'], 
                           iterations, neurons)

def optimal_seed(train_set, train_energies, val_set, val_energies):
    
    alpha = 10**-3
    solver = "lbfgs"
    neurons = 200    
    
    train_mse_values = []
    val_mse_values = []
    
    for i in range(20):
        mlpregressor = MLPRegressor(hidden_layer_sizes=(neurons, ), 
                                    activation='relu', solver=solver, 
                                   alpha=alpha, max_iter=1, random_state=i,
                                   warm_start=True)
        
        mlpregressor.fit(train_set, train_energies)
        y_predict_training = mlpregressor.predict(train_set)
        y_predict_val = mlpregressor.predict(val_set)
        min_train_mse = mean_squared_error(train_energies, y_predict_training)
        min_val_mse = mean_squared_error(val_energies, y_predict_val)
        
        print("seed: ", i)
        
        while True:
            mlpregressor.fit(train_set, train_energies)
            y_predict_training = mlpregressor.predict(train_set)
            y_predict_val = mlpregressor.predict(val_set)
            new_train_mse = mean_squared_error(train_energies, y_predict_training)
            new_val_mse = mean_squared_error(val_energies, y_predict_val)
            
            if new_val_mse > min_val_mse:
                train_mse_values.append(min_train_mse)
                val_mse_values.append(min_val_mse)
                break
            else:
                min_train_mse = new_train_mse
                min_val_mse = new_val_mse
    
        
    
    seed = np.argmin(val_mse_values)
    print("training mean: ", np.mean(train_mse_values), 
          " training standard deviation: ", np.std(train_mse_values),
          "best mse: ", train_mse_values[np.argmin(train_mse_values)])
    print("validation mean: ", np.mean(val_mse_values), 
          " validation standard deviation: ", np.std(val_mse_values),
          "best mse: ", val_mse_values[seed])
    print("best seed: ", seed)
     
cluster = 65

if __name__ == '__main__':
    
    train_entries, energies = load_data('new_training_set.xml')
    test_entries, _ = load_data('new_validation_set.xml')
    
    train_set = create_data_set(train_entries)
    test_set = create_data_set(test_entries)
    
    train_set, val_set, train_energy, val_energy = train_test_split(train_set, 
                                        energies, test_size=0.25, random_state=0) 
    
    neurons(train_set, train_energy, val_set, val_energy)
    alpha(train_set, train_energy, val_set, val_energy)
    optimal_solver(train_set, train_energy, val_set, val_energy)
    optimal_seed(train_set, train_energy, val_set, val_energy)
    