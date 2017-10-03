# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:25:45 2017

@author: Martin Lampel
"""


import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from iknowutil import *


if __name__ == '__main__':    
    entries, _ =  load_data('new_training_set.xml', True)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(entries[0,  :, 0], entries[0,  :, 1], entries[0,  :, 2])
    #plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.scatter(entries[96,  :, 0], entries[96,  :, 1], entries[96,  :, 2], alpha=0.5)
    ax.scatter(entries[97,  :, 0], entries[97,  :, 1], entries[97,  :, 2], c='red', alpha=0.5)
    plt.title('96 97')
    plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(entries[161,  :, 0], entries[161,  :, 1], entries[161,  :, 2], c='orange')
    ax.scatter(entries[162,  :, 0], entries[162,  :, 1], entries[162,  :, 2])
    ax.scatter(entries[163,  :, 0], entries[163,  :, 1], entries[163,  :, 2], c='red')
    plt.title('162 163')
    plt.show()
    
    cg = [-0.00182369, -0.00622189, -0.00363261]
    for i in range(0,5):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(entries[i,  :, 0], entries[i,  :, 1], entries[i,  :, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
    
 