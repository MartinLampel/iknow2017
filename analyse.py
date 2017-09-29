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
        c = entries[i].mean(axis=0)
        ax.scatter(c[0], c[1], c[2])
       
        s = entries[i].sum(axis=0)
        ax.scatter(s[0], s[1], s[2])
        plt.show()
        
    
    pca = PCA(n_components=2)
    
    entries, energies = load_data('new_training_set.xml')
    reduced_entries = pca.fit_transform(entries)
    
    kmeans = KMeans(n_clusters=60)
    labels = kmeans.fit_predict(reduced_entries)
    
    plt.figure(1)
    plt.clf()
    
    color = 'bgrcmyk'
    for i in range(60):
        plt.scatter(reduced_entries[labels == i, 0], reduced_entries[labels == i, 1], 
                    color=color[i%7])
    
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    
    plt.show()