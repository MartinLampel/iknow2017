# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:44:12 2017

@author: Martin Lampel
"""

import xml.etree.ElementTree as ET
import numpy as np

from sklearn.cluster import KMeans

def load_data(path, separeted=False):
    
    tree = ET.parse(path)
    root = tree.getroot()

    if separeted:
        entries = np.array([]).reshape(0,60,3)#
    else:
        entries = np.array([]).reshape(0,3)
    
    for coordinates in root.iter('coordinates'):
        coordinates_array = np.array([]).reshape(0, 3)
        for c in coordinates:
            values = np.array([float(x) for x in c.text.split(' ')])
            if separeted:
                coordinates_array = np.vstack((coordinates_array, values))
            else:
                entries = np.vstack((entries, values))
        if separeted:
            entries = np.vstack((entries, coordinates_array[np.newaxis, :]))
            
    energies = np.array([float(energy.text) for energy in root.iter('energy')])

    return entries, energies

def preprocess_data(train_entries, test_entries, energies):
    pass

def create_data_set(reduced_entries, no_atoms=60):
    
    data_set = np.array([]).reshape(0, no_atoms)
    
    for i in range(reduced_entries.shape[0] // no_atoms):
        entry = reduced_entries[i*no_atoms:(i+1)*no_atoms]
        d = np.sqrt(np.sum((entry)**2, axis=1))
        data_set = np.vstack((data_set, d))
    
    return data_set
