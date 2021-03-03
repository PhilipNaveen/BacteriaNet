# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:06:22 2021

@author: phjlj
"""

import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as nm

def load(path, classification=None):
    for subdir, dirs, files in os.walk(path):
        for file in files: 
            if (file.endswith("jpg")):
                X_data.append(cv2.imread(subdir + "\\" + file))
                Y_data.append([classification])
            
def resize_data(array, IMG_SIZE=30):
    return_val = []
    for j in array:
        return_val.append(cv2.resize(j, (IMG_SIZE, IMG_SIZE)))
    return return_val

if __name__ == "__main__":
    
    
    X_data, Y_data = [], []
    load(r"C:\Users\phjlj\Documents\Datasets\Bacteria\Spriocheata", classification=0)
    load(r"C:\Users\phjlj\Documents\Datasets\Bacteria\Tuberculosis", classification=1)
    X_data = resize_data(X_data)
    X_data = nm.array(X_data)
    Y_data = nm.array(Y_data)
    
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(130, activation="relu"))
    model.add(tf.keras.layers.Dense(30, activation="relu"))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_data, Y_data, epochs=10)
    print(model.summary())
    model.save("BacteriaNet.h5")
    print(model.summary())
    
    print(len(Y_data))
    
    