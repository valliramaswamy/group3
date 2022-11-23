#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:17:29 2022

@author: adil
"""
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datapath = "/Users/adil/Documents/INTRO TO AI COURSEWORK/dataset modified/Clean_Dataset_group3(classifier).csv"
# this will read the dataset as a dataframe
flight = pd.read_csv(datapath)

# print (flight.shape) 

# log price is target variable Y for now (for now but need to change to another column that has two classifications using median below)
print(flight.log_price) 
# this will give you 50% value which is median = 4.358886  that I used to create another column in dataset (price_classifier)
print(flight.log_price.describe())

Y = flight.price_classifier
print(Y)

# 19th line will give you 50% value which is median
# need to find way of classifiying the data with the median
# median = 

# splitting the data
flight_train, flight_test, Y_train, Y_test = train_test_split(flight, Y, test_size=0.2, random_state=41)

# row numbers showing a split of 80 to 20 when printed compared to just printing flight.shape
print (flight_train.shape)
print (flight_test.shape)
print (Y_train.shape)
print (Y_test.shape)

# will adjust training parameters 
percep = Perceptron(max_iter = 40, tol=0.001, eta0=1)

# ERROR IN CONSOLE STARTS FROM HERE: 
    
# training perceptron
percep.fit(flight_train,Y_train)

# make predication
Y_pred = percep.predict(flight_test)

# evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(Y_test, Y_pred))
