#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:17:29 2022

@author: adil
"""
import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPRegressor
from sklearn import metrics



datapath = "/Users/adil/Documents/INTRO TO AI COURSEWORK/Clean_Dataset.csv"
# this will read the dataset as a dataframe i.e. flight is the dataframe 
flight = pd.read_csv(datapath)

# FAIZA:
#Checking for null values in airline
flight['airline'].isnull().sum()

#Checking for null values in source city
flight['source_city'].isnull().sum()

#Checking for null values in departure time
flight['departure_time'].isnull().sum()

#Checking for null values in stops
flight['stops'].isnull().sum()

#Checking for null values in arrival_time
flight['arrival_time'].isnull().sum()

#Checking for null values in destination city
flight['destination_city'].isnull().sum()

#Checking for null values in class
flight['class'].isnull().sum()

#Checking for null values in duration
flight['duration'].isnull().sum()
 
#Checking for null values in days left
flight['days_left'].isnull().sum()

#Checking for null values in price
flight['price'].isnull().sum()

#Checking overall and asking for sum of null values
flight.isnull().values.all().sum()

# Dropping Flight column. Not in perceptron as not in that column. 
flight.drop('flight', 1, inplace=True)

# Changing price to Pounds
flight["price"] = (flight["price"]*0.011).round(2)

# VALLI:
# Feature Engineering: removing outliers & adding new logged price column
plt.boxplot(flight['price'])
plt.show()
print(flight.isnull().any())

# Creates a new column in the dataframe named 'price outlier'
flight['price_outlier'] = 0

#to find the mean and standard deviation of the price values to work out the outlier
price_mean = np.mean(flight['price'])
print(price_mean)
price_std = np.std(flight['price'])
print(price_std)

#calculation to assign 0 or 1 to the price values (0 if the datapoint is not an outlier & 1 if it is)
flight.loc[abs(flight['price'] - price_mean) > 2 * price_std,'price_outlier'] = 1

#This counts the number of unique outlier values
print(Counter(flight['price_outlier']))

flight = flight[flight.price_outlier != 1]
print (flight.shape) 

# Dropping 'price outlier' column
flight.drop('price_outlier', 1, inplace=True)


# AAKASH:
le = LabelEncoder()

flight['airline'] = le.fit_transform(flight['airline'])
flight['source_city'] = le.fit_transform(flight['source_city'])
flight['departure_time'] = le.fit_transform(flight['departure_time'])
flight['arrival_time'] = le.fit_transform(flight['arrival_time'])
flight['destination_city'] = le.fit_transform(flight['destination_city'])
flight['class'] = le.fit_transform(flight['class'])



# Find unique values within the stops column
print(list(set(flight['stops'])))

# Match and replace the numerical values in text with integers
flight['stops'] = flight['stops'].replace(["zero", "one", "two_or_more"], [0, 1, 2])

print (flight.head())

# ABOVE WAS PRE PROCESSING

# Commented lines right below applies only to classifier version of perceptron.
# flight.price_classifier was created using a seperate table with extra column under adil branch 
# and its value is either "high" or "low" relative to the median of the log price column = 4.358886.
# Median of log price was retrieved using python code: print(flight.log_price.describe())

print(flight)
print(flight.price)

# target variable
Y = flight.price
print(Y)

# splitting the data. differing random states widly affects the accuracy of the model. can also be None
# below code irrelevant as using k fold now.
flight_train, flight_test, Y_train, Y_test = train_test_split(flight, Y, test_size=0.2, random_state=55)
  
# MLPpercep.fit(flight_train,Y_train). We are using k fold instead
# evaluate accuracy (Only compatabile with classifer version of perceptron)

result = []


for x in flight.columns:
    if x != 'price':
        result.append(x)


X = flight[result].values
y = flight['price'].values

MLPpercep = MLPRegressor(hidden_layer_sizes=(100))

kf = KFold(5,shuffle=True)
fold = 1
for train_index, validate_index in kf.split(X,y):
    MLPpercep.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = MLPpercep.predict(X[validate_index])

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1
    
    
MLPpercep = MLPRegressor(hidden_layer_sizes=(150))

kf = KFold(5,shuffle=True)
fold = 1
for train_index, validate_index in kf.split(X,y):
    MLPpercep.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = MLPpercep.predict(X[validate_index])

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1
    
MLPpercep = MLPRegressor(hidden_layer_sizes=(250))

kf = KFold(5,shuffle=True)
fold = 1
for train_index, validate_index in kf.split(X,y):
    MLPpercep.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = MLPpercep.predict(X[validate_index])

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1
    
MLPpercep = MLPRegressor(hidden_layer_sizes=(400))

kf = KFold(5,shuffle=True)
fold = 1
for train_index, validate_index in kf.split(X,y):
    MLPpercep.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = MLPpercep.predict(X[validate_index])

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1
    
    
MLPpercep = MLPRegressor(hidden_layer_sizes=(750))

kf = KFold(5,shuffle=True)
fold = 1
for train_index, validate_index in kf.split(X,y):
    MLPpercep.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = MLPpercep.predict(X[validate_index])

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1

    
MLPpercep = MLPRegressor(hidden_layer_sizes=(150, 100))

kf = KFold(5,shuffle=True)
fold = 1
for train_index, validate_index in kf.split(X,y):
    MLPpercep.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = MLPpercep.predict(X[validate_index])

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1
    
    
MLPpercep = MLPRegressor(hidden_layer_sizes=(150, 100, 50))

kf = KFold(5,shuffle=True)
fold = 1
for train_index, validate_index in kf.split(X,y):
    MLPpercep.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = MLPpercep.predict(X[validate_index])

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1



plt.scatter(y_pred[:2000], y_test[:2000])
plt.title('MLP Model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x = y_test, y = y_pred, scatter = False)

plt.show()
