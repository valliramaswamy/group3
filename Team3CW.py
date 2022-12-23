# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import base64
import os
import platform
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model,metrics
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn import datasets
from sklearn.linear_model import Perceptron
import math
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Faiza's code
#Initial code cleanup
datapath = "/Users/valliramaswamy/Documents/MSci Data Science/Year-3/2IntrotoAI/group3CW/Clean_Dataset.csv"
# this will read the dataset as a dataframe i.e. flight is the dataframe 
df = pd.read_csv(datapath)
flight = pd.read_csv(datapath)

#Sorting values by airline
df = df.sort_values(by='airline', ascending=True)

#Checking overall and asking for sum of null values
df.isnull().values.all().sum()

# Dropping Flight column
df.drop('flight', 1, inplace=True)

# Changing price to Pounds
df["price"] = (df["price"]*0.011).round(2)

# Printinf dataset
print(df)

plt.boxplot(df['price'])
plt.title('Box plot distribution of "price"')
plt.show()

# Valli's code
# Feature Engineering: removing outliers

# Creates a new column in the dataframe named 'price outlier'
df['price_outlier'] = 0

#to find the mean and standard deviation of the price values to work out the outlier
price_mean = np.mean(df['price'])
print(price_mean)
price_std = np.std(df['price'])
print(price_std)

#calculation to assign 0 or 1 to the price values (0 if the datapoint is not an outlier & 1 if it is)
df.loc[abs(df['price'] - price_mean) > 2 * price_std,'price_outlier'] = 1

#This counts the number of unique outlier values
print(Counter(df['price_outlier']))

df = df[df.price_outlier != 1]
print (df.shape)

# Dropping 'price outlier' column
df.drop('price_outlier', 1, inplace=True)

#Aakash's code - transforms the dataset with categorical values to numerical values

le = LabelEncoder()

df['airline'] = le.fit_transform(df['airline'])
df['source_city'] = le.fit_transform(df['source_city'])
df['departure_time'] = le.fit_transform(df['departure_time'])
df['arrival_time'] = le.fit_transform(df['arrival_time'])
df['destination_city'] = le.fit_transform(df['destination_city'])
df['class'] = le.fit_transform(df['class'])

# Find unique values within the stops column
print(list(set(df['stops'])))
# Match and replace the numerical values in text with integers
df['stops'] = df['stops'].replace(["zero", "one", "two_or_more"], [0, 1, 2])

print(df)

# Correlation Heatmap - Initial Investigation
# Plotting a correlation heatmap of the features

plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot = True, fmt = ".2f", linewidth = .7, cmap="rocket_r")
plt.title("Relationship Between Features")
plt.show()

# Defines X & y values for modelling 
result = []

for x in flight.columns:
    if x != 'price':
        result.append(x)


X = flight[result].values
y = flight['price'].values

# Linear Regression---------------------

model = LinearRegression()

# Using 5 k-fold cross validation

kf = KFold(5, shuffle = True, random_state = 44)

fold = 1

for train_index, validate_index in kf.split(X,y):
    model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    fold += 1
    
print(model.coef_)

# Regression plot

plt.scatter(y_pred[:2000], y_test[:2000])
plt.title ("Linear Regression Model")
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x = y_test, y = y_pred, scatter = False)

originalDataset_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
originalDataset_head = originalDataset_compare.head(25)

print(originalDataset_head)

print('Mean:', np.mean(y_test))
print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))
print('Correlation: ', stats.pearsonr(y_test,y_pred))

# MLP Regressor--------------------
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

plt.scatter(y_pred[:2000], y_test[:2000])
plt.title('MLP Model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x = y_test, y = y_pred, scatter = False)

plt.show()

# kNN Regressor-------------------------------

# Uses 5-fold split for 1 neighbour
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    knn_model = KNeighborsRegressor(n_neighbors=1)
    knn_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = knn_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 2 neighbours
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    knn_model = KNeighborsRegressor(n_neighbors=2)
    knn_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = knn_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 3 neighbours
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = knn_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 4 neighbours
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    knn_model = KNeighborsRegressor(n_neighbors=4)
    knn_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = knn_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 5 neighbours
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = knn_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 6 neighbours
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    knn_model = KNeighborsRegressor(n_neighbors=6)
    knn_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = knn_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
#Applying GridsearchCV to determine the optimal number of neighbours

parameters = {"n_neighbors": range(1, 5)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X, y)
gridsearch.best_params_

#Plotting actual VS predicted price values to display its regression line

plt.scatter(y_pred[:2000], y_test[:2000])
plt.title('kNN Model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x = y_test, y = y_pred, scatter = False)

# Decision Tree Regressor---------------------------

#Model using Friedman_mse criterion
decision_tree = DecisionTreeRegressor(criterion='friedman_mse',splitter='random')

# Use 5-fold split
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    decision_tree.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = decision_tree.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
plt.scatter(y_test[:100], y_pred[:100])
#plt.plot(y_pred, y_test, color = 'green')
plt.xlabel('Actual')
plt.ylabel('Predicted')

sns.regplot(x= y_test, y= y_pred, scatter = False)

plt.show()

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)


#Model using default criterion
decision_tree = DecisionTreeRegressor()

# Use 5-fold split
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    decision_tree.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = decision_tree.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
plt.scatter(y_test[:100], y_pred[:100])
#plt.plot(y_pred, y_test, color = 'green')
plt.xlabel('Actual')
plt.ylabel('Predicted')

sns.regplot(x= y_test, y= y_pred, scatter = False)

plt.show()

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)

# Randon Forest Regressor--------------------

# Uses 5-fold split for 50 estimators
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    randomforest_model = RandomForestRegressor(n_estimators = 50, random_state = 0)
    randomforest_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = randomforest_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 100 estimators
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    randomforest_model = RandomForestRegressor(n_estimators = 100, random_state = 0)
    randomforest_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = randomforest_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 125 estimators
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    randomforest_model = RandomForestRegressor(n_estimators = 125, random_state = 0)
    randomforest_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = randomforest_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
# Uses 5-fold split for 150 estimators
kf = KFold(5,shuffle=True)

fold = 1

for train_index, validate_index in kf.split(X,y):
    randomforest_model = RandomForestRegressor(n_estimators = 150, random_state = 0)
    randomforest_model.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = randomforest_model.predict(X[validate_index])
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    
    fold += 1
    
#Plotting actual VS predicted price values to display its regression line

plt.scatter(y_pred[:2000], y_test[:2000])
plt.title('Random Forest Model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x = y_test, y = y_pred, scatter = False)

# Neural Networks Model AVG COMPUTATION TIME: 6 HOURS----------------

# Source: Intro to AI, Session 6 (City, University of London)
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# Source: Intro to AI, Session 6 (City, University of London)
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    if target_type in (np.int64, np.int32):
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

# Source: Intro to AI, Session 6 (City, University of London)
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)

path = "."

filename_read = os.path.join(path, "Clean_Dataset.csv")
df = pd.read_csv(filename_read, index_col = 0)

df.isna().sum()

df.head()

flights = df['flight']
df.drop('flight', 1, inplace = True)

# Converting the price from Indian rupees to GBP
df['price'] = (df["price"]*0.011).round(2)

df.head()

# Label encoding to ensure the dataset does not contain any string values
encode_text_index(df, "airline")
encode_text_index(df, "source_city")
encode_text_index(df, "departure_time")
encode_text_index(df, "arrival_time")
encode_text_index(df, "destination_city")
encode_text_index(df, "class")

# Match and replace the numerical values in text with integers
df['stops'] = df['stops'].replace(["zero", "one", "two_or_more"], [0, 1, 2])
df.head()

X,y = to_xy(df, "price")

print(y.shape)
print(y[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = Sequential()
model.add(Dense(4, input_shape=X[1].shape, activation='sigmoid')) # Hidden layer
model.add(Dense(1)) # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=10)
model.summary()

# Predict the flight prices
pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])

# Return the RMSE score of the model
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Score (RMSE): {score}")

model = Sequential()
model.add(Dense(4, input_dim=X.shape[1], activation='sigmoid')) # Hidden layer
model.add(Dense(1)) # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=200)

# Predict the flight prices using the fine tuned model
pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])

# Return the RMSE score of the model
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Score (RMSE): {score}")

model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='sigmoid')) # Hidden layer 1
model.add(Dense(1)) # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=200)

# Predict the flight prices using the fine tuned model
pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])

# Return the RMSE score of the model
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Score (RMSE): {score}")

model = Sequential()
model.add(Dense(1024, input_dim=X.shape[1], activation='sigmoid')) # Hidden layer
model.add(Dense(1)) # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=200)

# Predict the flight prices using the fine tuned model
pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])

# Return the RMSE score of the model
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Score (RMSE): {score}")

model = Sequential()
model.add(Dense(1024, input_dim=X.shape[1], activation='relu')) # Hidden layer
model.add(Dense(1)) # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=200)

# Predict the flight prices using the fine tuned model
pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])

# Return the RMSE score of the model
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Score (RMSE): {score}")

model = Sequential()
model.add(Dense(1024, input_dim=X.shape[1], activation='relu')) # Hidden layer one
model.add(Dense(1024,activation='relu')) # Hidden layer two
model.add(Dense(1)) # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=200)

# Predict the flight prices using the fine tuned model
pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])

# Return the RMSE score of the model
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Score (RMSE): {score}")

# Return the final RMSE score of the model
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Final score (RMSE): {score}")

# Predicted prices
for i in range(20):
    print(f"{i+1}. Flight: {flights[i]}, Price: {y[i]}, predicted price: {pred[i]}")
    
print("Dataset size: {}".format(len(df)))
remove_outliers(df,'price',2) 
print("Dataset size {}".format(len(df)))

sc = StandardScaler()
sc.fit(X_train)
X_train= sc.transform(X_train)
X_test = sc.transform(X_test)

# Scatterplot of actual vs predicted prices
plt.scatter(pred[:1000], y_test[:1000])
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x = y_test, y = pred, scatter = False)

# Visualisations to answer objectives and questions-------------------

#Does price vary with airlines? 

# 1.Spicejet
# 2.AirAsia
# 3.Vistara
# 4.GO_First
# 5.Indigo
# 6.AirIndia

plt.figure(figsize = (8,5))
sns.boxplot(data=df, x="airline", y="price")
plt.title("Airlines with Prices")
plt.show()

#How does the ticket price vary with the number of stops of a flight?
plt.figure(figsize = (8,5))
sns.boxplot(data=df, x="stops", y="price")
plt.title("Stops with Prices")
plt.show()

correlation_coeff = df.corr()
print(correlation_coeff["price"])