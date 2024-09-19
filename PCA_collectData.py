#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:06:01 2024

@author: mohammad-reza.nilchiyan
"""

from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
print (X_iris.shape)
print (y_iris.shape)
print (X_iris[0])

#Imports:
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Get dataset with only the first two attributes
#Selecting the Dataset:
X, y = X_iris[:, :2], y_iris

#Split the dataset into a training and a testing set
#Test set will be the 25% taken randomly
#Splitting the Data:
#This splits the dataset into a training set (75% of the data) and a test set (25% of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

#Standardizing the Features:     
#Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Output Shapes:
print (X_train.shape, y_train.shape)

import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']

# Loop through each class (0, 1, 2 for the iris dataset)
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]  # Sepal length (feature 1)
    ys = X_train[:, 1][y_train == i]  # Sepal width (feature 2)

    # Plot scatter for each class with different colors
    plt.scatter(xs, ys, c=colors[i], label=iris.target_names[i])

# Add legend and labels (outside the loop)
plt.legend()
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# Show the plot
plt.show()
