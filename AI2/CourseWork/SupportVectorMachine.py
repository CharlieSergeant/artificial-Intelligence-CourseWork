# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:17:37 2019

@author: CDS
"""

# Load libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#Import scikit-learn metrics module for accuracy calculation

col_names = ['Y','A1','A2','A3','A4']
# load dataset
pima = pd.read_csv("C:/Users/CDS/Desktop/midterm.csv", header=0,names=col_names)
pimaObj = pima.iloc[:,:].values
labelEncoder = LabelEncoder()
col = 0
for i,col in enumerate(col_names,start = 0):
    pimaObj[:,i] = labelEncoder.fit_transform(pimaObj[:,i])

#split dataset in features and target variable
feature_cols = ['A1','A2','A3','A4']
X = pima[feature_cols] # Features
target = ['Y']
y = pima[target] # Target variable

X = X.astype('int')
y = y.astype('int')


#Create a svm Classifier
clf = svm.SVC(kernel='linear',C=1000) # Linear Kernel

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#Train the model using the training sets
clf.fit(X_std,y)

conf = clf.score(X_std,y)
print (conf)
 

print('weights: ')
print(clf.coef_)
print('Intercept: ')
print(clf.intercept_)
	 # Put the result into a color plot
