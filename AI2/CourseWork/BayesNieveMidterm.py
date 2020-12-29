# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:47:53 2019

@author: CDS
"""

# Load libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz

from info_gain import info_gain

from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
#Import scikit-learn metrics module for accuracy calculation

col_names = ['Y','A1','A2','A3','A4']
# load dataset
pima = pd.read_csv("C:/Users/CDS/Desktop/FinalData.csv", header=0,names=col_names)
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

X_train,X_test, y_train, y_test = train_test_split(X, y,train_size = 6,shuffle = False) # 70% training and 30% test


ig = info_gain.info_gain(X,y)

print("Information gain: ",ig)

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
model = gnb.fit(X_train, np.ravel(y_train))

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

print (gnb.predict([[-1,-1,1,1]]))


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
