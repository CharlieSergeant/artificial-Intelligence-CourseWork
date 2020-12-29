# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:47:53 2019

@author: CDS
"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.preprocessing import LabelEncoder

from info_gain import info_gain
#Import scikit-learn metrics module for accuracy calculation

col_names = ['Y','A1','A2','A3','A4']
# load dataset
pima = pd.read_csv("C:/Users/CDS/Desktop/FinalData.csv", header=0,names=col_names)
pimaObj = pima.iloc[:,:].values

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


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion = 'entropy')

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

predicted = clf.predict([[-1,-1,1,1]]) 
print(predicted)

#Predict the response for test dataset
#y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('C:/Users/CDS/Desktop/HW1.png')
Image(graph.create_png())