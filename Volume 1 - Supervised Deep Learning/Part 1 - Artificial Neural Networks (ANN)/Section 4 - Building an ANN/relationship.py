# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:29:12 2018

@author: Min Han
"""
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from pycorenlp import StanfordCoreNLP
import json
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('tf_input.txt','\t')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

XX = []
for i in range(len(X)):
    XX.append(np.array(list(X[i])))

X = np.array(XX)

from sklearn.model_selection import train_test_split   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units = len(X_train[0]), init = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
classifier.add(Dropout(0.2))
# Adding the second hidden layer
#classifier.add(Dense(units = len(X_train[0]), init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 5000, epochs = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  

#--------------------------------------------------------------------------------
#Naive Bayes Classifier
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
X_train_int = X_train.astype(np.int)
y_train_int = y_train.astype(np.int)
classifier.fit(X_train_int, y_train_int)

# Predicting the Test set results
X_test_int = X_test.astype(np.int)
y_pred_bayes = classifier.predict(X_test_int)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_bayes = confusion_matrix(y_test.astype(np.int), y_pred_bayes)

# --------------------------------------------------------------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_int, y_train_int)

# Predicting the Test set results
y_pred_logistic = classifier.predict(X_test_int)
from sklearn.metrics import confusion_matrix
cm_logistic = confusion_matrix(y_test.astype(np.int), y_pred_logistic)



