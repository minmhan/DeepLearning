import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pycorenlp import StanfordCoreNLP
import json
import numpy as np


with open('C:/Users/Ameya/Documents/My Received Files/tf_input.txt','rb') as file:
    data = file.readlines()    
x = []
y = []
for d in data:
    d1 = str(d).split('\\t')
    d2 = d1[0]
    x.append(d2)
    d3 = d1[1]
    y.append(d3)

new_x = []    
for k in x:
    temp = ','.join(str(k))
    temp2 = temp.split(',')
    temp2 = temp2[2:]
    temp2 = [np.float(l) for l in temp2]
    new_x.append(temp2)

x = np.array(new_x)
y = [np.float16(yi) for yi in y]
from sklearn.model_selection import train_test_split   

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim = len(X_train[0]), init = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
# Adding the second hidden layer
classifier.add(Dense(output_dim = len(X_train[0]), init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 5000, nb_epoch = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  