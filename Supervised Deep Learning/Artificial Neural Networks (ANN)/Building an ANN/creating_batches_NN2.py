
import numpy as np
import pandas as pd  
dataset = pd.read_csv('tf_input_5.txt','\t')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

XX = []
for i in range(len(X)):
    XX.append(np.array(list(X[i])))

X = np.array(XX)

X = X.astype(np.byte)
y = y.astype(np.byte)

from sklearn.model_selection import train_test_split   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim = len(X_train[0]), init = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
# Adding the second hidden layer
#classifier.add(Dense(output_dim = len(X_train[0]), init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 5000, nb_epoch = 20)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
cm = confusion_matrix(y_test, y_pred)  

precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_logistic = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_logistic = confusion_matrix(y_test, y_pred_logistic)



# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svc = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test, y_pred_logistic)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

