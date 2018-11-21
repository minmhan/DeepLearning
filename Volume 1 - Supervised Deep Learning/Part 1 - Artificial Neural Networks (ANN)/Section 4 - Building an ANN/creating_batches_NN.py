
import numpy as np
import pandas as pd  
dataset = pd.read_csv('tf_input_4.txt','\t')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

y = y.astype(np.bool)
XX = []
for i in range(len(X)):
    XX.append(np.array(list(X[i])))

X = np.array(XX) 

X = X.astype(np.float16)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
indices = (np.nonzero(y_train==1)[0]).tolist()
        
def chunkify(lst,n):
    return [lst[i::n] for i in range(0,n)] 
    
indices2 = chunkify(indices,5)

indices3 = [i for i in range(len(y_train)) if i not in indices]
indices4 = chunkify(indices3,36)

batch_size = len(indices2[0])+len(indices4[0])

all_indices = []
my_dict = {}
for j in range(0,len(indices2)):
    seq = [i for i in range(j,len(indices4),len(indices2))]
    my_dict.update({j:seq})

for key in my_dict.keys():
    for v in my_dict[key]:
        vals = indices2[key]+indices4[v]
        all_indices.append(vals)
flat_list = [item for sublist in all_indices for item in sublist]
X_new = X_train[flat_list,] 
y_new = y_train[flat_list]   


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units = len(X_new[0]), init = 'uniform', activation = 'relu', input_dim = len(X_new[0])))
#classifier.add(Dropout(0.2))
# Adding the second hidden layer
#classifier.add(Dense(units = len(X_new[0]), init = 'uniform', activation = 'relu'))
#classifier.add(Dropout(0.2))
# Adding the output layer
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_new, y_new, batch_size = batch_size, epochs = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  

# Logistic
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_new, y_new)

# Predicting the Test set results
y_pred_logistic = classifier2.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_logistic = confusion_matrix(y_test, y_pred_logistic)


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)

# Predicting the Test set results
y_pred_randomforest = classifier3.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_randomforest = confusion_matrix(y_test, y_pred_randomforest)  


