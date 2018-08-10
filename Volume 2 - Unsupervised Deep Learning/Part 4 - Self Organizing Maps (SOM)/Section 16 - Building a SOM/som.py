# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
'''
DataSet - http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
There are 6 numerical and 8 categorical attributes. The labels have been changed for the convenience of the statistical algorithms. 
For example, attribute 4 originally had 3 labels p,g,gg and these have been changed to labels 1,2,3. 
A1: 0,1 CATEGORICAL (formerly: a,b) 
A2: continuous. 
A3: continuous. 
A4: 1,2,3 CATEGORICAL (formerly: p,g,gg) 
A5: 1, 2,3,4,5, 6,7,8,9,10,11,12,13,14 CATEGORICAL (formerly: ff,d,i,k,j,aa,m,c,w, e, q, r,cc, x) 
A6: 1, 2,3, 4,5,6,7,8,9 CATEGORICAL (formerly: ff,dd,j,bb,v,n,o,h,z) 
A7: continuous. 
A8: 1, 0 CATEGORICAL (formerly: t, f) 
A9: 1, 0	CATEGORICAL (formerly: t, f) 
A10: continuous. 
A11: 1, 0	CATEGORICAL (formerly t, f) 
A12: 1, 2, 3 CATEGORICAL (formerly: s, g, p) 
A13: continuous. 
A14: continuous. 
A15: 1,2 class attribute (formerly: +,-)'''

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)