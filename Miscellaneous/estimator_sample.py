# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:52:17 2018

@author: Min Han
https://medium.com/learning-machine-learning/introduction-to-tensorflow-estimators-part-1-39f9eb666bc7
Tensorflow Estimator Sample (Regression)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plt.style.use("seaborn-colorblind")
%matplotlib inline
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.FATAL)

used_features = ['property_type','room_type','bathrooms','bedrooms','beds','bed_type','accommodates','host_total_listings_count'
                ,'number_of_reviews','review_scores_value','neighbourhood_cleansed','cleaning_fee','minimum_nights','security_deposit','host_is_superhost',
                 'instant_bookable', 'price']
boston = pd.read_csv('data/listings.csv', usecols = used_features)
print(boston.shape)
boston.head(2)

# Clean
for feature in ["cleaning_fee", "security_deposit", "price"]:
    boston[feature] = boston[feature].map(lambda x: x.replace("$","").replace(",",""), na_action="ignore")
    boston[feature] = boston[feature].astype(float)
    boston[feature].fillna(boston[feature].median(),inplace = True)
    
for feature in ["bathrooms","bedrooms","beds","review_scores_value"]:
    boston[feature].fillna(boston[feature].median(),inplace = True)
    
boston['property_type'].fillna('Apartment',inplace = True)

boston["price"].plot(kind = 'hist',grid = True)
plt.title("Price histogram before subsetting and log-transformation")

boston['price'].skew()

boston = boston[(boston["price"]>50)&(boston["price"]<500)]
target = np.log(boston.price)
target.hist()
plt.title("Price distribution after the subsetting and log-transformation")

features = boston.drop('price',axis=1)
features.head()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

numeric_columns = ['host_total_listings_count','accommodates','bathrooms','bedrooms','beds',
                   'security_deposit','cleaning_fee','minimum_nights','number_of_reviews',
                   'review_scores_value']

# Get all the categorical feature names that contains strings
categorical_columns = ['host_is_superhost','neighbourhood_cleansed','property_type','room_type','bed_type','instant_bookable']

numeric_features = [tf.feature_column.numeric_column(key = column) for column in numeric_columns]
print(numeric_features[0])


categorical_features = [tf.feature_column.categorical_column_with_vocabulary_list(key = column, vocabulary_list = features[column].unique()) 
                            for column in categorical_columns]
print(categorical_features[3])

linear_features = numeric_features + categorical_features

# Create training input function
training_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train,
                                                        y=y_train,
                                                        batch_size=32,
                                                        shuffle= True,
                                                        num_epochs = None)

# create testing input function
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                    y=y_test,
                                                    batch_size=32,
                                                    shuffle=False,
                                                    num_epochs = 1)

linear_regressor = tf.estimator.LinearRegressor(feature_columns=linear_features,
                                                model_dir = "linear_regressor")

linear_regressor.train(input_fn = training_input_fn, steps=2000)

linear_regressor.evaluate(input_fn = eval_input_fn)

pred = list(linear_regressor.predict(input_fn = eval_input_fn))
pred = [p['predictions'][0] for p in pred]

prices = np.exp(pred)
print(prices)