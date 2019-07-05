import numpy as np 
import pandas as pd 
dataset = pd.read_csv('./Day_01_Data_Preprocessing/50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 4].values
print(X[:10])
print(Y[:10])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[ : ,3] = labelencoder.fit_transform(X[ : , 3])
print("labelencoder:", X[:10])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
print("onehot:", X[:10])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
print(y_pred)