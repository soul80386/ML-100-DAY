"""第一步：导入需要的库"""
#以下两个是我们每次都需要导入的库
#numpy包含数学计算函数
#pandas用于导入和管理数据集
import numpy as np 
import pandas as pd
#处理数据的库
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

#获取当前绝对路径 
#import os
#print(os.getcwd()) #F:\GitHub\ML-100-DAY

"""第二步：导入数据集"""
dataset = pd.read_csv('./Day_01_Data_Preprocessing/Data.csv')
X = dataset.iloc[ : , : -1].values  #.iloc[行，列],全部行，排除最后一列（结果列）
Y = dataset.iloc[ : , 3 ].values    #全部行，最后一列
print("第二步：导入数据")
print("X:")
print(X)
print("Y：")
print(Y)

"""第三步：处理丢失数据"""
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print("------------------------------------")
print("第三步：处理丢失数据")
print("X")
print(X)
""""第四步：解析分类数据"""
#分类数据是含有标签值而不是数据值的变量
#分类数据不能用于模型的数据计算，需要解析成数字
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : ,0])
#创建虚拟变量
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("------------------------------------")
print("第四步：解析分类数据")
print("X:")
print(X)
print("Y：")
print(Y)

"""第五步：拆分数据为训练集和测试集"""
#一般训练集和测试集的比例为80：20
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print("------------------------------------")
print("第五步：拆分数据为训练集和测试集")
print("X_train:")
print(X_train)
print("X_test：")
print(X_test)
print("Y_train:")
print(Y_train)
print("Y_test：")
print(Y_test)

"""第六步：特征量化(特征缩放)"""
#
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("------------------------------------")
print("第六步：特征量化(特征缩放)")
print("X_train:")
print(X_train)
print("X_test：")
print(X_test)