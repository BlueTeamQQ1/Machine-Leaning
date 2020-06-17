# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:31:12 2020

@author: BlueQQ1
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 
import math
from sklearn.metrics import r2_score
train= pd.read_csv('Position_SalariesTrain.csv')
test= pd.read_csv('Position_SalariesTest.csv')
X_train = train.iloc[:,1:-1].values
Y_train = train.iloc[:,-1].values.reshape(-1,1)
X_test= test.iloc[:,1:-1].values
Y_test= test.iloc[:,-1].values.reshape(-1,1)
plt.scatter(X_train,Y_train, color='red')
plt.title('position level vs salary')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
SC_X=StandardScaler()
SC_Y=StandardScaler()
X_trans=SC_X.fit_transform(X_train)
Y_trans=SC_Y.fit_transform(Y_train)
svr= SVR( kernel = 'rbf')
svr.fit(X_trans,Y_trans)
def predict(model, X,SC_X,SC_Y):
    X_trans=SC_X.transform(X)
    Y_trans_pred=model.predict(X_trans)
    Y_pred=SC_Y.inverse_transform(Y_trans_pred)
    return Y_pred
Y_pred_train=predict(svr,X_train,SC_X,SC_Y)
Y_pred_test=predict(svr,X_test,SC_X,SC_Y)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,Y_pred_train, color='blue')
plt.title('position level vs salary  ')
plt.xlabel('position level')
plt.ylabel('salary (dollars/year)')
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred_test, color='blue')
plt.title('position level vs salary  ')
plt.xlabel('position level')
plt.ylabel('salary (dollars/year)')
plt.show()
print('r2_score_train ',r2_score(Y_train, Y_pred_train))
print('r2_score_test ',r2_score(Y_test, Y_pred_test))
rmsetrain = math.sqrt(mean_squared_error(Y_train,Y_pred_train))
rmsetest = math.sqrt(mean_squared_error(Y_test,Y_pred_test))
print('RMSE_Train',rmsetrain)
print('RMSE_Test',rmsetest)
