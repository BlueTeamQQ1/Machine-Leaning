# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:02:16 2020

@author: BlueQQ1
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
test = pd.read_csv('Position_SalariesTest.csv')
train = pd.read_csv('Position_SalariesTrain.csv')
X_train = train.Level
X_test = test.Level
Y_train =train.Salary
Y_test = test.Salary
plt.scatter(X_train, Y_train,color ='red')
plt.title('positition level vs salary ')
plt.xlabel('positition level ')
plt.ylabel('salary')
plt.show()
transform=PolynomialFeatures(degree =4)
X_poly = transform.fit_transform(np.array(X_train).reshape(-1,1))
X_test_poly=transform.fit_transform(np.array(X_test).reshape(-1,1))
poly_lin=LinearRegression()
poly_lin.fit(X_poly,Y_train)
poly_lin.fit(X_test_poly,Y_test)
plt.scatter(X_train, Y_train,color ='red')
plt.plot(X_train, poly_lin.predict(X_poly),color= 'blue')
plt.title('positition level vs salary ')
plt.xlabel('positition level ')
plt.ylabel('salary')
plt.show()
plt.scatter(X_test, Y_test,color ='red')
plt.plot(X_test_poly, poly_lin.predict(X_test_poly),color= 'blue')
plt.title('positition level vs salary ')
plt.xlabel('positition level ')
plt.ylabel('salary')
plt.show()
print(r2_score(Y_test, poly_lin.predict(X_test_poly)))