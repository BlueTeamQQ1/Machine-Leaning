# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:54:39 2020

@author: BlueQQ1
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
train = pd.read_csv('Social_Network_Ads_Train.csv')
test = pd.read_csv('Social_Network_Ads_Test.csv')
X_train =train.iloc[:,[3,4]].values
X_test =test.iloc[:,[3,4]].values
Y_train =train.iloc[:,-1].values
Y_test =test.iloc[:,-1].values
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.fit_transform(X_test)
from matplotlib.colors import ListedColormap
def VisualizingDataset(X_, Y_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    for i, label in enumerate(np.unique(Y_)):
        plt.scatter(X1[Y_ == label], X2[Y_ == label],
        color = ListedColormap(("red", "green"))(i),label = label)
VisualizingDataset(X_train,Y_train)
plt.show()
model = LogisticRegression(random_state=0)
model.fit(X_train,Y_train)
cm = confusion_matrix(Y_train, model.predict(X_train))
print(cm)
def VisualizingResult(model, X_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    X1_range = np.arange(start = X1.min()-1, stop = X1.max()+1, step = 0.01)
    X2_range = np.arange(start= X2.min()- 1, stop= X2.max()+1,step = 0.01)
    X1_matrix, X2_matrix = np.meshgrid(X1_range, X2_range)
    X_grid= np.array([X1_matrix.ravel(), X2_matrix.ravel()]).T
    Y_grid= model.predict(X_grid).reshape(X1_matrix.shape)
    plt.contourf(X1_matrix, X2_matrix, Y_grid, alpha = 0.5,cmap = ListedColormap(("red", "green")))
VisualizingResult(model,X_train)
VisualizingDataset(X_train,Y_train)
plt.show()
cm = confusion_matrix(Y_test,model.predict(X_test))
print(cm)
VisualizingResult(model,X_train)
VisualizingDataset(X_test,Y_test)
plt.show()
