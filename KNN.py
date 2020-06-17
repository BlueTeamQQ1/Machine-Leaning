# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:10:52 2020

@author: BlueQQ1
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import joblib
train=pd.read_csv('Social_Network_Ads_Train.csv')
test=pd.read_csv('Social_Network_Ads_Test.csv')
X_train=train.iloc[:,[3,4]].values
Y_train=train.iloc[:,5].values
X_test=test.iloc[:,[3,4]].values
Y_test=test.iloc[:,5].values
SC=StandardScaler()
X_train=SC.fit_transform(X_train)
X_test=SC.fit_transform(X_test)
def VisualizingDataset(X_,Y_):
    X1=X_[:,0]
    X2=X_[:,1]
    for i,label in enumerate(np.unique(Y_)):
        plt.scatter(X1[Y_ == label],X2[Y_ ==label], color = ListedColormap(("Red","Green"))(i))
def VisualizingResult(model ,X_):
    X1 = X_[:,0]
    X2 = X_[:,1]
    X1_range = np.arange(start = X1.min()-1, stop = X1.max()+1, step = 0.01)
    X2_range = np.arange(start = X2.min()-1, stop = X2.max()+1, step =0.01)
    X1_matrix, X2_matrix = np.meshgrid(X1_range, X2_range)
    X_grid = np.array([X1_matrix.ravel(), X2_matrix.ravel()]).T
    Y_grid =model.predict(X_grid).reshape(X1_matrix.shape)
    plt.contourf(X1_matrix, X2_matrix,Y_grid,alpha= 0.5 ,cmap = ListedColormap(("red","green")))
so={3,5,7,9}
for i in so:
    classifier = KNeighborsClassifier(n_neighbors = i )
    classifier.fit(X_train, Y_train)
    joblib.dump(classifier, 'KNNmodel_with_{} neighbors.h5'.format(i))
    cm = confusion_matrix(Y_test, classifier.predict(X_test))
    print(cm)
    VisualizingResult(classifier, X_test)
    VisualizingDataset(X_test, Y_test)
    plt.show()
    plt.savefig('KNNmodel_with_{} neighbors'.format(i))
