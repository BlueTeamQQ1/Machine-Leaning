# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:23:29 2020

@author: BlueQQ1
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras
data = keras.datasets.mnist
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
(X_train, y_train), (X_test, y_test) = data.load_data()
X_train_new = []
X_test_new = []
for i in range(0, 60000):
  X_train_new.append(hog(X_train[i]))
for i in range(0, 10000):
  X_test_new.append(hog(X_test[i]))
X_train_new = np.array(X_train_new)
X_test_new = np.array(X_test_new)
model = LogisticRegression(solver= 'newton-cg')
model.fit(X_train_new, y_train)
p  = model.predict(X_test_new)
accuracy_score(y_test, p)
fig = plt.figure(figsize=(16,9))
plot_confusion_matrix(model, X_test_new, y_test, cmap = plt.cm.Reds, )
from sklearn.metrics import classification_report
print(classification_report(y_test, p))
def logistic_model(X_train, X_test, y_train, y_test, solver = 'lbfgs'):
  model = LogisticRegression(solver= solver)
  model.fit(X_train.reshape(60000, 28*28), y_train)
  p = model.predict(X_test.reshape(10000,28*28))
  plot_confusion_matrix(model, X_test.reshape(10000, 28*28), y_test)
  print(classification_report(y_test, p))
logistic_model(X_train,X_test, y_train, y_test)