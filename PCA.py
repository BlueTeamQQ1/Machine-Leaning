#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[43]:


dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:,0:13].values
Y = dataset.iloc[:,-1].values


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state=0)


# In[45]:


from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X_train_sc = SC.fit_transform(X_train)
X_test_sc = SC.transform(X_test)


# In[46]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_sc_pca = pca.fit_transform(X_train_sc)
X_test_sc_pca = pca.transform(X_test_sc)


# In[47]:


from sklearn.linear_model import LogisticRegression
pca_log_reg = LogisticRegression(random_state=0)
pca_log_reg.fit(X_train_sc_pca, Y_train)


# In[48]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(pca_log_reg,X_test_sc_pca,Y_test)


# In[49]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, pca_log_reg.predict(X_test_sc_pca)))


# In[ ]:





# In[ ]:




