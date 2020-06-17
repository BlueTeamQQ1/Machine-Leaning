#!/usr/bin/env python
# coding: utf-8

# In[181]:


from sklearn.datasets import load_iris
import pandas as pd


# In[182]:


import pandas as pd
data=pd.read_csv(r'C:\Users\18521\Downloads\iris.csv')
data.head()


# In[183]:


data.shape
X=data.iloc[:,0:4]
Y=data.iloc[:,-1]
print(Y)


# In[184]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, train_size= 0.8, random_state=0)


# In[185]:


X_test.shape


# In[186]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion= "entropy",random_state=0)
classifier.fit(X_train, Y_train)


# In[187]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#Use the pandas apply method to numerically encode our attrition target variable


# In[188]:


plt.figure(figsize=(10,11))
plot_tree(classifier)
plt.show


# In[189]:


classifier.score(X_train,Y_train)


# In[190]:


classifier.score(X_test,Y_test)


# In[ ]:




