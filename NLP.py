#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


# In[40]:


df = pd.read_csv("news.csv")


# In[41]:


df


# In[50]:


X = df.drop(columns= "label")
Y = df["label"]


# In[43]:


from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
X['title'] = labelencoder.fit_transform(X['title'])
X['text'] = labelencoder.fit_transform(X['text'])
Y=labelencoder.fit_transform(Y)


# In[44]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[45]:


from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)


# In[54]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy = ",(accuracy))
plot_confusion_matrix(model,X_test,Y_test)


# In[ ]:




