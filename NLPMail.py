#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import KFold


# In[2]:


df = pd.read_csv("emails.csv")


# In[3]:


df.head()


# In[4]:


df.spam.hist()


# In[6]:


nltk.download("stopwords")
corpus = []
for i in range(0, 5728):
    text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)


# In[18]:


a=[]
for i in corpus:
    b = i.replace("subject ","")
    a.append(b)


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 25000)
X = cv.fit_transform(a).toarray()
Y = df.iloc[:,1].values


# In[81]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 0)


# In[82]:


#from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()
#model.fit(X_train, Y_train)
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200,learning_rate=0.2)
model.fit(X_train, Y_train)


# In[83]:


Y_pred = model.predict(X_test)


# In[84]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model,X_test,Y_test)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


# In[85]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy: ",accuracy)
from sklearn.metrics import precision_score
print("Precision = ",precision_score(Y_test, Y_pred))
from sklearn.metrics import recall_score
print("Recall = ",recall_score(Y_test, Y_pred))
from sklearn.metrics import f1_score
print("F-measure = ",f1_score(Y_test, Y_pred))


# In[ ]:




