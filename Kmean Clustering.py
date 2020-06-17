#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values


# In[2]:


import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], color= "red")
plt.show()


# In[4]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(wcss)
plt.show()


# In[5]:


kmeans = KMeans(n_clusters = 5, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# In[8]:


from matplotlib.colors import ListedColormap
raw_colors = ("red", "green", "black", "cyan", "magenta")
colors = ListedColormap(raw_colors)
for i in range(5):
    plt.scatter(X[y_kmeans == i,0], X[y_kmeans == i,1],s = 100, c = colors(i))
    X_clusters = kmeans.cluster_centers_[:, 0]
    Y_clusters = kmeans.cluster_centers_[:, 1]
    plt.scatter(X_clusters, Y_clusters, s = 300, c = "yellow")
plt.show()


# In[ ]:




