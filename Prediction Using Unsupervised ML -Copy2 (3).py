#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION 
# ## Task 2: Prediction Using Unsupervised ML

# # Nourhan Mohammed Ali Ebrahim

# # Importing All Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime
from sklearn import datasets
import sklearn 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# # loading dataset..

# In[4]:


iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df.head()


# In[6]:


iris_df.info()


# In[7]:


iris_df.shape


# In[8]:


sns.pairplot(iris_df)
plt.show()


# In[9]:


sns.heatmap(iris_df.corr())


# In[11]:


sns.distplot(iris_df['sepal length (cm)'])


# In[13]:


iris_df.columns


# In[16]:


x=iris_df.iloc[:,[0,1,2,3]].values
x


# # Finding The Optimum Num of Clusters for K-means Classification..

# In[35]:


x=iris_df.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans= KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    


# # applying Kmeans to The Dataset..

# In[19]:


kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)


# # Visualising The Clusters

# In[34]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='yellow',label='Iris-versicolor')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='black',label='Iris_virginca')                      
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='green',label='Centroids')
plt.legend()


# In[ ]:




