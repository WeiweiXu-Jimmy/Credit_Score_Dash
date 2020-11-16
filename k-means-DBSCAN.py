#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)


# In[71]:


km= pd.read_csv('/Users/xudawei/Desktop/Dash/Credit的副本/data/Kmeans')

km.head(n=2)


# In[72]:


# Data des individus qui ont mauvais credits 
km_M = km.groupby('Credit').get_group('M').drop(columns = ['Credit'])
df = km_M[['Durée_du_crédit','Montant_du_crédit','Montant_du_renouvelable','Age','Revenu']]
df.head()


# In[73]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[74]:


kmeans = KMeans(n_clusters=4, random_state=10).fit(df)
df['LabelClass']=kmeans.labels_
df_count_type=df.groupby('LabelClass').count()


# In[75]:


df_count_type


# In[76]:


kmeans.cluster_centers_


# In[77]:


df.head()


# In[ ]:





# In[78]:


df.loc[df['LabelClass'] == 3,'K-means'] = '1_catégorie'
df.loc[df['LabelClass'] == 0,'K-means'] = '2_catégorie'
df.loc[df['LabelClass'] == 2,'K-means'] = '3_catégorie'
df.loc[df['LabelClass'] == 1,'K-means'] = 'Autres'


# In[79]:


df.head(n=2)


# In[80]:


df =df.drop(['LabelClass'],axis =1)


# In[81]:


df.head(n=2)


# In[82]:


from sklearn.cluster import DBSCAN
 
X = df[["Durée_du_crédit","Montant_du_crédit","Montant_du_renouvelable",
        "Age","Revenu"]]

db = DBSCAN(eps=5, min_samples=2).fit(X)
 
labels = db.labels_ 
df['cluster_db'] = labels  
df.head(n=2)
 


# In[83]:


df_type=df.groupby('cluster_db').count()
df_type


# In[84]:


df.loc[df['cluster_db'] == -1,'DBSCAN'] = '1_catégorie'
df.loc[df['cluster_db'] == 0,'DBSCAN'] = '2_catégorie'
df.loc[df['cluster_db'] == 1,'DBSCAN'] = '3_catégorie'
df.loc[df['cluster_db'] == 2,'DBSCAN'] = 'Autres'


# In[85]:


df.head(n=2)


# In[86]:


df =df.drop(['cluster_db'],axis =1)


# In[87]:


df.head()


# In[88]:


km_M.shape


# In[89]:


km_M['km_labels'] = df['K-means']
km_M['db_labels'] = df['DBSCAN']
km_M.head(n=2)


# In[ ]:





# In[ ]:





# In[90]:


km_M.to_csv('/Users/xudawei/Desktop/Dash/Credit的副本/data/K-means-DBSCAN')


# In[ ]:





# In[ ]:




