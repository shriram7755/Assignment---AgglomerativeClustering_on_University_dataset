# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:17:26 2023

@author: SHRI
"""

import pandas as pd
import matplotlib.pylab as plt

#now import file  from data set and create a dataframe

Univ1=pd.read_excel('University_Clustering.xlsx')

#five Number Summary
a=Univ1.describe()

#finding Columns
Univ1.columns

Univ=Univ1.drop(['State'],axis=1)
Univ
                                                                                                                                                                       y# we know there is scale difference between among the columns,
#which we have remove
#either by using the normalization and standardization
#when ever there is mixed data apply normalization



def norm_func(i):
    x=(i-i.min())/(i.max()- i.min())
    return x

#Now apply this normalization function to dataframe
# for all the rows and columns from 1 until end
# Since 0th columns has university names henced skipped
df_norm=norm_func(Univ.iloc[:,1:])
df_norm


b=df_norm.describe()


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering Tendrogram");
plt.xlabel('Index')
plt.ylabel('Distance')


sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show


#TEndogram
#apply agglomerative clustering choosing 3 as clusters
#from Tendogram
#it is just showing the number of possible clusters

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean').fit(df_norm)
#label to the clusters

cluster_label=pd.Series(h_complete.labels_)

Univ['clust']=cluster_label


Univ.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()

