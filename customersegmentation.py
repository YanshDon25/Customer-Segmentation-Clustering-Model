import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('Mall_Customers.csv')
x=data.iloc[:,[3,4]].values
#using dendrogram to find optimal no. of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
#training the model
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage= 'ward')
y_hc=hc.fit_predict(x)
#visualising the clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='pink',label='Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='orange',label='Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='brown',label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()