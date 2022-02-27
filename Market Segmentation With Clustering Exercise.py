from sklearn import preprocessing
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv ('3.12. Example.csv')
data

plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
# To check the plot and see if I can spot any preliminary patterns.

x = data.copy()
kmeans = KMeans(2)
kmeans.fit(x)
clusters = x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)

plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

x_scaled = preprocessing.scale(x)
x_scaled
# Here I preprocessed the data using sklearn to easily standardize the variables.

wcss =[]
# To measure the distance between points in a cluster and among different clusters and decide on the appropriate number of clusters.
for i in range(1,10):
# For the Elbow Method
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# This is the Elbow Method.

kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)

plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
