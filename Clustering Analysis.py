
  
#2.cluster analysis -when we don't know the target variable apply the cluster analysis
#COMPLETED

#AIRLINES DATA
import pandas as pd
df=pd.read_excel("EastWestAirlines.xlsx")
df
df.dtypes
list(df)
df.shape
df.head()

#graph
import matplotlib.pyplot as plt
plt.scatter(df['Balance'],df['Qual_miles'],color='purple')
plt.ylabel("Qual_mile")
plt.xlabel("Balance")
plt.show()


#splitting
X=df.iloc[:,1:]

# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X

#a)AGGLOMERATIVE CLUSTER==============================
#forming a group using  clusters

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='complete')
Y = cluster.fit_predict(SS_X)
df["Y"] = pd.DataFrame(Y)
df["Y"].value_counts()
df

#B)K.MEANS CLUSTER=================================
from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters=5,n_init=30)
KMeans.fit(SS_X)
df["Kmeans_Y"]=pd.DataFrame(Y)
df["Kmeans_Y"].value_counts()

KMeans.inertia_

#To identify the best K value from all possible K values
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(SS_X)
    inertia.append(km.inertia_)
    
print(inertia)
plt.plot(range(1,11),inertia,linestyle='--',marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#C)DBSCAN=NOT ONLY FORMING THE CLUSTERS BUT ALSO FINDING THE OUTLIERS=========================================

from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)

#Noisy samples are given the label -1.
dbscan.labels_

df["dbscan_labels"] = pd.DataFrame(dbscan.labels_)
df["dbscan_labels"].value_counts()

df_final = df[df["dbscan_labels"] != -1]
df_final.shape

df_final.head()
list(df_final)

df_final





#CRIME DATA

import pandas as pd
df=pd.read_csv("crime_data.csv")
df
df.dtypes
list(df)
df.shape

#graph
import matplotlib.pyplot as plt
plt.scatter(df['Murder'],df['Rape'],color='red')
plt.ylabel("Rape")
plt.xlabel("Murder")
plt.show()


#splitting
X=df.iloc[:,1:]

# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X


#alternative normalization function 
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


#a)AGGLOMERATIVE CLUSTER==============================
#forming a group using  clusters

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='complete')

Y = cluster.fit_predict(SS_X)
df["Y"] = pd.DataFrame(Y)
df["Y"].value_counts()
df



#B)K.MEANS CLUSTER=================================
from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters=4,n_init=30)
KMeans.fit(SS_X)
df["Kmeans_Y"]=pd.DataFrame(Y)
df["Kmeans_Y"].value_counts()

KMeans.inertia_

#To identify the best K value from all possible K values
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(SS_X)
    inertia.append(km.inertia_)
    
print(inertia)
plt.plot(range(1,11),inertia,linestyle='--',marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#C)DBSCAN=NOT ONLY FORMING THE CLUSTERS BUT ALSO FINDING THE OUTLIERS=========================================

from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)

#core
#border
#Noisy samples are given the label -1.
dbscan.labels_

df["dbscan_labels"] = pd.DataFrame(dbscan.labels_)
df["dbscan_labels"].value_counts()

df_final = df[df["dbscan_labels"] != -1]
df_final.shape

df_final.head()
list(df_final)

df_final