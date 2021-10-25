import csv
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
fig = px.scatter(df, x="petal_size", y="sepal_size")
fig.show()

#choosing the right amount of k using wcss parameter
X = df.iloc[:,[0,1]].values
print(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plotting the figure to show an elbow structure
plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), wcss, marker = "o", color = "red")
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters = 3, init = "k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize = (10, 7))
sns.scatterplot(X[y_kmeans==0, 0], X[y_kmeans==0, 1], color="yellow", label = "Cluster 1")
sns.scatterplot(X[y_kmeans==1, 0], X[y_kmeans==1, 1], color="blue", label = "Cluster 2")
sns.scatterplot(X[y_kmeans==2, 0], X[y_kmeans==2, 1], color="green", label = "Cluster 3")

sns.scatterplot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = "red", label = "Centeroids", s = 100, marker = ",")
plt.grid(False)
plt.title("Clusters of flowers")
plt.xlabel("Petal size")
plt.ylabel("Sepal size")
plt.legend()
plt.show()