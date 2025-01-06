#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kneed
from kneed import KneeLocator
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# In[2]:


# input data
df = pd.read_csv('E:\PKL\Penyandang Disabilitas di Kota Yogyakarta 2022.csv',sep=",", header=0)
df.info()


# In[3]:


# ubah ke data perhitungan
data = df.drop('Kecamatan Subdistrict', axis=1)
data.info()


# In[4]:


# standarisasi
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data


# In[5]:


# memilih jumlah cluster yang sesuai menggunakan metode siku
# model building
kmeans_kwargs = {
    "init": "random",
    "n_init": 'auto',
    "max_iter": 300,
    "random_state": 42,}
# metode siku
sse_list = [] # membuat variabel untuk menyimpan nilai SSE
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_data)
    sse = kmeans.inertia_ # menghitung nilai SSE
    sse_list.append(sse) # menambahkan nilai SSE ke dalam variabel
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse_list, label='SSE') # menambahkan label 'SSE'
plt.xticks(range(1, 11))
plt.xlabel("Cluster")
plt.ylabel("SSE")
# menentukan nilai k optimal dengan fungsi knee_locator
kl = KneeLocator(range(1, 11), sse_list, curve="convex", direction="decreasing")
# menambahkan titik siku pada plot dengan warna merah dan label 'Elbow'
plt.vlines(kl.elbow, ymin=min(sse_list), ymax=max(sse_list), linestyles='dashed', color='red', label='Elbow')
# menambahkan legenda pada plot
plt.legend()
plt.show()


# In[6]:


# jumlah cluster
k = kl.elbow

# model building
kmeans = KMeans(
    init="random",
    n_clusters=k,
    n_init='auto',
    max_iter=300,
    random_state=42)
kmeans.fit(data)

# nilai terkecil SSE
kmeans.inertia_

# lokasi final centroid
kmeans.cluster_centers_

# banyak iterasi
kmeans.n_iter_

# urutan label
kmeans.labels_

# Mengambil kolom yang ingin digunakan sebagai fitur untuk clustering
features = data.iloc[:, [0, 1]].values

# Membuat model K-Means dengan jumlah cluster k
kmeans = KMeans(n_clusters=k, n_init='auto')
kmeans.fit(features)

# Mendapatkan label untuk setiap data
labels = kmeans.labels_

# Mendapatkan koordinat pusat untuk setiap cluster
centroids = kmeans.cluster_centers_

# Menambahkan 1 ke label cluster
labels = kmeans.labels_ + 1

# Menambahkan kolom cluster ke data frame asli
df['cluster'] = labels

# Menampilkan data frame dengan informasi cluster
df


# In[7]:


# mendapatkan nilai unik pada labels
unique_labels = np.unique(labels)
# membuat scatter plot untuk setiap nilai unik dengan label yang sesuai
for label in unique_labels:
    # memilih data yang memiliki label yang sama
    data = features[labels == label]
    # membuat scatter plot dengan warna yang berbeda untuk setiap label
    plt.scatter(data[:, 0], data[:, 1], label=f'Cluster {label}')
# menambahkan scatter plot untuk centroid dengan label 'Centroid'
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
            s=200, linewidths=3, color='r', label='Centroid')
plt.title('Hasil Clustering Daerah Penyandang Disabilitas')
plt.legend()
plt.show()


# In[ ]:




