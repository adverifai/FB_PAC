''' Pedram Hosseini '''
from sklearn.cluster import KMeans
import codecs
from pre_processing import pre_process
import pandas as pd
import numpy as np
from utils import load_file
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min

# %%
# READ DATA FILES
data_path = "fbpac-ads-en-US-train/fbpac-ads-en-US-train.csv"
advertiser_partisanship_path = "data/advertiser_partisanship.csv"
train_file = codecs.open(data_path, 'r', encoding="utf-8")
train_df = pd.read_csv(train_file)
train_file.close()
advertiser_df = train_df['advertiser']

# ===================================================================================================
# listing democrats (Dem), republicans (GOP), and other ('nonpartisan' and 'other') political parties
mapping_file = codecs.open(advertiser_partisanship_path, 'r', encoding="utf-8")
advertiser_partisanship = pd.read_csv(mapping_file)
mapping_file.close()
dems = []
gop = []
others = []
for index, row in advertiser_partisanship.iterrows():
    if row[2] == "Dem":
        dems.append(row[0])
    elif row[2] == "GOP":
        gop.append(row[0])
    elif row[2] == "nonpartisan" or row[2] == "other":
        others.append(row[0])


docs_topics_vectors = []
lda_model = load_file("models/LDAbow_fbpac.pickle")
lda_dictionary = load_file("models/LDAdict_fbpac.pickle")
for doc in train_df['text']:
    try:
        bow_vector = lda_dictionary.doc2bow(pre_process(doc))
        docs_topics_vectors.append(lda_model[bow_vector])
    except Exception as e:
        print(e)
        print("Error in computing topic vector")
n, nx, ny = np.array(docs_topics_vectors).shape
d2_all_docs = np.array(docs_topics_vectors).reshape((n, nx * ny))
X = d2_all_docs[:, 1::2]

x_filtered = []
x_advertiser = []
for i in range(n):
    result = np.sort(X[i])
    if not (round(X[i][3], 3) == 0.2 and round(X[i][4], 3) == 0.2):
        if str(advertiser_df[i]) != 'nan':
            x_filtered.append([X[i][3], X[i][4]])
            x_advertiser.append(advertiser_df[i])

print("Number of data points: " + str(len(x_filtered)))
# ======================================
# k means determine k using Elbow method
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(x_filtered)
    kmeanModel.fit(x_filtered)
    distortions.append(sum(np.min(cdist(x_filtered, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / len(x_filtered))

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
# plt.title('The Elbow Method showing k, the optimal number of clusters')
plt.show()

# the best k value is 6 based on elbow method.
k = 3

winners = ["Jon Tester", "Martin Heinrich", "Ted Cruz", "Dianne Feinstein"]

kmeans = KMeans(n_clusters=k, random_state=0).fit(x_filtered)
y_kmeans = kmeans.predict(x_filtered)

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, x_filtered)
print(closest)
for item in closest:
    print(x_advertiser[item])

# plt.scatter(np.array(x_filtered)[:, 0], np.array(x_filtered)[:, 1], c=y_kmeans, s=60, cmap='viridis')
plt.scatter(np.array(x_filtered)[:, 0], np.array(x_filtered)[:, 1], c=y_kmeans, s=60, cmap='rainbow')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5)
plt.xlabel('Topic distribution')
plt.ylabel('Topic distribution')
plt.show()


# getting top records based on frequency in each cluster
clusters_list = [[] for i in range(k)]
clusters_list_partisanship = [[] for i in range(k)]
for i in range(len(kmeans.labels_)):
    label = kmeans.labels_[i]
    clusters_list[label].append(x_advertiser[i])

    if x_advertiser[i] in dems:
        clusters_list_partisanship[label].append("Democrat")
    elif x_advertiser[i] in gop:
        clusters_list_partisanship[label].append("Republican")
    elif x_advertiser[i] in others:
        clusters_list_partisanship[label].append("Others")

print("\nTop records in clusters by advertiser's name")
for item in clusters_list:
    item = [x for x in item if str(x) != 'nan']
    c = Counter(item)
    print(c.most_common(10))

print("\nTop records in clusters based on partisanship")
for item in clusters_list_partisanship:
    item = [x for x in item if str(x) != 'nan']
    c = Counter(item)
    print(c.most_common(4))

# print(kmeans.labels_)

# checking winners in clusters
print("\nWinning candidates in clusters")
for item in clusters_list:
    item = [x for x in item if str(x) in winners]
    print(set(item))
