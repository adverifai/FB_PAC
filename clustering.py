from sklearn.cluster import KMeans
import codecs
from pre_processing import pre_process
import pandas as pd
import numpy as np
from utils import load_file
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# %%
# READ DATA FILES
train_file = codecs.open('fbpac-ads-en-US-train.csv', 'r', encoding="utf-8")
train_df = pd.read_csv(train_file)
train_file.close()

advertiser_df = train_df['advertiser']


docs_topics_vectors = []
lda_model = load_file("models/LDAtfidf_fbpac.pickle")
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

# k means determine k using Elbow method
# distortions = []
# K = range(1, 10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(X)
#     kmeanModel.fit(X)
#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
#
# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

# the best k value is 6 based on elbow method.
k = 6
kmeans = KMeans(n_clusters=k, random_state=0).fit(d2_all_docs[:, 1::2])
clusters_list = [[] for i in range(k)]
for i in range(len(kmeans.labels_)):
    label = kmeans.labels_[i]
    clusters_list[label].append(advertiser_df[i])

for item in clusters_list:
    item = [x for x in item if str(x) != 'nan']
    c = Counter(item)
    print(c.most_common(5))

print(kmeans.labels_)