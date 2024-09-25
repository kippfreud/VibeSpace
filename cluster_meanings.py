import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import json
import random

import gensim.models
import hdbscan
import joblib
import numpy as np  # array handling
import pandas as pd
import requests
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE  # final reduction
from tqdm import tqdm

# domain = "book"
# domain = "movie"
# domain = "song"


def reduce_dimensions(wv, ndims=2):
    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(wv.vectors)
    labels = np.asarray(wv.index_to_key)  # fixed-width numpy strings
    # reduce using t-SNE
    tsne = TSNE(
        n_components=ndims, random_state=0, early_exaggeration=12, perplexity=30
    )
    vectors = tsne.fit_transform(vectors)
    # pca = IncrementalPCA(n_components=num_dimensions, whiten=True)
    # vectors = pca.fit_transform(vectors)
    #
    # x_vals = [v[0] for v in vectors]
    # y_vals = [v[1] for v in vectors]
    return vectors, labels


def plot_with_matplotlib(x_vals, y_vals, labels=[], colors=[]):
    if len(labels):
        df = pd.DataFrame.from_dict(
            {
                "x": x_vals,
                "y": y_vals,
                "labels": labels,
            }
        )
    else:
        df = pd.DataFrame.from_dict(
            {
                "x": x_vals,
                "y": y_vals,
            }
        )
    if len(colors):
        plt.scatter(df["x"], df["y"], s=1, c=colors, alpha=0.5, cmap="viridis")
    else:
        plt.scatter(df["x"], df["y"], s=1)
    if len(labels):
        indices = list(range(len(labels)))
        selected_indices = random.sample(indices, 300)
        for i in selected_indices:
            plt.annotate(labels[i], (x_vals[i], y_vals[i]), fontsize=3)
    # plt.show()


# %%
domain = "movie"
with open(f"{domain}_meta_fullv8_v3.json", "r", encoding="utf8") as f:
    corpus = json.load(f)
c_keys = list(corpus.keys())
to_unique = {}
unique_to_genre = {}
unique_to_meta = {}
v = 0
for i in tqdm(c_keys):
    if type(corpus[i]) is dict:
        v += 1
        # if "similar" in corpus[i].keys():
        #     if len(corpus[i]["similar"]):
        if (
            ("title" in corpus[i].keys())
            and ("cast" in (corpus[i].keys()))
            and ("directors" in (corpus[i].keys()))
        ):
            # if (corpus[i]["cast"] is not None):
            cast = ""
            if corpus[i]["cast"] is None:
                cast = "NONE "
            else:
                for ca in corpus[i]["cast"]:
                    # print(ca)
                    cast += f"{ca} "
            directors = ""
            if corpus[i]["directors"] is None:
                directors = "NONE "
            else:
                for di in corpus[i]["directors"]:
                    # print(ca)
                    directors += f"{di} "
            # print(cast)
            i_ = i.replace(" ", "_").replace("&", "and").lower()
            i_ = i_ if i_[0] != "_" else i_[1:]
            id_ = (corpus[i]["title"] + cast).replace(" ", "_")
            to_unique[i_] = id_
            if corpus[i]["genres"] is not None:
                unique_to_genre[id_] = corpus[i]["genres"]
            else:
                unique_to_genre[id_] = None
            unique_to_meta[id_] = corpus[i]
            # unique_t_c_d.append(corpus[i]["title"]+cast+directors)
            # unique_c.append(cast)


# model = gensim.models.Word2Vec.load("movie.wordvectors")
wv = KeyedVectors.load("movie.wordvectors", mmap="r")

# vectors, labels = reduce_dimensions(wv, 2)

kmeans = KMeans(n_clusters=10)
kmeans.fit(wv.vectors)


vectors, labels = reduce_dimensions(wv, 2)
plot_with_matplotlib(
    [v[0] for v in vectors], [v[1] for v in vectors], colors=kmeans.labels_
)
plt.show()


# names = model.wv.index_to_key
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, alpha=1.0)
clusterer.fit(vectors)
print(clusterer.labels_.max())
clusterer.condensed_tree_.plot()
# filename = 'clusterer.joblib'
# joblib.dump(clusterer, filename)
plt.show()
print("ok")
