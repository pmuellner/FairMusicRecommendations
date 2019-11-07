import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, AffinityPropagation, Birch
from utils import visualize_clusters, plot_silhouette, normalize, gap_statistic, explain_clusters, sse, chunkify
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.impute import SimpleImputer
import sqlalchemy
import json
import umap
from collections import Counter
import ast
from hdbscan import HDBSCAN
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA, DictionaryLearning, dict_learning
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Memory
from collections import defaultdict
from pprint import PrettyPrinter
import pickle
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

plt.style.use("seaborn")

N_FEATURES = 8
COL_SEPARATOR = "123456789987654321"

def compute_rel_genre_score(list_of_genres):
    n = len(list_of_genres)
    track_genres_occ = list(Counter(list_of_genres).items())
    rel_scores = [(genre, occ / n) for genre, occ in track_genres_occ]
    rel_scores = sorted(rel_scores, key=lambda tup: tup[1])[::-1]
    return rel_scores[:10]

def compute_genre_idf(list_of_genres, idfs, n_returns=None):
    track_genres_occ = list(Counter(list_of_genres).items())
    print(track_genres_occ)
    scores = []
    for genre, occ in track_genres_occ:
        if genre in idfs:
            idf = idfs[genre]
            scores.append((genre, idf * occ))
    scores = sorted(scores, key=lambda tup: tup[1])[::-1]
    return scores[:n_returns]


"""
read and scale input data
"""
SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')

"""user_track_df = pd.read_csv("user_track.csv", sep=";")
print(len(user_track_df))
rel_tracks = list(set(user_track_df["track_id"]))"""

"""rel_tracks = pd.read_csv("../feature_engineering/data/lowms_events_nondominating.csv", sep=";")["track_id"].unique()
print(len(rel_tracks))
#print(user_track_df["user_id"].nunique())

statement = "SELECT track_id, danceability, energy, speechiness, acousticness, instrumentalness, tempo, valence, liveness FROM acoustic_features"
df = pd.read_sql(sql=statement, con=engine).set_index("track_id")
print(len(df))
df = df.loc[rel_tracks].dropna()
print(len(df))


track_genres_df = pd.read_csv("track_genres.csv", sep=";")
track_genres_df.columns = ["track_id", "genres"]
track_genres_df = track_genres_df[track_genres_df["genres"] != "[]"]
track_genres_df.set_index("track_id", inplace=True)

# use only tracks with genre annotations
df = df.loc[track_genres_df.index].dropna()
print(len(df))

exit()"""

"""rel_tracks = list(df.index)
with open("relevant_tracks_lowms.txt", "w") as f:
    for item in rel_tracks:
        f.write("%s\n" % item)

exit()"""



"""scaler = MinMaxScaler()
tracks_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
print(len(tracks_df))

#tracks_df.to_csv("lowms_tracks.csv", sep=";", index=True)
tracks_df.to_csv("lowms_tracks_clean.csv", sep=";", index=True)

# umap dimensionality reduction
#tracks_df = pd.read_csv("lowms_tracks.csv", sep=";", index_col="track_id")
tracks_df = pd.read_csv("lowms_tracks_clean.csv", sep=";", index_col="track_id")
tracks_df = pd.DataFrame(umap.UMAP().fit_transform(tracks_df), index=tracks_df.index)
tracks_df.to_csv("tracks_umap_enc_clean.csv", sep=";", index=True)"""

tracks_df = pd.read_csv("tracks_umap_enc_clean.csv", sep=";", index_col="track_id")
print(len(tracks_df))

# use only genre annotated tracks for clustering
"""track_genres_df = pd.read_csv("track_genres.csv", sep=";")
track_genres_df.columns = ["track_id", "genres"]
track_genres_df.set_index("track_id", inplace=True)

# use only tracks with genre annotations
tracks_df = tracks_df.loc[track_genres_df.index].dropna()"""
print(len(tracks_df))

"""models = []
scores = []
min_cluster_sizes = list(range(1000, 1500, 25))
df = tracks_df.copy()
p, predictions = None, None
n_clusters = []
for min_cs in min_cluster_sizes:
    min_samples = list(range(min_cs - 50, min_cs + 75, 25))
    sse_min_samples = []
    best_sse = np.inf
    min_samples = [min_cs]
    for min_s in min_samples:
        hdbscan = HDBSCAN(min_cluster_size=min_cs, min_samples=min_s, memory="./cachedir")
        p = hdbscan.fit_predict(tracks_df)

        df["c"] = p
        cluster_means = df.groupby(by="c").mean()
        loss = np.mean(sse(data=tracks_df, predictions=p, cluster_means=cluster_means))
        if loss < best_sse:
            best_sse = loss
            predictions = p
            best_config = (min_cs, min_s)

        print(min_cs, min_s, len(set(hdbscan.labels_)))

    df["c"] = predictions
    cluster_means = df.groupby(by="c").mean()
    #sserrors = sse(data=tracks_df, predictions=predictions, cluster_means=cluster_means)
    #scores.append(np.mean(sserrors))
    #print("sse: %f" % np.mean(sserrors))

    n_clusters.append(len(set(predictions)))

    #visualize_clusters(data=tracks_df, predictions=predictions, n_cluster=len(set(predictions)), stochastic=True)
    #print(np.mean(sserrors))

    gc.collect()


plt.plot(min_cluster_sizes, scores)
plt.show()"""










"""
hdbscan
"""
hdbscan = HDBSCAN(min_cluster_size=1375)

best_model = hdbscan
predictions = hdbscan.fit_predict(tracks_df)
print(len(set(predictions)))
visualize_clusters(data=tracks_df, predictions=predictions, n_cluster=len(set(hdbscan.labels_)), stochastic=True)
best_k = len(set(hdbscan.labels_))

print("len of tracks_df: %d" % len(tracks_df))

"""
explain track clusters with original afs
"""
statement = "SELECT track_id, danceability, energy, speechiness, acousticness, instrumentalness, tempo, valence, liveness FROM acoustic_features"
af_df = pd.read_sql(sql=statement, con=engine).set_index("track_id")
scaler = MinMaxScaler()
af_df = pd.DataFrame(scaler.fit_transform(af_df), columns=af_df.columns, index=af_df.index)

df = tracks_df.merge(af_df, left_index=True, right_index=True)
df = df.merge(pd.DataFrame(data=predictions, index=tracks_df.index, columns=["cluster"]), left_index=True, right_index=True).dropna()
print(df.groupby(by="cluster").count())


features_df = df[["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "tempo", "valence", "liveness"]]
features_df.columns = ["Danceability", "Energy", "Speechiness", "Acousticness", "Instrumentalness", "Tempo", "Valence", "Liveness"]
features_df = features_df.stack().reset_index()
features_df.columns = ["track_id", "acoustic_features", "value"]

features_df = features_df.merge(df["cluster"], left_on="track_id", right_index=True)
features_df.set_index("track_id", inplace=True)

features_df = features_df[features_df["cluster"] != -1]
features_df["cluster"] += 1
print(features_df)

print("len of features_df: %d" % len(features_df))
print("n tracks %d" % features_df.index.nunique())



g = sns.boxplot(x="value", y="acoustic_features", hue="cluster", data=features_df, showfliers=False)
g.legend_.set_title("")
plt.legend(loc="lower right")
new_labels = [r"$C_1$", r"$C_2$", r"$C_3$", r"$C_4$"]
for t, l in zip(g.legend_.texts, new_labels):
    t.set_text(l)
plt.ylabel("")
plt.xlabel("")
plt.grid(False)
plt.show()


"""
explain track clusters
"""
track_genres_df = pd.read_csv("track_genres.csv", sep=";")

track_genres_df.columns = ["track_id", "genres"]
track_genres_df.set_index("track_id", inplace=True)
track_genres_df["genres"] = track_genres_df["genres"].apply(lambda r: ast.literal_eval(r))
print("len of track_genres: %d" % len(track_genres_df))

track_id_to_cluster_df = pd.DataFrame()
track_id_to_cluster_df["track_id"] = tracks_df.index
track_id_to_cluster_df["cluster"] = predictions

print(track_id_to_cluster_df.head())
print(track_id_to_cluster_df.groupby(by="cluster").size())

track_id_to_cluster_df.set_index("track_id", inplace=True)
track_id_to_cluster_df.to_csv("track_to_cluster.csv", sep=";")


cluster_genres_df = track_id_to_cluster_df.merge(track_genres_df, left_index=True, right_index=True).groupby(by="cluster").sum()

idf_scores_df = pd.read_csv("track_genres_idf_dist.csv", sep=";")
idf_scores_df.columns = ["track_id", "score"]
idf_scores_df.set_index("track_id", inplace=True)

print(cluster_genres_df.head())
print([cluster_genres_df.to_numpy()[row, 0] for row in range(len(cluster_genres_df))])
genres = set([cluster_genres_df.to_numpy()[row, 0] for row in range(len(cluster_genres_df))][0])


idf_scores_df = idf_scores_df[idf_scores_df["score"] > 0.9]

explanation_df = cluster_genres_df.apply(lambda r: compute_genre_idf(r[0], idf_scores_df.to_dict()["score"], n_returns=10), axis=1)
cluster_genres_df.to_csv("genres_per_cluster.csv", sep=";")
explanation_df.to_csv("track_cluster_explanation.csv", sep=";")

for cluster, _ in cluster_genres_df.iterrows():
    if cluster != -1:
        g, s = zip(*explanation_df.loc[cluster])
        plt.barh(g, s)
    plt.show()


tracks_df = pd.read_csv("lowms_tracks.csv", sep=";")
df = track_id_to_cluster_df.reset_index()
df = df.merge(tracks_df, left_on="track_id", right_on="track_id")
df.set_index(["track_id", "cluster"], inplace=True)
df.groupby("cluster").boxplot()
plt.show()


"""
content weighted
"""
labels_df = pd.DataFrame(data=predictions, index=tracks_df.index)
labels_df.columns = ["cluster"]

user_track_df = pd.read_csv("user_track.csv", sep=";", index_col="user_id", usecols=["track_id", "user_id"])
user_cluster_df = user_track_df.merge(labels_df, left_on="track_id", right_index=True)

print("n tracks user track: %d" % user_track_df["track_id"].nunique())

# ignore all boundary points
user_cluster_df = user_cluster_df[user_cluster_df["cluster"] != -1]
dist_tracks_in_group_df = user_cluster_df.groupby(by=["user_id", "cluster"])["track_id"].nunique()

# weighted mean
weights_df = dist_tracks_in_group_df / dist_tracks_in_group_df.groupby(by="user_id").sum()

weights_df = weights_df.reset_index()
weights_df.columns = ["user_id", "cluster", "weight"]

df = weights_df.copy()
df = df.pivot(index="user_id", columns="cluster", values="weight")
df.boxplot()
plt.show()

weights_df.to_csv("weights.csv", sep=";", index=False)




# TODO: try listening events instead of tracks
user_events_df = pd.read_csv("lowms_les.csv", sep=";", usecols=["user_id", "track_id"])
user_cluster_df = user_events_df.merge(labels_df, left_on="track_id", right_index=True)

print(user_events_df["user_id"].nunique())

# ignore all boundary points
user_cluster_df = user_cluster_df[user_cluster_df["cluster"] != -1]

events_in_group_df = user_cluster_df.groupby(by=["user_id", "cluster"])["track_id"].count()
weights_df = events_in_group_df / events_in_group_df.groupby(by="user_id").sum()
weights_df = weights_df.reset_index()
weights_df.columns = ["user_id", "cluster", "weight"]

weights_df.to_csv("weights_events.csv", sep=";", index=False)

df = weights_df.copy()
df = df.pivot(index="user_id", columns="cluster", values="weight")
df.boxplot()
plt.title("weighted on events")
plt.show()