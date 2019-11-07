import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from itertools import combinations
import matplotlib.cm as cm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
import umap
from sklearn.manifold import SpectralEmbedding


def inverse_scalling(df, scaler):
    df_inv = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns, index=df.index)
    return df_inv


def get_random_combination(data, n_samples, n_elements):
    #samples = [np.random.choice(data, size=n_elements, replace=True) for _ in range(n_samples)]
    random_idxs = [np.random.randint(0, len(data), size=n_elements) for _ in range(n_samples)]
    samples = np.array(random_idxs)

    return samples

def get_subsets_up_to_k(data, max_k):
    subsets = []
    for k in range(2, max_k + 1):
        features = [list(tup) for tup in combinations(data, k)]
        samples_idxs = [np.random.randint(0, len(features)) for _ in range(10)]
        samples = np.array(features)[samples_idxs]
        subsets.extend(samples.tolist())
    return subsets

def visualize_clusters(data, predictions, n_cluster, stochastic=True):
    ndim = len(data.columns)
    if ndim > 2:
        if stochastic:
            reducer = umap.UMAP()
        else:
            reducer = PCA(n_components=2)

        top2_pc = reducer.fit_transform(data)
    else:
        top2_pc = np.array(data)


    groups = []
    for c in range(-1, n_cluster-1):
        points_within_group_c = top2_pc[predictions == c]
        groups.append(points_within_group_c)

    clusters = [r"$C_" + str(i + 1) + "$" for i in range(-1, n_cluster-1)]

    fig = plt.figure(frameon=False)
    for datapoints, cluster in zip(groups, clusters):
        if cluster == r"$C_0$":
            continue
        xs, ys = datapoints[:, 0], datapoints[:, 1]
        plt.scatter(xs, ys, label=cluster, alpha=0.7, edgecolors="none", s=30)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()


def explain_clusters(data, predictions, n_cluster):
    groups = []
    overall_expl = dict()
    for c in range(n_cluster):
        points_within_group_c = np.array(data)[predictions == c]
        #groups.append(points_within_group_c)

        df = pd.DataFrame(points_within_group_c, columns=data.columns)
        del df["track_id"]

        groups.append(df.describe())

        df.hist(bins=30)
        plt.show()

        d = df.describe().to_dict()
        overall_expl["C" + str(c)] = d
    """means_per_group = np.array([np.mean(c, axis=0).tolist() for c in groups])
    return means_per_group"""

    return overall_expl



def plot_silhouette(data, metric, predictions, k):
    if data.ndim == 1:
        data = data.to_frame()

    score_per_sample = silhouette_samples(data, predictions, metric=metric)
    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = \
            score_per_sample[predictions == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    score = silhouette_score(data, predictions, metric=metric)
    plt.axvline(x=score, color="red", linestyle="--")

    plt.yticks([])
    plt.xticks(np.arange(-1, +1, 0.1))
    plt.show()


def normalize(data, min=0, max=1):
    s = (max - min) * ((data - data.min()) / (data.max() - data.min())) + min
    return s

def gap_statistic(disp, random_data, k):
    random_model = KMeans(n_clusters=k).fit(random_data)
    random_disp = random_model.inertia_

    gap = np.log(np.mean(random_disp)) - np.log(disp)
    return gap

def sse(data, predictions, cluster_means):
    errors = []
    for i in range(len(data)):
        observation = data.iloc[i]
        cluster = predictions[i]

        errors.append(np.power(observation.values - cluster_means.loc[cluster].values, 2))

    return errors

def chunkify(df, size):
    list_df = [df[i:i + size] for i in range(0, len(df), size)]
    return list_df




