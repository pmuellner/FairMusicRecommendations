import pandas as pd
from surprise import Dataset, Reader, KNNBasic, BaselineOnly, NormalPredictor, KNNWithMeans, NMF
from surprise.model_selection import KFold, train_test_split
from pprint import PrettyPrinter
import numpy as np
from surprise_recommendations.utils import get_group_measures, ResultDict, scale, get_bll_scores
from surprise_recommendations.estimators.TOP import TOP
from surprise_recommendations.estimators.PowerLawPredictor import PowerLawPredictor
import sqlalchemy
import sys
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

from surprise_recommendations.estimators.WNMF import WNMF
from surprise_recommendations.estimators.WKNN import WKNN
from surprise_recommendations.magic_array import DFARRAY
import pickle
import guppy
from datetime import datetime
import seaborn as sns

import matplotlib.pyplot as plt

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')


#original_stdout = sys.stdout
#f = open("results.txt", "a")
#sys.stdout = f

pp = PrettyPrinter()


if __name__ == "__main__":
    """
    Create datasets for rating prediction
    """
    """lowms_users = pd.read_csv("data/low_main_users.txt", usecols=["user_id"], squeeze=True).unique().tolist()
    n_users = len(lowms_users)
    playcounts_lowms_df = compute_listening_counts(list_of_users=lowms_users, n=n_users)
    playcounts_lowms_df.columns = ["user_id", "item_id", "rating"]
    playcounts_lowms_df.to_csv("data/playcounts_lowms.csv", sep=";", index=False)

    stmt = "SELECT user_id, M_global_R_APC, country from user_mainstreaminess WHERE M_global_R_APC > 0.097732 AND user_id NOT IN " + str(tuple(lowms_users))
    all_norm_users = pd.read_sql(con=engine, sql=stmt).dropna()["user_id"].tolist()
    playcounts_normms_df = compute_listening_counts(list_of_users=all_norm_users, n=n_users)
    normms_users = playcounts_normms_df["user_id"].unique().tolist()
    playcounts_normms_df.columns = ["user_id", "item_id", "rating"]
    playcounts_normms_df.to_csv("data/playcounts_normms.csv", sep=";", index=False)"""


    """
    Recommendation
    """
    playcounts_lowms_df = pd.read_csv("data/playcounts_lowms.csv", sep=";")
    playcounts_normms_df = pd.read_csv("data/playcounts_normms.csv", sep=";")

    scaled_playcounts_lowms_df = pd.DataFrame()
    for user_id, group in playcounts_lowms_df.groupby("user_id"):
        min_rating = group["rating"].min()
        max_rating = group["rating"].max()
        scaled_ratings = scale(group["rating"].values, range=(0, 1000))
        new_rows = group.copy()
        new_rows["rating"] = scaled_ratings
        scaled_playcounts_lowms_df = scaled_playcounts_lowms_df.append(new_rows)


    scaled_playcounts_normms_df = pd.DataFrame()
    for user_id, group in playcounts_normms_df.groupby("user_id"):
        min_rating = group["rating"].min()
        max_rating = group["rating"].max()
        scaled_ratings = scale(group["rating"].values, range=(0, 1000))
        new_rows = group.copy()
        new_rows["rating"] = scaled_ratings
        scaled_playcounts_normms_df = scaled_playcounts_normms_df.append(new_rows)

    lowms_users = scaled_playcounts_lowms_df["user_id"].unique().tolist()
    normms_users = scaled_playcounts_normms_df["user_id"].unique().tolist()

    playcounts_df = scaled_playcounts_lowms_df.append(scaled_playcounts_normms_df)
    playcounts_df = playcounts_df.sample(frac=1).reset_index(drop=True)

    lowms_sparsity = 1 - (len(playcounts_lowms_df) / (
            playcounts_lowms_df["user_id"].nunique() * playcounts_lowms_df["item_id"].nunique()))
    normms_sparsity = 1 - (len(playcounts_normms_df) / (
            playcounts_normms_df["user_id"].nunique() * playcounts_normms_df["item_id"].nunique()))
    all_sparsity = 1 - (len(playcounts_df) / (playcounts_df["user_id"].nunique() * playcounts_df["item_id"].nunique()))

    print("[LowMs] %d users, %d tracks, %d ratings, %.4f sparsity" % (playcounts_lowms_df["user_id"].nunique(),
                                                                      playcounts_lowms_df["item_id"].nunique(),
                                                                      len(playcounts_lowms_df),
                                                                      lowms_sparsity))

    print("[NormMs] %d users, %d tracks, %d ratings, %.4f sparsity" % (playcounts_normms_df["user_id"].nunique(),
                                                                       playcounts_normms_df["item_id"].nunique(),
                                                                       len(playcounts_normms_df),
                                                                       normms_sparsity))

    print("[AllMs] %d users, %d tracks, %d ratings, %.4f sparsity" % (playcounts_df["user_id"].nunique(),
                                                                      playcounts_df["item_id"].nunique(),
                                                                      len(playcounts_df),
                                                                      all_sparsity))

    classification_df = pd.read_csv("data/classification_clean.csv", sep=";")
    usergroups_df = classification_df.merge(playcounts_df, left_on="user_id", right_on="user_id")



    playcounts_U1_df = usergroups_df[usergroups_df["cluster"] == 1]
    playcounts_U2_df = usergroups_df[usergroups_df["cluster"] == 2]
    playcounts_U3_df = usergroups_df[usergroups_df["cluster"] == 3]
    playcounts_U4_df = usergroups_df[usergroups_df["cluster"] == 4]

    U1_sparsity = 1 - (
            len(playcounts_U1_df) / (playcounts_U1_df["user_id"].nunique() * playcounts_U1_df["item_id"].nunique()))
    U2_sparsity = 1 - (
            len(playcounts_U2_df) / (playcounts_U2_df["user_id"].nunique() * playcounts_U2_df["item_id"].nunique()))
    U3_sparsity = 1 - (
            len(playcounts_U3_df) / (playcounts_U3_df["user_id"].nunique() * playcounts_U3_df["item_id"].nunique()))
    U4_sparsity = 1 - (
            len(playcounts_U4_df) / (playcounts_U4_df["user_id"].nunique() * playcounts_U4_df["item_id"].nunique()))
    all_sparsity = 1 - (len(usergroups_df) / (usergroups_df["user_id"].nunique() * usergroups_df["item_id"].nunique()))

    print("[U_1] %d users, %d tracks, %d ratings, %.4f sparsity" % (playcounts_U1_df["user_id"].nunique(),
                                                                    playcounts_U1_df["item_id"].nunique(),
                                                                    len(playcounts_U1_df), U1_sparsity))
    print("[U_2] %d users, %d tracks, %d ratings, %.4f sparsity" % (playcounts_U2_df["user_id"].nunique(),
                                                                    playcounts_U2_df["item_id"].nunique(),
                                                                    len(playcounts_U2_df), U2_sparsity))
    print("[U_3] %d users, %d tracks, %d ratings, %.4f sparsity" % (playcounts_U3_df["user_id"].nunique(),
                                                                    playcounts_U3_df["item_id"].nunique(),
                                                                    len(playcounts_U3_df), U3_sparsity))
    print("[U_4] %d users, %d tracks, %d ratings, %.4f sparsity" % (playcounts_U4_df["user_id"].nunique(),
                                                                    playcounts_U4_df["item_id"].nunique(),
                                                                    len(playcounts_U4_df), U3_sparsity))
    print("[U_all] %d users, %d tracks, %d ratings, %.4f sparsity" % (usergroups_df["user_id"].nunique(),
                                                                      usergroups_df["item_id"].nunique(),
                                                                      len(usergroups_df), all_sparsity))

    U1_users = playcounts_U1_df["user_id"].unique().tolist()
    U2_users = playcounts_U2_df["user_id"].unique().tolist()
    U3_users = playcounts_U3_df["user_id"].unique().tolist()
    U4_users = playcounts_U4_df["user_id"].unique().tolist()

    """
    Run recommendations and evaluate
    """
    top_results = []
    knn_results = []
    rand_results = []
    pl_results = []
    baseline_results = []
    knnmean_results = []
    nmf_results = []
    wnmf_results = []
    wknn_results = []

    reader = Reader(rating_scale=(0, np.inf))
    data = Dataset.load_from_df(playcounts_df, reader)

    folds_it = KFold(n_splits=5).split(data)
    i = 1
    pl_fit = []
    with open("data/bll.pkl", "rb") as f:
        bll_dict = pickle.load(f)
    for trainset, testset in folds_it:
        """print("Baseline")
        baseline = BaselineOnly()
        baseline.fit(trainset)
        baseline_predictions = baseline.test(testset)
        results = get_group_measures(preds_all=baseline_predictions, low_ms=lowms_users, norm_ms=normms_users)
        baseline_results.append(results)"""

        """print("WKNN")
        wknn = WKNN(sim_options={"name": "weighted_neighbors", "weights": bll_dict})
        wknn.fit(trainset)
        wknn_predictions = wknn.test(testset)
        results = get_group_measures(preds_all=wknn_predictions, low_ms=lowms_users, norm_ms=normms_users)
        wknn_results.append(results)

        print("KNN")
        knn = KNNBasic(sim_options={"name": "cosine"})
        knn.fit(trainset)
        knn_predictions = knn.test(testset)
        results = get_group_measures(preds_all=knn_predictions, low_ms=lowms_users, norm_ms=normms_users)
        knn_results.append(results)"""

        """print("KNNMean")
        knnmean = KNNWithMeans(sim_options={"name": "cosine"})
        knnmean.fit(trainset)
        knnmean_predictions = knnmean.test(testset)
        results = get_group_measures(preds_all=knnmean_predictions, low_ms=lowms_users, norm_ms=normms_users)
        knnmean_results.append(results)"""

        """print("NMF")
        start = datetime.now()
        nmf = NMF(biased=False, verbose=True, n_epochs=10)
        nmf.fit(trainset)
        nmf_predictions = nmf.test(testset)
        results = get_group_measures(preds_all=nmf_predictions, low_ms=lowms_users, norm_ms=normms_users)
        nmf_results.append(results)
        print("--> time elapsed " + str(datetime.now() - start))"""


        print("WNMF")
        start = datetime.now()
        wnmf = WNMF(bll_dict, biased=False, verbose=True, n_epochs=50)
        wnmf.fit(trainset)
        wnmf_predictions = wnmf.test(testset)
        results = get_group_measures(preds_all=wnmf_predictions, low_ms=lowms_users, norm_ms=normms_users)
        wnmf_results.append(results)
        print("--> time elapsed " + str(datetime.now() - start))


        """print("TOP")
        top = TOP()
        top.fit(trainset)
        top_predictions = top.test(testset)
        results = get_group_measures(preds_all=top_predictions, low_ms=lowms_users, norm_ms=normms_users)
        top_results.append(results)

        print("NormalPredictor")
        rand = NormalPredictor()
        rand.fit(trainset)
        rand_predictions = rand.test(testset)
        results = get_group_measures(preds_all=rand_predictions, low_ms=lowms_users, norm_ms=normms_users)
        rand_results.append(results)

        print("PL")
        pl = PowerLawPredictor()
        pl.fit(trainset)
        D, std = pl.goodness_of_fit()
        pl_fit.append((pl.alpha, pl.xmin, D, std))
        pl_predictions = pl.test(testset)
        results = get_group_measures(preds_all=pl_predictions, low_ms=lowms_users, norm_ms=normms_users)
        pl_results.append(results)"""


    print("TOP")
    pp.pprint(ResultDict.aggregate(top_results))
    print("KNN")
    pp.pprint(ResultDict.aggregate(knn_results))
    print("RAND")
    pp.pprint(ResultDict.aggregate(rand_results))
    print("PL")
    pp.pprint(ResultDict.aggregate(pl_results))
    print("Baseline")
    pp.pprint(ResultDict.aggregate(baseline_results))
    print("KNNMean")
    pp.pprint(ResultDict.aggregate(knnmean_results))
    print("NMF")
    pp.pprint(ResultDict.aggregate(nmf_results))
    print("WNMF")
    pp.pprint(ResultDict.aggregate(wnmf_results))
    print("WKNN")
    pp.pprint(ResultDict.aggregate(wknn_results))

    #print("(alpha, xmin, D, std) for H0: observations == fitted for PL fit")
    #print(pl_fit)

    #sys.stdout = original_stdout


    """
    Create datasets for time dependent recommendation
    """
    """lowms_users = pd.read_csv("data/playcounts_lowms.csv", sep=";")["user_id"].unique().tolist()
    normms_users = pd.read_csv("data/playcounts_normms.csv", sep=";")["user_id"].unique().tolist()

    stmt = "SELECT user_id, track_id, timestamp FROM events WHERE user_id IN " + str(tuple(lowms_users))
    events_lowms_df = pd.read_sql(con=engine, sql=stmt)
    stmt = "SELECT user_id, track_id, timestamp FROM events WHERE user_id IN " + str(tuple(normms_users))
    events_normms_df = pd.read_sql(con=engine, sql=stmt)

    train_lowms_df, test_lowms_df = compute_le_split(events_df=events_lowms_df, test_size=0.01)
    train_normms_df, test_normms_df = compute_le_split(events_df=events_normms_df, test_size=0.01)

    train_lowms_df.to_csv("data/topk_train_lowms.csv", sep=";", index=False)
    test_lowms_df.to_csv("data/topk_test_lowms.csv", sep=";", index=False)
    train_normms_df.to_csv("data/topk_train_normms.csv", sep=";", index=False)
    test_normms_df.to_csv("data/topk_test_normms.csv", sep=";", index=False)"""