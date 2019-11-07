import pandas as pd
import numpy as np
from surprise import Reader, Dataset, KNNBasic, NormalPredictor, BaselineOnly, KNNWithMeans, NMF
from surprise.model_selection import KFold
from surprise_recommendations.estimators.TOP import TOP
from surprise_recommendations.estimators.PowerLawPredictor import  PowerLawPredictor
from surprise_recommendations.utils import scale, get_group_measures, ResultDict
from surprise_recommendations.construct_datasets import compute_listening_counts
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)

if __name__ == "__main__":
    """
    Create datasets for rating prediction
    """
    """lowms_users = pd.read_csv("data/low_main_users.txt", usecols=["user_id"], squeeze=True).unique().tolist()
    n_users = len(lowms_users)
    playcounts_lowms_df = compute_listening_counts(list_of_users=lowms_users, n=n_users)
    playcounts_lowms_df.columns = ["user_id", "item_id", "rating"]
    playcounts_lowms_df.to_csv("data/playcounts_lowms.csv", sep=";", index=False)"""

    playcounts_lowms_df = pd.read_csv("data/playcounts_lowms.csv", sep=";")
    scaled_playcounts_lowms_df = pd.DataFrame()
    for user_id, group in playcounts_lowms_df.groupby("user_id"):
        min_rating = group["rating"].min()
        max_rating = group["rating"].max()
        scaled_ratings = scale(group["rating"].values, range=(0, 1000))
        new_rows = group.copy()
        new_rows["rating"] = scaled_ratings
        scaled_playcounts_lowms_df = scaled_playcounts_lowms_df.append(new_rows)

    playcounts_df = scaled_playcounts_lowms_df.sample(frac=1).reset_index(drop=True)

    classification_df = pd.read_csv("data/classification_clean.csv", sep=";")
    usergroups_df = classification_df.merge(playcounts_df, left_on="user_id", right_on="user_id")

    playcounts_U1_df = usergroups_df[usergroups_df["cluster"] == 1]
    playcounts_U2_df = usergroups_df[usergroups_df["cluster"] == 2]
    playcounts_U3_df = usergroups_df[usergroups_df["cluster"] == 3]
    playcounts_U4_df = usergroups_df[usergroups_df["cluster"] == 4]

    U1_sparsity = 1 - (len(playcounts_U1_df) / (playcounts_U1_df["user_id"].nunique() * playcounts_U1_df["item_id"].nunique()))
    U2_sparsity = 1 - (len(playcounts_U2_df) / (playcounts_U2_df["user_id"].nunique() * playcounts_U2_df["item_id"].nunique()))
    U3_sparsity = 1 - (len(playcounts_U3_df) / (playcounts_U3_df["user_id"].nunique() * playcounts_U3_df["item_id"].nunique()))
    U4_sparsity = 1 - (len(playcounts_U4_df) / (playcounts_U4_df["user_id"].nunique() * playcounts_U4_df["item_id"].nunique()))
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

    reader = Reader(rating_scale=(0, np.inf))
    data = Dataset.load_from_df(usergroups_df[["user_id", "item_id", "rating"]], reader)
    folds_it = KFold(n_splits=5).split(data)
    i = 1
    pl_fit = []
    for trainset, testset in folds_it:
        print("Fold: %d" % i)
        i += 1

        print("Baseline")
        baseline = BaselineOnly()
        baseline.fit(trainset)
        baseline_predictions = baseline.test(testset)
        results = get_group_measures(preds_all=baseline_predictions, U1=U1_users, U2=U2_users, U3=U3_users, U4=U4_users)
        baseline_results.append(results)

        print("KNN")
        knn = KNNBasic(sim_options={"name": "pearson"})
        #knn = KNNBasic(sim_options={"name": "cosine"})
        knn.fit(trainset)
        knn_predictions = knn.test(testset)
        results = get_group_measures(preds_all=knn_predictions, U1=U1_users, U2=U2_users, U3=U3_users, U4=U4_users)
        knn_results.append(results)

        print("KNNMean")
        knnmean = KNNWithMeans(sim_options={"name": "cosine"})
        knnmean.fit(trainset)
        knnmean_predictions = knnmean.test(testset)
        results = get_group_measures(preds_all=knnmean_predictions, U1=U1_users, U2=U2_users, U3=U3_users, U4=U3_users)
        knnmean_results.append(results)

        print("NMF")
        nmf = NMF()
        nmf.fit(trainset)
        nmf_predictions = nmf.test(testset)
        results = get_group_measures(preds_all=nmf_predictions, U1=U1_users, U2=U2_users, U3=U3_users, U4=U3_users)
        nmf_results.append(results)

        """print("TOP")
        top = TOP()
        top.fit(trainset)
        top_predictions = top.test(testset)
        results = get_group_measures(preds_all=top_predictions, U1=U1_users, U2=U2_users, U3=U3_users, U4=U4_users)
        top_results.append(results)

        print("NormalPredictor")
        rand = NormalPredictor()
        rand.fit(trainset)
        rand_predictions = rand.test(testset)
        results = get_group_measures(preds_all=rand_predictions, U1=U1_users, U2=U2_users, U3=U3_users, U4=U4_users)
        rand_results.append(results)

        print("PL")
        pl = PowerLawPredictor()
        pl.fit(trainset)
        D, std = pl.goodness_of_fit()
        pl_fit.append((pl.alpha, pl.xmin, D, std))
        pl_predictions = pl.test(testset)
        results = get_group_measures(preds_all=pl_predictions, U1=U1_users, U2=U2_users, U3=U3_users, U4=U4_users)
        pl_results.append(results)"""

    print("TOP")
    pp.pprint(ResultDict.aggregate(top_results))
    print("KNN")
    pp.pprint(ResultDict.aggregate(knn_results))
    print("NORM")
    pp.pprint(ResultDict.aggregate(rand_results))
    print("PL")
    pp.pprint(ResultDict.aggregate(pl_results))
    print("Baseline")
    pp.pprint(ResultDict.aggregate(baseline_results))
    print("KNNMean")
    pp.pprint(ResultDict.aggregate(knnmean_results))
    print("NMF")
    pp.pprint(ResultDict.aggregate(nmf_results))

    print("(alpha, xmin, D, std) for H0: observations == fitted for PL fit")
    print(pl_fit)









