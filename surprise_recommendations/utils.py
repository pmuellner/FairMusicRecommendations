from collections import defaultdict
import pandas as pd
import sqlalchemy
from surprise import accuracy
import numpy as np
from scipy.stats import ttest_ind, f_oneway, shapiro, levene, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from sklearn.preprocessing import MinMaxScaler
from surprise_recommendations.evaluation import top_k_measures
from collections import defaultdict
import powerlaw
import matplotlib.pyplot as plt
from time import time

MIN_PER_HOUR = 3600

SQL_CREDENTIALS = "root:1234"
engine = sqlalchemy.create_engine('mysql+pymysql://' + SQL_CREDENTIALS + '@localhost:3306/music_recommender_db')

def scale(data, range=(0, 1000)):
    scaler = MinMaxScaler(feature_range=range)
    scaled = scaler.fit_transform(data.reshape(-1, 1).astype(float))
    return scaled


def _get_main_groups(verbose=True):
    low_ms = pd.read_csv("data/low_main_users.txt")["user_id"].values.tolist()

    stmt = "SELECT user_id FROM users"
    all_ms = pd.read_sql(con=engine, sql=stmt)["user_id"].values.tolist()

    norm_ms = [user_id for user_id in all_ms if user_id not in low_ms]

    if verbose:
        print("|all|: %d" % len(all_ms))
        print("|norm_ms|: %d" % len(norm_ms))
        print("|low_ms|: %d" % len(low_ms))

    return low_ms, norm_ms, all_ms

def get_group_measures(preds_all, low_ms=None, norm_ms=None, U1=None, U2=None, U3=None, U4=None):
    predictions = defaultdict(list)
    maes = defaultdict(list)
    predictions["All"] = preds_all
    for user_id, item_id, r, r_, details in preds_all:
        mae = np.abs(r - r_)
        maes["All"].append(mae)

    if low_ms is not None and norm_ms is not None:
        for user_id, item_id, r, r_, details in preds_all:
            mae = np.abs(r - r_)
            if user_id in low_ms:
                maes["LowMs"].append(mae)
                predictions["LowMs"].append((user_id, item_id, r, r_, details))
            elif user_id in norm_ms:
                maes["NormMs"].append(mae)
                predictions["NormMs"].append((user_id, item_id, r, r_, details))


        # H0: MAE(NormMs) >= MAE(LowMs)
        t_statistic, p_val = ttest_ind(maes["NormMs"], maes["LowMs"])
        print(p_val)
        mae_ms_p = p_val / 2.0
        print("[t-TEST] t-statistic: %f, p-value: %f" % (t_statistic, mae_ms_p))

    if U1 is not None and U2 is not None and U3 is not None and U4 is not None:
        for user_id, item_id, r, r_, details in preds_all:
            mae = np.abs(r - r_)
            if user_id in U1:
                predictions["U1"].append((user_id, item_id, r, r_, details))
                maes["U1"].append(mae)
            elif user_id in U2:
                predictions["U2"].append((user_id, item_id, r, r_, details))
                maes["U2"].append(mae)
            elif user_id in U3:
                predictions["U3"].append((user_id, item_id, r, r_, details))
                maes["U3"].append(mae)
            elif user_id in U4:
                predictions["U4"].append((user_id, item_id, r, r_, details))
                maes["U4"].append(mae)

        # H0: all means are the same
        f_val, p_val = f_oneway(maes["U1"], maes["U2"], maes["U3"], maes["U4"])
        print("[ANOVA] p-value: %f" % p_val)


        df = pd.DataFrame(data={"MAE": maes["U1"], "Usergroup": "U1"})
        df = df.append(pd.DataFrame(data={"MAE": maes["U2"], "Usergroup": "U2"}))
        df = df.append(pd.DataFrame(data={"MAE": maes["U3"], "Usergroup": "U3"}))
        df = df.append(pd.DataFrame(data={"MAE": maes["U4"], "Usergroup": "U4"}))
        df = df.append(pd.DataFrame(data={"MAE": maes["All"], "Usergroup": "All"}))

        tukeyhsd_results = pairwise_tukeyhsd(df["MAE"], df["Usergroup"])
        print(tukeyhsd_results.summary())

        # from https://stackoverflow.com/questions/48200699/how-can-i-get-p-values-of-each-group-comparison-when-applying-the-tukey-s-hones
        p_values = psturng(np.abs(tukeyhsd_results.meandiffs / tukeyhsd_results.std_pairs), len(tukeyhsd_results.groupsunique), tukeyhsd_results.df_total)
        print("[TukeyHSD] p-values : " + str(p_values))

    results = ResultDict()
    for group in predictions.keys():
        results.add_result(usergroup=group, metric="MAE", value=accuracy.mae(predictions[group], verbose=False))
        """results.add_result(usergroup=group, metric="RMSE", value=accuracy.rmse(predictions[group], verbose=False))
        results.add_result(usergroup=group, metric="FCP", value=accuracy.fcp(predictions[group], verbose=False))"""

        """for k in [5, 10, 20]:
            _, _, f1, mrr, map, ndcg = top_k_measures(predictions[group], k=k)
            #results.add_result(usergroup=group, metric="P@" + str(k), value=p)
            #results.add_result(usergroup=group, metric="R@" + str(k), value=r)
            results.add_result(usergroup=group, metric="F1@" + str(k), value=f1)
            results.add_result(usergroup=group, metric="MRR@" + str(k), value=mrr)
            results.add_result(usergroup=group, metric="MAP@" + str(k), value=map)
            results.add_result(usergroup=group, metric="nDCG@" + str(k), value=ndcg)
        for k in range(1, 21):
            p, r, _, _, _, _ = top_k_measures(predictions[group], k=k)
            results.add_result(usergroup=group, metric="P@" + str(k), value=p)
            results.add_result(usergroup=group, metric="R@" + str(k), value=r)"""

    return results


def get_bll_scores(playcounts, d=1.5):
    relevant_users = playcounts["user_id"].unique()
    relevant_items = playcounts["item_id"].unique()
    print(len(relevant_users), len(relevant_items))

    """stmt = "SELECT user_id, track_id, timestamp FROM events WHERE user_id IN " + str(tuple(relevant_users))
    events_df = pd.read_sql(con=engine, sql=stmt)
    events_df = events_df[events_df["track_id"].isin(relevant_items)]
    events_df.head()

    #events_df["t_since_le"] = pd.to_timedelta(events_df["timestamp"].max() - events_df["timestamp"], unit="s")
    #events_df["t_since_le"] += 1
    # TODO wrong!!
    events_df["t_since_le"] = pd.to_timedelta(time() - events_df["timestamp"], unit="s")
    events_df["t_since_le"] = events_df["t_since_le"] / np.timedelta64(1, "h")

    events_df["arg"] = events_df["t_since_le"].apply(lambda v: v ** -d)
    bll_df = events_df.groupby(by=["user_id", "track_id"])["arg"].sum().apply(np.log).reset_index()
    bll_df.columns = ["user_id", "track_id", "bll"]

    normalization_df = bll_df.copy()
    normalization_df["exp_bll"] = normalization_df["bll"].apply(np.exp)
    normalization_df.set_index("user_id", inplace=True)
    normalization_df["normalization"] = normalization_df.groupby("user_id")["exp_bll"].sum()
    normalization_df["score"] = normalization_df["exp_bll"] / normalization_df["normalization"]

    scores_df = normalization_df.reset_index()[["user_id", "track_id", "score"]]"""


    # TODO just for testing
    scores_df = playcounts.drop_duplicates(subset=["user_id", "item_id"])
    scores_df["rating"] = 1
    scores_df.columns = ["user_id", "track_id", "score"]
    return scores_df


class ResultDict():
    def __init__(self):
        self._results = dict()

    def add_result(self, usergroup=None, metric=None, value=None):
        if usergroup and metric and value:
            if usergroup not in self._results:
                self._results[usergroup] = defaultdict(list)
            self._results[usergroup][metric].append(value)
        else:
            return None

    def get_value(self, usergroup=None, metric=None):
        if usergroup and metric:
            return self._results[usergroup][metric]
        else:
            return None

    @property
    def dict(self):
        return self._results

    @classmethod
    def aggregate(cls, list_of_results):
        result = ResultDict()
        for rdict in list_of_results:
            for group in rdict.dict:
                for metric in rdict.dict[group]:
                    value = rdict.get_value(usergroup=group, metric=metric)
                    result.add_result(usergroup=group, metric=metric, value=value)

        for group in result.dict:
            for metric in result.dict[group]:
                mean = np.mean(result.dict[group][metric])
                std = np.std(result.dict[group][metric])
                result.dict[group][metric] = {"mean": mean, "std": std}

        return result.dict
