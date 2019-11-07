import numpy as np
import pandas as pd
from collections import defaultdict

def top_k_measures(predictions, k=10):
    user_ratings = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_ratings[uid].append((iid, true_r, est))

    precisions, recalls, f1s, rrs, aps, ndcgs = [], [], [], [], [], []
    for uid, rating in user_ratings.items():
        #real_topk = [iid for iid, _, _ in sorted(rating, key=lambda t: t[1])[::-1][:k]]
        real_topk = [iid for iid, _, _ in sorted(rating, key=lambda t: t[1])[::-1][:10]]
        est_topk = [iid for iid, _, _ in sorted(rating, key=lambda t: t[2])[::-1][:k]]


        p_u = precision_score(y_pred=est_topk, y_true=real_topk)
        precisions.append(p_u)

        r_u = recall_score(y_pred=est_topk, y_true=real_topk)
        recalls.append(r_u)

        f1_u = f1_score(y_pred=est_topk, y_true=real_topk)
        f1s.append(f1_u)

        rr_u = reciprocal_rank(y_pred=est_topk, y_true=real_topk)
        rrs.append(rr_u)

        ap_u = averagePrecision(y_pred=est_topk, y_true=real_topk)
        aps.append(ap_u)

        ndcg_u = ndcg(y_pred=est_topk, y_true=real_topk)
        ndcgs.append(ndcg_u)


    return np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(rrs), np.mean(aps), np.mean(ndcgs)


def precision_score(y_pred, y_true):
    s_pred = set(y_pred)
    s_targets = set(y_true)
    tp = len(s_pred.intersection(s_targets))
    fp = len(s_pred.difference(s_targets))
    """tp, fp = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] in y_true:
            tp += 1
        else:
            fp += 1"""

    return tp / (tp + fp)


def recall_score(y_pred, y_true):
    s_pred = set(y_pred)
    s_targets = set(y_true)
    tp = len(s_pred.intersection(s_targets))
    fn = len(s_targets.difference(s_pred))
    """tp, fn = 0, 0
    for i in range(len(y_true)):
        if y_true[i] in y_pred:
            tp += 1
        else:
            fn += 1"""

    return tp / (tp + fn)


def f1_score(y_pred, y_true):
    p = precision_score(y_pred=y_pred, y_true=y_true)
    r = recall_score(y_pred=y_pred, y_true=y_true)

    score = (2 * (p * r) / (p + r)) if (p + r) != 0 else 0
    return score


def reciprocal_rank(y_pred, y_true):
    recrank_sum = 0
    for i in range(len(y_pred)):
        if y_pred[i] in y_true:
            recrank_sum += 1 / (i + 1)

    return recrank_sum / len(y_true)

def precisionAtK(y_pred, y_true):
    relevant_items = [item in y_true for item in y_pred]

    precision_at_ks = []
    for k in range(len(relevant_items)):
        precision_at_k = np.mean(relevant_items[:k + 1])
        precision_at_ks.append(precision_at_k)

    return precision_at_ks

def averagePrecision(y_pred, y_true):
    precision_at_ks = precisionAtK(y_pred, y_true)

    avgPrecision = np.mean(precision_at_ks)
    return avgPrecision


def icdg(N_groundtruth):
    icdg_score = 0
    for i in range(N_groundtruth):
        relevance_of_item = 1
        icdg_score += (np.power(2, relevance_of_item) - 1) / np.log2(i + 1 + 1)
    return icdg_score


def ndcg(y_pred, y_true):
    icdg_score = icdg(N_groundtruth=len(y_true))

    dcg_score = 0
    for i in range(len(y_pred)):
        if y_pred[i] in y_true:
            relevance_of_item = 1
        else:
            relevance_of_item = 0

        dcg_score += (np.power(2, relevance_of_item) - 1) / np.log2(i + 1 + 1)

    return dcg_score / icdg_score if icdg != 0 else 0

if __name__ == "__main__":
    print(precision_score([1, 2, 3], [2, 3, 7]))