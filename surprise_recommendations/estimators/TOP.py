from surprise import AlgoBase, PredictionImpossible
from collections import defaultdict, OrderedDict
import numpy as np
from utils import scale
import pandas as pd

class TOP(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        all_item_ratings = defaultdict(list)
        for _, iid, r in trainset.all_ratings():
            all_item_ratings[iid].append(r)

        aggregated_item_ratings = OrderedDict()
        for iid in all_item_ratings:
            n_events = np.sum(all_item_ratings[iid])
            aggregated_item_ratings[iid] = n_events

        a = scale(np.array(list(aggregated_item_ratings.values())))
        df = pd.DataFrame(index=aggregated_item_ratings.keys(), data={"rating": a.ravel()})

        for iid, row in df.iterrows():
            r = row["rating"]
            aggregated_item_ratings[iid] = r

        sorted_playcounts = sorted(aggregated_item_ratings.items(), key=lambda t: t[1])[::-1]
        self.most_popular_items = OrderedDict(sorted_playcounts)

    def estimate(self, u, i):
        if self.trainset.knows_item(i) and self.trainset.knows_user(u):
            return self.most_popular_items[i]
        else:
            raise PredictionImpossible
