import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from surprise import Prediction
from surprise import AlgoBase
from scipy.stats import ks_2samp

class PowerLawPredictor(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ratings = [r for _, _, r in trainset.all_ratings()]

        self._model = powerlaw.Fit(ratings)
        self._alpha = self._model.power_law.alpha
        self._xmin = self._model.power_law.xmin
        self._D = self._model.power_law.D
        self._std = self._model.power_law.sigma

    def estimate(self, u, i):
        est = self._model.power_law.generate_random(1)
        return est[0]

    def goodness_of_fit(self):
        print("Kolmogorov Smirnov gives D: %f" % self._D)
        print("Standard Deviation of fitted curve is %f" % self._std)

        return self._D, self._std



    @property
    def alpha(self):
        return self._alpha

    @property
    def xmin(self):
        return self._xmin