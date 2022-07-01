import random
import itertools
import warnings

import pandas as pd
import numpy as np


class PairwiseScoreSampler:
    def __init__(self):
        pass

    def __call__(self, id_frame: pd.DataFrame) -> pd.DataFrame:
        pass


class PairwiseFullSampler(PairwiseScoreSampler):
    def __init__(self, method="product"):
        """
        Constructor
        :param method: which full set to produces, can be "combinations", or "product"
        """
        if method == "product":
            self.sample_func = lambda x: itertools.product(x, repeat=2)
        elif method == "combinations":
            self.sample_func = lambda x: itertools.combinations(x, 2)
        else:
            raise ValueError("method must be 'combinations' or 'product'")
        super(PairwiseFullSampler, self).__init__()

    def __call__(self, id_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a full comparison set
        :param id_frame: pointwise ranking output for sampling, column "docno" must be present
        """
        ids = id_frame.sort_values("score").loc[:, "docno"].values.tolist()
        comparisons = list(self.sample_func(ids))
        comparisons = pd.DataFrame(comparisons, columns=["id_a", "id_b"]).sort_values(["id_a", "id_b"])
        comparisons = comparisons[comparisons["id_a"] != comparisons["id_b"]]
        return comparisons


class PairwiseWindowedSampler(PairwiseScoreSampler):
    def __init__(self, window_size: int = 1, window_dilation=1, cyclical: bool = True):
        """
        Constructor
        :param window_size: size if the sliding comparison window
        :param prior: if true, does not reshuffle items before grouping; default False
        :param cyclical: flag to specify if sliding window should terminate or wrap around when reaching the end
            of the ID list
        """
        super().__init__()
        self.window = window_size
        self.dilation = window_dilation
        if self.window < 1:
            raise ValueError("Window size must be > 1")
        self.cyclical = cyclical

    def __str__(self):
        return "structured-"+str(self.window)+{True: "cyclical", False: "non-cyclical"}[self.cyclical]

    def __call__(self, id_frame: pd.DataFrame) -> pd.DataFrame:
        ids = id_frame.sort_values("score").loc[:, "docno"].values.tolist()
        if len(ids) < self.window:
            raise ValueError(f"Window size larger than supplied ID list. Given: {self.window}, minimum required: {len(ids)} ")
        comparisons = []
        for i, id_a in enumerate(ids):
            for j in range(1, self.window+1):
                z = (i + (j * self.dilation)) % len(ids)
                comparisons.append([id_a, ids[z]])

        df = pd.DataFrame(comparisons, columns=["id_a", "id_b"]).sort_values(["id_a", "id_b"])
        return df[df["id_a"] != df["id_b"]]


class BiasedWindowedSampler(PairwiseScoreSampler):
    def __init__(self, window_size: int = 1, window_dilation=2, cyclical: bool = True):
        """
        Constructor
        :param window_size: size if the sliding comparison window
        :param prior: if true, does not reshuffle items before grouping; default False
        :param cyclical: flag to specify if sliding window should terminate or wrap around when reaching the end
            of the ID list
        """
        super().__init__()
        self.window = window_size
        self.dilation = window_dilation
        if self.window < 1:
            raise ValueError("Window size must be > 1")
        self.cyclical = cyclical

    def __str__(self):
        return "structured-" + str(self.window) + {True: "cyclical", False: "non-cyclical"}[self.cyclical]

    def __call__(self, id_frame: pd.DataFrame) -> pd.DataFrame:
        ids = id_frame.sort_values("score").loc[:, "docno"].values.tolist()
        if len(ids) < self.window:
            raise ValueError(
                f"Window size larger than supplied ID list. Given: {self.window}, minimum required: {len(ids)} ")
        comparisons = []
        for i, id_a in enumerate(ids):
            for j in np.unique((np.logspace(1, self.dilation, self.window) / 2).astype(int) + 1):
                z = (i + j) % len(ids)
                comparisons.append([id_a, ids[z]])

        df = pd.DataFrame(comparisons, columns=["id_a", "id_b"]).sort_values(["id_a", "id_b"])
        return df[df["id_a"] != df["id_b"]]


class PairwiseRandomSampler(PairwiseScoreSampler):
    def __init__(self, frac: float = 1, l: float = 0):
        """
        Constructor
        :param frac: sampling fraction
        :param prior: if true, does not reshuffle items before grouping; default False
        :param agg: aggregation function for pointwise scores; default "max"
        """
        super().__init__()
        self.frac = frac
        if self.frac <= 0 or self.frac > 1:
            raise ValueError("Fraction must be 0 < f <= 1")
        self.l = l
        #if self.l < 0:
        #    raise ValueError("Lambda must be >= 0")

    def __str__(self):
        return "random-"+str(self.frac)+"-"+str(self.l)

    def __call__(self, id_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly samples a comparison subset
        :param id_frame: pointwise ranking output for sampling, column "docno" and "score" must be present
        """
        comparisons = PairwiseFullSampler(method="product")(id_frame)
        if self.l == 0:
            df = comparisons.groupby("id_a").sample(frac=self.frac).sort_values(["id_a", "id_b"])
            return df[df["id_a"] != df["id_b"]]
        else:
            # Calculate rank divergence
            weights = np.power(
                comparisons
                .merge(id_frame, how="left", left_on="id_a", right_on="docno")
                .merge(id_frame, how="left", left_on="id_b", right_on="docno")
                .assign(divergence=lambda df: abs(df["score_x"] - df["score_y"]))
                .loc[:, "divergence"]
                .values,
                self.l
            )
            df = comparisons.sample(frac=self.frac, weights=weights).sort_values(["id_a", "id_b"])
            return df[df["id_a"] != df["id_b"]]



class BiasedRandomSampler(PairwiseScoreSampler):
    def __init__(self, frac: float = 1, spread: float = 0, bias: float=0):
        """
        Constructor
        :param frac: sampling fraction
        :param prior: if true, does not reshuffle items before grouping; default False
        :param agg: aggregation function for pointwise scores; default "max"
        """
        super().__init__()
        self.frac = frac
        if self.frac <= 0 or self.frac > 1:
            raise ValueError("Fraction must be 0 < f <= 1")
        self.spread = spread
        self.bias = bias
        #if self.l < 0:
        #    raise ValueError("Lambda must be >= 0")

    def __str__(self):
        return f"randomB-{self.frac}-{self.spread}-{self.bias}"

    def __call__(self, id_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly samples a comparison subset
        :param id_frame: pointwise ranking output for sampling, column "docno" and "score" must be present
        """
        comparisons = PairwiseFullSampler(method="product")(id_frame)
        comparisons = (
            comparisons
            .assign(weight=(
                comparisons
                .merge(id_frame, how="left", left_on="id_a", right_on="docno")
                .merge(id_frame, how="left", left_on="id_b", right_on="docno")
                .assign(divergence=lambda df: np.power(1 - abs(df["score_x"] - df["score_y"]) / len(id_frame), self.spread))
                .assign(bias=lambda df: np.power(((df["score_x"] + df["score_y"]) / (2*len(id_frame))), self.bias))
                .assign(weight=lambda df: df["divergence"] * df["bias"])
                .loc[:, "weight"]
                .values
            ))
        )
        df = []
        for id_a in comparisons["id_a"].unique():
            subset = comparisons[comparisons["id_a"] == id_a]
            df.append(subset.sample(frac=self.frac, weights=subset["weight"]).sort_values(["id_a", "id_b"]).loc[:, ["id_a", "id_b"]])
        df = pd.concat(df)
        return df[df["id_a"] != df["id_b"]]


class PairwiseGroupedSampler(PairwiseScoreSampler):
    def __init__(self, k: int, strict: bool = False, prior: bool = False, cyclical: bool = False):
        """
        Constructor
        :param k: number of groups
        :param strict: test if k is a natural factor of item count, if not raise error; default False
        :param prior: if true, does not reshuffle items before grouping; default False
        :param cyclical: if true, performs cyclical sampling
        """
        super().__init__()
        self.prior = prior
        self.k = k
        self.strict = strict
        self.cyclical = cyclical

    def __call__(self, id_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Samples a (cyclical) grouped comparison subset
        :param id_frame: pointwise ranking output for sampling, columns "docno" and "score" must be present
        """
        ids = id_frame.sort_values("score").loc[:, "docno"].values.tolist()
        if self.strict and (len(ids) % self.k) != 0:
            raise ValueError('Group number needs to be a natural factor of item count')
        if self.strict and len(ids) < self.k:
            raise ValueError("Group number needs to be larger than item count")
        elif len(ids) < self.k:
            self.k = len(ids)
        if not self.prior:
            random.shuffle(ids)

        comparisons = []
        # Split the items into chunks
        p = int(len(ids) / self.k)
        groups = [ids[i:i + p] for i in range(0, len(ids), p)]
        # Complete comparisons per group
        for group in groups:
            #comparisons.extend(itertools.combinations(group, 2))
            comparisons.extend(itertools.product(group, group))
        # Comparisons of neighboring groups
        for i in range(1, len(groups)):
            comparisons.extend(itertools.product(groups[i], groups[i - 1]))
        # Comparisons first and last (if cyclical)
        if self.cyclical:
            comparisons.extend(itertools.product(groups[0], groups[-1]))
        # Order comparisons by id
        df = pd.DataFrame(comparisons, columns=["id_a", "id_b"]).sort_values(["id_a", "id_b"])
        return df[df["id_a"] != df["id_b"]]
