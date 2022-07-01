import unittest
import numpy as np
import pandas as pd

from aggregation import BradleyTerryAggregator
from sampling import PairwiseFullSampler


class BradleyTerryAggregatorTest(unittest.TestCase):

    def test_reconstruct_ranking(self):
        N = 50
        pointwise_scores = pd.DataFrame(zip(range(N), np.random.rand(N)), columns=["docno", "score"])
        score_dict = pointwise_scores.set_index("docno").to_dict()["score"]
        pairwise_scores = PairwiseFullSampler(method="combinations")(pointwise_scores)
        pairwise_scores["score"] = np.log(0.1)
        for _, (id_a, id_b, score) in pairwise_scores.iterrows():
            if score_dict[id_a] > score_dict[id_b]:
                pairwise_scores.loc[
                    (pairwise_scores["id_a"] == id_a) & (pairwise_scores["id_b"] == id_b), "score"] = np.log(0.9)
        res = BradleyTerryAggregator(cython=True)(pairwise_scores)
        self.assertTrue((pointwise_scores.sort_values("score", ascending=False)["docno"].values == res.sort_values("score")["docno"].values).all())

