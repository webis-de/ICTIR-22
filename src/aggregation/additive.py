import pandas as pd
from .base import PairwiseAggregator


class AdditiveAggregator(PairwiseAggregator):
    def __init__(self, log_scores: bool = False, logit_scores = True):
        super().__init__(-1, -1,  log_scores=log_scores, logit_scores=logit_scores)

    def __str__(self):
        return "additive"

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        scores = {}
        for _, (id_a, id_b, score) in pairwise_scores.iterrows():
            try:
                scores[id_a] += score
            except KeyError:
                scores[id_a] = score
            try:
                scores[id_b] += (1 - score)
            except KeyError:
                scores[id_b] = (1 - score)
        return pd.DataFrame(scores.items(), columns=["docno", "score"])
