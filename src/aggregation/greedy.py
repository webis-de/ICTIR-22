import pandas as pd
import numpy as np
from .base import PairwiseAggregator


class GreedyAggregator(PairwiseAggregator):
    """

    References
    ----------
    William W. Cohen, Robert E. Schapire, Yoram Singer Learning to Order Things. J. Artif. Intell. Res. 10: 243-270 (1999)

    """

    def __init__(self, log_scores: bool = True, logit_scores: bool = False):
        super().__init__(0.5, 0, log_scores=log_scores, logit_scores=logit_scores)

    def __str__(self):
        return "greedy"

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        pairwise_scores = super().__call__(pairwise_scores)

        items = list(set(pairwise_scores["id_a"].unique().tolist() + pairwise_scores["id_b"].unique().tolist()))
        item_mapping = {x: i for i, x in enumerate(items)}

        # Construct score lookup
        scores = np.zeros(shape=(len(items), len(items)), dtype=np.float32)
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            scores[item_mapping[id_a], item_mapping[id_b]] = p

        # Calculate initial score for each item
        pi_v = np.zeros(shape=len(items), dtype=np.float32)
        for v in range(pi_v.shape[0]):
            pi_v[v] = np.sum(scores[v, :]) - np.sum(scores[:, v])

        # Initialize ranks
        ranks = np.zeros(shape=len(items), dtype=np.int32)

        for i in range(ranks.shape[0]):
            # Choose remaining item with the highest potential
            t = np.argmax(np.where(ranks == 0, pi_v, -np.inf))
            # Assign rank to t (inverted ranks to qualify as descendingly sortable scores)
            ranks[t] = len(ranks) - i
            # Adjust remaining scores
            for v in np.where(ranks == 0)[0]:
                pi_v[v] = pi_v[v] + scores[t, v] - scores[v, t]

        return pd.DataFrame(zip(item_mapping.keys(), ranks), columns=["docno", "score"])
