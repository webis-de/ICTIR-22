import pandas as pd
import numpy as np


class PairwiseAggregator:
    def __init__(self, threshold, margin, log_scores=True, logit_scores=False):
        self.threshold = threshold
        self.margin = margin
        self.log_scores = log_scores
        self.logit_scores = logit_scores

    def _infer_tie(self, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold + self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold + self.margin) +
                             '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold - self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold - self.margin) +
                             '. Expected p in Interval [0, 1]')

        if p > self.threshold + self.margin:
            return False
        elif p < self.threshold - self.margin:
            return False
        else:
            return True

    def _order_pair(self, id_a, id_b, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')
        if p >= self.threshold:
            return id_a, id_b, p
        else:
            return id_b, id_a, 1 - p

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        if self.log_scores:
           pairwise_scores["score"] = pairwise_scores["score"].apply(np.exp)
        if self.logit_scores:
            pairwise_scores["score"] = pairwise_scores["score"].apply(lambda x: np.exp(x)/(1+np.exp(x)))
        return pairwise_scores

        """"
        data = []
        if self.log_scores:
            for _, (id_a, id_b, p) in pairwise_scores.iterrows():
                data.append(self._order_pair(id_a, id_b, np.exp(p)))
        else:
            for _, (id_a, id_b, p) in pairwise_scores.iterrows():
                data.append(self._order_pair(id_a, id_b,p))
        return pd.DataFrame(data, columns=pairwise_scores.columns)
        """