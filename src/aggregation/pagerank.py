import pandas as pd
import networkx as nx
from .base import PairwiseAggregator


class PageRankAggregator(PairwiseAggregator):
    def __init__(self, tie_margin: float = 0, log_scores: bool = True, logit_scores: bool = False):
        super().__init__(0.5, tie_margin,  log_scores=log_scores, logit_scores=logit_scores)

    def __str__(self):
        return "pagerank"

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        pairwise_scores = super().__call__(pairwise_scores)

        data = []
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            data.append(self._order_pair(id_a, id_b, p))
        pairwise_scores = pd.DataFrame(data, columns=pairwise_scores.columns)

        graph = nx.DiGraph()
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            # invert edge for correct results
            graph.add_edge(id_b, id_a, weight=p)

        return pd.DataFrame(nx.pagerank(graph, weight="weight").items(), columns=["docno", "score"])
