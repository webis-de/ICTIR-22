import pandas as pd
import numpy as np
from .base import PairwiseAggregator


class KwikSortAggregator(PairwiseAggregator):
    def __init__(self, pivot_selection_function: str = "random", log_scores: bool = True, logit_scores: bool = False):
        super().__init__(0.5, 0, log_scores=log_scores, logit_scores=logit_scores)
        if pivot_selection_function == "random":
            self._select_pivot = self._random_pivot
        elif pivot_selection_function == "rowsum":
            self._select_pivot = self._select_pivot_by_rowsum
        elif pivot_selection_function == "misranking":
            self._select_pivot = self._select_pivot_by_misrankings
        else:
            raise ValueError("pivot_selection_function must be random, rowsum, or misranking, got: {}".format(pivot_selection_function))
        self._index_type = np.int64

    def __str__(self):
        return "kwiksort"

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        pairwise_scores = super().__call__(pairwise_scores)
        pairs = []
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            pairs.append(self._order_pair(id_a, id_b, p))
        pairwise_scores = pd.DataFrame(pairs, columns=["id_a", "id_b", "score"])

        items = list(set(pairwise_scores["id_a"].unique().tolist() + pairwise_scores["id_b"].unique().tolist()))
        item_mapping = {x: i for i, x in enumerate(items)}
        pref_matrix = np.zeros(shape=(len(item_mapping), len(item_mapping)), dtype=np.int64)

        # Mapped comparisons
        for _, (id_a, id_b, _) in pairwise_scores.iterrows():
            pref_matrix[item_mapping[id_a], item_mapping[id_b]] = 1

        order = self._call(np.arange(len(items), dtype=self._index_type), pref_matrix)
        # Invert scores to be descending for standard run sorting
        order = [len(items) - x for x in order]
        return pd.DataFrame(list(zip(items, order)), columns=["docno", "score"])

    def _call(self, V, A):
        if V.size == 0:
            return V

        Vl = np.empty(len(V) - 1, dtype=self._index_type)
        Vr = np.empty(len(V) - 1, dtype=self._index_type)
        il = 0
        ir = 0

        pivot = self._select_pivot(V, A)

        for j in V:
            if j == pivot:
                continue

            if A[j, pivot] >= A[pivot, j]:
                Vl[il] = j
                il += 1
            else:
                Vr[ir] = j
                ir += 1

        Vl = Vl[:il]
        Vr = Vr[:ir]

        return np.concatenate((self._call(Vl, A), [pivot], self._call(Vr, A)))

    def _random_pivot(self, V, A):
        return V[np.random.randint(0, V.size)]

    def _select_pivot_by_rowsum(self, V, A):
        rowsums = A[V, :].sum(axis=1)
        median_idx = np.argsort(rowsums)[rowsums.size//2]
        return V[median_idx]

    def _select_pivot_by_misrankings(self, V, A):
        misrank = np.zeros_like(V)
        Vl = np.empty(len(V) - 1, dtype=self._index_type)
        Vr = np.empty(len(V) - 1, dtype=self._index_type)

        for i, v in enumerate(V):
            il = 0
            ir = 0
            for v1 in V:
                if v1 == v:
                    continue
                if A[v1, v] >= A[v, v1]:
                    Vl[il] = v1
                    il += 1
                else:
                    Vr[ir] = v1
                    ir += 1
            for j in range(0, il):
                for k in range(0, ir):
                    m = Vr[k] - Vl[j]
                    misrank[i] += max(0, m)
        least_misranking = np.argsort(misrank)[0]
        return V[least_misranking]
