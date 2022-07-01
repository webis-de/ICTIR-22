import itertools

import pandas as pd
import numpy as np

from tqdm import tqdm
from trectools import TrecRun

from ..sampling import PairwiseScoreSampler, PairwiseFullSampler
from ..aggregation import PairwiseAggregator


class PointwiseRunFile(pd.DataFrame):
    def __init__(self, path, sep=" ", depth=None):
        df = (
            pd.read_csv(path, sep=sep, header=None)
            .rename({
                0: "qid",
                2: "docno",
                3: "rank",
                4: "score"
            }, axis=1)
            .drop([1, 5], axis=1)
            .sort_values(["qid", "rank"])
            .astype({
                "qid": int,
                "docno": str,
                "rank": int,
                "score": float
            })
        )

        if depth is not None:
            super().__init__(df.head(depth))
        else:
            super().__init__(df)

    def get_topics(self):
        return self.qid.unique()


class PairwiseCache:
    def __init(self):
        self.cache = None
        self.interpolated = None

    def get(self, qid, id_a, id_b):
        res = self.cache.loc[
            (self.cache.id_a == str(id_a)) &
            (self.cache.id_b == str(id_b)) &
            (self.cache.qid == str(qid)),
            "score"
        ]
        if len(res) != 1:
            return np.nan
        else:
            return res.values[0]

    def get_batch(self, other: pd.DataFrame, qid: str) -> pd.DataFrame:
        return pd.merge(
            other.astype({"id_a": str, "id_b": str}),
            self.cache[self.cache.qid == qid],
            how="left",
            left_on=["id_a", "id_b"],
            right_on=["id_a", "id_b"]
        ).dropna(subset=["score"])

    def get_interpolated(self, other: pd.DataFrame, qid: str) -> pd.DataFrame:
        return pd.merge(
            other.astype({"id_a": str, "id_b": str}),
            self.interpolated[self.interpolated.qid == qid],
            how="left",
            left_on=["id_a", "id_b"],
            right_on=["id_a", "id_b"]
        ).dropna(subset=["score"])


    def run(self, tag, pointwise_scores, sampler: PairwiseScoreSampler, aggregator: PairwiseAggregator,
            depth: int = 50, qids = None, interpolate: bool = False, verbose: bool = False) -> (TrecRun, int):
        topics = []
        c = []
        if qids is None:
            qids = pointwise_scores.get_topics()
        if verbose:
            from tqdm import tqdm
            qids = tqdm(qids)

        for qid in qids:
            q_pointwise = pointwise_scores[pointwise_scores["qid"] == qid].sort_values("score", ascending=False).head(depth)
            if len(q_pointwise) < depth:
                if verbose:
                    qids.set_description(f"Skipped topic {qid}, only {len(q_pointwise)} documents found for minimum depth {depth}")
                continue

            comparisons = (
                sampler(q_pointwise)
                .assign(qid=qid)
            )

            full = (
                PairwiseFullSampler()(q_pointwise)
                .assign(qid=qid)
            )

            if verbose:
                qids.set_description(f"Topic: {qid}, {len(comparisons)} comparisons")
            c.append(len(comparisons))

            if interpolate:
                scores = (
                    pd.concat([
                        self.get_batch(comparisons, str(qid)).loc[:, ["id_a", "id_b", "score"]],
                        self.get_interpolated(full, str(qid)).loc[:, ["id_a", "id_b", "score"]]
                    ])
                    .drop_duplicates(subset=["id_a", "id_b"], keep="first")
                    .reset_index(drop=True)
                )
            else:
                scores = self.get_batch(comparisons, str(qid)).loc[:, ["id_a", "id_b", "score"]]

            topics.append(
                aggregator(scores)
                .assign(qid=qid)
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
                .reset_index()
                .rename({"index": "rank"}, axis=1)
            )
        r = TrecRun()
        r.run_data = (
            (
                pd.concat(topics)
                .assign(q0=0)
                .assign(system=tag)
                .loc[:, ["qid", "q0", "docno", "rank", "score", "system"]]
            )
            .rename({
                "qid": "query",
                "docno": "docid"
            }, axis=1)
            .astype({
                "docid": str,
                "query": str
            })
        )
        return r, pd.Series(c).max()


class PairwiseFileCache(PairwiseCache):
    def __init__(self, path, pointwise: pd.DataFrame = None, depth: int = None, n_bins=None, round=None):
        super().__init__()
        self.cache = pd.read_parquet(path).drop_duplicates().astype({
            "qid": str,
            "id_a": str,
            "id_b": str,
            "score": float
        })
        if round is not None:
            self.cache["score"] = self.cache["score"].round(round)
        if n_bins is not None:
            bins = np.linspace(0, 1, n_bins) + 0.00000000001
            d = dict(enumerate(bins, 1))
            self.cache["score"] = np.log(np.vectorize(d.get)(np.digitize(np.exp(self.cache["score"]), bins)))

        if pointwise is not None and depth is not None:
            self.interpolated = pd.concat([
                pd.DataFrame([(x[0][0], x[0][1], x[1][0], x[1][1]) for x in itertools.product(
                    pointwise
                    .loc[pointwise.qid == qid]
                    .sort_values("score", ascending=False)
                    .head(depth)
                    .reset_index()
                    .loc[:, ["docno", "rank"]]
                    .values,
                    repeat=2
                )], columns=["id_a", "rank_a", "id_b", "rank_b"])
                .assign(score = lambda df: (0.5 - ((df["rank_a"] - df["rank_b"]) / (2 * depth))).round(0))
                .assign(qid=qid)
                .drop(["rank_a", "rank_b"], axis=1)
                .astype({
                    "qid": str,
                    "id_a": str,
                    "id_b": str,
                    "score": float
                }) for qid in pointwise.get_topics()
            ])

            self.cache = (
                pd.concat([
                    pd.DataFrame(itertools.product(
                        pointwise
                        .loc[pointwise.qid == qid]
                        .sort_values("score", ascending=False)
                        .head(depth)
                        .loc[:, "docno"]
                        .values,
                        repeat=2
                    ), columns=["id_a", "id_b"]).assign(qid=qid).astype({
                        "qid": str,
                        "id_a": str,
                        "id_b": str,
                    }) for qid in pointwise.get_topics()
                ])
                .merge(
                    self.cache,
                    on=["qid", "id_a", "id_b"],
                    how="left",
                )
            )
