import pandas as pd
from trectools import TrecQrel, TrecRun, TrecEval
from glob import glob
from tqdm import tqdm

COLLECTIONS = [
    "cw12",
    "dl-passages",
    "cw09"
]
MODELS = [
    "3b"
]


def evaluate_sampled():
    filesets = {
        f"{collection}-{model}": glob(f"../data/processed/runs-sampled/{collection}-{model}/*.txt.gz") for model in MODELS for collection in COLLECTIONS
    }

    pbar = tqdm(total=sum(map(len, filesets.values())))
    data = []

    for collection in COLLECTIONS:
        qrels = TrecQrel()
        qrels.qrels_data = pd.concat([TrecQrel(f).qrels_data for f in glob(f"../data/external/qrels/{collection}/*.txt")])

        for model in MODELS:
            pbar.set_description(f"{collection}-{model}")
            runs = []
            for f in filesets[f"{collection}-{model}"]:
                r = TrecRun(f)
                eval = (
                    TrecEval(r, qrels)
                    .evaluate_all(per_query=True)
                    .data
                    .pivot("query", "metric", "value")
                    .reset_index()
                )
                eval = eval.assign(runid=eval.loc[eval["query"] == "all", "runid"].values[0])
                eval = eval.loc[eval["query"] != "all"]
                runs.append(eval)
                pbar.update(1)
            data.append(pd.concat(runs).assign(model=model, collection=collection))

    data = pd.concat(data)

    runs_complete = (
        pd.concat([
            data.reset_index(drop=True),
            data["runid"]
            .apply(lambda x: pd.Series(
                x.split("-"),
                index=["i", "k", "sampler", "sample_rate", "aggregator"]
            ))
            .assign(aggregator=lambda df: df["aggregator"].replace({
                "pagerank": "PageRank",
                "bradleyterry": "Bradley-Terry",
                "additive": "Additive",
                "greedy": "Greedy"
            }))
            .assign(sampler=lambda df: df["sampler"].replace({
                "random": "Random",
                "structured": "Structured",
            }))
            .reset_index(drop=True)
        ], axis=1)
        .set_index(["collection", "model", "sampler", "sample_rate", "aggregator", "i", "runid", "query"])
        .loc[
            :,
            ["NDCG_10"]
        ]
        .reset_index()
        .astype({
            "sample_rate": float,
        })
    )
    runs_complete.to_csv("../data/processed/aggregated/evaluated-sampled.csv", index=False)


def evaluate_gridsearch():
    filesets = {
        f"{collection}-{model}": glob(f"../data/processed/runs-gridsearch/{collection}-{model}/*.txt.gz") for model in MODELS for collection in COLLECTIONS
    }

    pbar = tqdm(total=sum(map(len, filesets.values())))
    data = []

    for collection in COLLECTIONS:
        qrels = TrecQrel()
        qrels.qrels_data = pd.concat([TrecQrel(f).qrels_data for f in glob(f"../data/external/qrels/{collection}/*.txt")])

        for model in MODELS:
            pbar.set_description(f"{collection}-{model}")
            runs = []
            for f in filesets[f"{collection}-{model}"]:
                r = TrecRun(f)
                eval = (
                    TrecEval(r, qrels)
                    .evaluate_all(per_query=True)
                    .data
                    .pivot("query", "metric", "value")
                    .reset_index()
                )
                eval = eval.assign(runid=eval.loc[eval["query"] == "all", "runid"].values[0])
                eval = eval.loc[eval["query"] != "all"]
                runs.append(eval)
                pbar.update(1)
            data.append(pd.concat(runs).assign(model=model, collection=collection))

    data = pd.concat(data)

    runs_complete = (
        pd.concat([
            data.reset_index(drop=True),
            data["runid"]
            .apply(lambda x: pd.Series(
                x.split("-"),
                index=["i", "k", "sampler", "sample_rate", "aggregator"]
            ))
            .assign(aggregator=lambda df: df["aggregator"].replace({
                "pagerank": "PageRank",
                "bradleyterry": "Bradley-Terry",
                "additive": "Additive",
                "greedy": "Greedy"
            }))
            .assign(sampler=lambda df: df["sampler"].replace({
                "random": "Random",
                "structured": "Structured",
            }))
            .reset_index(drop=True)
        ], axis=1)
        .set_index(["collection", "model", "sampler", "sample_rate", "aggregator", "i", "runid", "query"])
        .loc[
            :,
            ["NDCG_10"]
        ]
        .reset_index()
        .astype({
            "sample_rate": float,
        })
    )
    runs_complete.to_csv("../data/processed/aggregated/evaluated-gridsearch.csv", index=False)


def evaluate_full():
    filesets = {
        f"{collection}-{model}": glob(f"../data/processed/runs-full/{collection}-{model}/*.txt") for model in
        MODELS for collection in COLLECTIONS
    }

    pbar = tqdm(total=sum(map(len, filesets.values())))
    data = []

    for collection in COLLECTIONS:
        qrels = TrecQrel()
        qrels.qrels_data = pd.concat(
            [TrecQrel(f).qrels_data for f in glob(f"../data/external/qrels/{collection}/*.txt")])

        for model in MODELS:
            pbar.set_description(f"{collection}-{model}")
            runs = []
            for f in filesets[f"{collection}-{model}"]:
                r = TrecRun(f)
                eval = (
                    TrecEval(r, qrels)
                    .evaluate_all(per_query=True)
                    .data
                    .pivot("query", "metric", "value")
                    .reset_index()
                )
                eval = eval.assign(runid=eval.loc[eval["query"] == "all", "runid"].values[0])
                eval = eval.loc[eval["query"] != "all"]
                runs.append(eval)
                pbar.update(1)
            data.append(pd.concat(runs).assign(model=model, collection=collection))

    runs_complete = (
        pd.concat(data)
        .assign(aggregator=lambda df: (
            df["runid"]
            .apply(lambda x: x.split("-")[-1])
            .replace({
                "pagerank": "PageRank",
                "bradleyterry": "Bradley-Terry",
                "additive": "Additive",
                "greedy": "Greedy",
                "kwiksort": "Kwiksort"
            }))
        )
        .reset_index(drop=True)
        .loc[
            :,
            ["collection", "model", "query", "aggregator", "NDCG_10"]
        ]
    )
    runs_complete.to_csv("../data/processed/aggregated/evaluated-full.csv", index=False)


def evaluate_baselines():
    filesets = {
        f"{collection}-{model}": glob(f"../data/interim/pointwise/{collection}/mono_t5_{model}-run.txt") for model in
        MODELS for collection in COLLECTIONS
    }

    pbar = tqdm(total=sum(map(len, filesets.values())))
    data = []
    for collection in COLLECTIONS:
        qrels = TrecQrel()
        qrels.qrels_data = pd.concat(
            [TrecQrel(f).qrels_data for f in glob(f"../data/external/qrels/{collection}/*.txt")])

        for model in MODELS:
            pbar.set_description(f"{collection}-{model}")
            runs = []
            for f in filesets[f"{collection}-{model}"]:
                r = TrecRun(f)
                eval = (
                    TrecEval(r, qrels)
                    .evaluate_all(per_query=True)
                    .data
                    .pivot("query", "metric", "value")
                    .reset_index()
                )
                eval = eval.assign(runid=eval.loc[eval["query"] == "all", "runid"].values[0])
                eval = eval.loc[eval["query"] != "all"]
                runs.append(eval)
                pbar.update(1)
            data.append(pd.concat(runs).assign(model=model, collection=collection))

    runs_complete = (
        pd.concat(data)
        .reset_index(drop=True)
        .loc[
            :,
            ["collection", "model", "query", "NDCG_10"]
        ]
    )
    runs_complete.to_csv("../data/processed/aggregated/evaluated-baselines.csv", index=False)


if __name__=="__main__":
    evaluate_baselines()
    evaluate_full()
    evaluate_sampled()
    evaluate_gridsearch()
