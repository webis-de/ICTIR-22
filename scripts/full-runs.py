import sys
sys.path.insert(0, "../")

from tqdm import tqdm
from src.caching import PairwiseFileCache, PointwiseRunFile
from src.sampling import PairwiseFullSampler
from src.aggregation import (
    AdditiveAggregator,
    BradleyTerryAggregator,
    GreedyAggregator,
    PageRankAggregator,
    KwikSortAggregator
)

K = 50
AGGREGATORS = [
    AdditiveAggregator(logit_scores=True),
    BradleyTerryAggregator(logit_scores=True, cython=True),
    GreedyAggregator(logit_scores=True),
    PageRankAggregator(logit_scores=True),
    KwikSortAggregator(logit_scores=True)
]
COLLECTIONS = [
    "cw09",
    "cw12",
    "dl-passages"
]
MODELS = [
    "3b"
]


if __name__ == "__main__":
    pbar = tqdm(total = len(COLLECTIONS)*len(MODELS)*len(AGGREGATORS))
    for collection in COLLECTIONS:
        for model in MODELS:
            pbar.set_description(f"{collection}-{model}")
            pointwise = PointwiseRunFile(f"../data/interim/pointwise/{collection}/mono_t5_{model}-run.txt")
            pairwise = PairwiseFileCache(
                f"../data/interim/pairwise/{collection}-{model}.parquet",
                depth=K,
                pointwise=pointwise
            )
            for agg in AGGREGATORS:
                try:
                    run, _ = pairwise.run(
                        tag=f"{collection}-{model}-{str(agg)}",
                        pointwise_scores=pointwise,
                        sampler=PairwiseFullSampler(),
                        aggregator=agg,
                        depth=K,
                    )
                    filename = f"../data/processed/runs-full-logits/{collection}-{model}/{str(agg)}.txt"
                    run.run_data.to_csv(filename, sep=" ", index=False, header=False)
                    pbar.update(1)
                except Exception as e:
                    print(e)
                    pbar.update(1)