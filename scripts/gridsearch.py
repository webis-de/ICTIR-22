import sys

sys.path.insert(0, "../")

from src.sampling import (
    PairwiseRandomSampler,
    PairwiseWindowedSampler,
)
from src.aggregation import (
    AdditiveAggregator,
    BradleyTerryAggregator,
    PageRankAggregator,
    GreedyAggregator
)
from src.caching import PointwiseRunFile
from src.caching import PairwiseFileCache

import time
import numpy as np
from tqdm import tqdm

# Concurrent worker method
def worker(path, pointwise, pairwise, aggregator, window_size, window_skip, k):
    # Structured sampling for a window size calculated based on a given sample rate
    tag = f"{k}-structured-{str(aggregator)}-{str(window_size)}-{str(window_skip)}"
    run, _ = pairwise.run(
        tag,
        pointwise,
        PairwiseWindowedSampler(window_size=window_size, window_dilation=window_skip, cyclical=True),
        aggregator,
        depth=k,
    )
    run.run_data.to_csv(path + tag + ".txt.gz", sep=" ", index=False, header=False, compression="gzip")


if __name__ == "__main__":
    from multiprocessing import Pool

    # Parameters
    K = 50
    PATH = "../data/processed/runs-gridsearch/"
    AGGREGATORS = [
        AdditiveAggregator(log_scores=True),
        BradleyTerryAggregator(cython=True, log_scores=True),
        PageRankAggregator(log_scores=True),
        GreedyAggregator(log_scores=True)
    ]
    CACHES = [
        ("cw09", "3b"),
        ("cw12", "3b"),
        ("dl-passages", "3b"),
    ]

    total = len(AGGREGATORS) * len(CACHES)
    pbar = tqdm(total=total)

    for cache, version in CACHES:
        pbar.set_description(cache)
        pointwise = PointwiseRunFile(f"../data/interim/pointwise/{cache}/mono_t5_{version}-run.txt")
        pairwise = PairwiseFileCache(
            f"../data/interim/pairwise/{cache}-{version}.parquet",
            depth=K,
            pointwise=pointwise
        )

        for aggregator in AGGREGATORS:
            with Pool(processes=8) as pool:
                results = []
                parameters = []
                for window_size in range(1,50,2):
                    for window_skip in range(1,15, 1):
                        parameters.append((f"{PATH}{cache}-{version}/", pointwise, pairwise, aggregator, window_size, window_skip, K))
                result = pool.starmap_async(worker, parameters)
                results.append(result)
                while True:
                    time.sleep(1)
                    # catch exception if results are not ready yet
                    try:
                        ready = [result.ready() for result in results]
                        successful = [result.successful() for result in results]
                    except Exception:
                        continue
                    # exit loop if all tasks returned success
                    if all(successful):
                        break
                    # raise exception reporting exceptions received from workers
                    if all(ready) and not all(successful):
                        print(
                            f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')
                        break
            pbar.update(1)
