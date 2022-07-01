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
def worker(path, pointwise, pairwise, aggregator, sample_rate, k, i):
    # Random sampling for different values of lambda
    tag = f"{i}-{k}-random-{round(sample_rate, 2)}-{str(aggregator)}"
    run, _ = pairwise.run(
        tag,
        pointwise,
        PairwiseRandomSampler(frac=sample_rate, l=0),
        aggregator,
        depth=k,
    )
    run.run_data.to_csv(path + tag + ".txt.gz", sep=" ", index=False, header=False, compression="gzip")
    # Structured sampling for a window size calculated based on a given sample rate
    l = int((sample_rate * (k * (k - 1))) / k)
    tag = f"{i}-{k}-structured-{round(sample_rate, 2)}-{str(aggregator)}"
    run, _ = pairwise.run(
        tag,
        pointwise,
        PairwiseWindowedSampler(window_size=l, cyclical=True),
        aggregator,
        depth=k,
    )
    run.run_data.to_csv(path + tag + ".txt.gz", sep=" ", index=False, header=False, compression="gzip")


if __name__ == "__main__":
    from multiprocessing import Pool

    # Parameters
    K = 50
    REPETITIONS = 10
    SAMPLING_STEPS = 20
    PATH = "../data/processed/runs-sampled/"
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

    total = REPETITIONS * len(AGGREGATORS) * len(CACHES)
    pbar = tqdm(total=total)

    for cache, version in CACHES:
        pbar.set_description(cache)
        pointwise = PointwiseRunFile(f"../data/interim/pointwise/{cache}/mono_t5_{version}-run.txt")
        pairwise = PairwiseFileCache(
            f"../data/interim/pairwise/{cache}-{version}.parquet",
            depth=K,
            pointwise=pointwise
        )

        for i in range(REPETITIONS):
            for aggregator in AGGREGATORS:
                with Pool(processes=8) as pool:
                    results = []
                    parameters = []
                    for sample_rate in np.linspace(1 / SAMPLING_STEPS, 1 - (1 / SAMPLING_STEPS), SAMPLING_STEPS - 1):
                        parameters.append((f"{PATH}{cache}-{version}/", pointwise, pairwise, aggregator, round(sample_rate, 2), K, i))
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
