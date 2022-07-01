import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

from src.sampling import (
    PairwiseRandomSampler,
    PairwiseFullSampler,
    PairwiseGroupedSampler
)
from src.aggregation import (
    AdditiveAggregator,
    BradleyTerryAggregator,
    KwikSortAggregator
)
from src.retrieval_pipelines import (
    DuoT5ReRankerWithCache,
    FileSystemCache
)


class Experiment:
    def __init__(self, cache_dir, identifier=None, preranking=None, pointwise=None, sampling=None, sampling_kwargs=None,
                 pairwise=None, aggregation=None, aggregation_kwargs=None):
        self.identifier = identifier
        self.preranking = self._preranking_from_name(preranking)
        self.pointwise = self._pointwise_from_name(pointwise)
        self.pairwise = self._pairwise_from_name(
            model=pairwise,
            cache_dir=cache_dir,
            sampler=self._sampling_from_name(sampling, sampling_kwargs),
            aggregator=self._aggregation_from_name(aggregation, aggregation_kwargs)
        )

        self.dataset_ = None
        self.pipeline_ = None
        self.shared_task_ = None

    @staticmethod
    def _preranking_from_name(preranking):
        if preranking is None:
            raise ValueError("preranking is not specified, but needed.")
        try:
            return {
                "BM25": "BM25",
                "TF_IDF": "TF_IDF"
            }[preranking]
        except KeyError:
            raise ValueError("invalid preranking specified: {}".format(preranking))

    @staticmethod
    def _pointwise_from_name(pointwise):
        if pointwise is None:
            raise ValueError("pointwise is not specified, but needed.")
        if pointwise == "":
            return None
        return MonoT5ReRanker(
            tok_model='t5-' + pointwise,
            model='castorini/duot5-' + pointwise + '-msmarco',
            batch_size=32
        )

    @staticmethod
    def _pairwise_from_name(model, cache_dir, sampler, aggregator):
        return DuoT5ReRankerWithCache(
            tok_model='t5-' + model,
            model='castorini/duot5-' + model + '-msmarco',
            batch_size=32,
            cache=FileSystemCache(cache_dir + 'castorini-duot5-' + model + '-msmarco'),
            sampler=sampler,
            aggregator=aggregator,
        )

    @staticmethod
    def _sampling_from_name(sampling, sampling_kwargs):
        if sampling is None:
            raise ValueError("sampling is not specified, but needed.")
        if sampling_kwargs is None:
            sampling_kwargs = {}
        try:
            return {
                "": PairwiseFullSampler(**sampling_kwargs),
                "full": PairwiseFullSampler(**sampling_kwargs),
                "random": PairwiseRandomSampler(**sampling_kwargs),
                "grouped": PairwiseGroupedSampler(**sampling_kwargs)
            }[sampling]
        except KeyError:
            raise ValueError("invalid sampling specified: {}".format(sampling))

    @staticmethod
    def _aggregation_from_name(aggregation, aggregation_kwargs):
        if aggregation is None:
            raise ValueError("aggregation is not specified, but needed.")
        if aggregation_kwargs is None:
            aggregation_kwargs = {}
        try:
            return {
                "": AdditiveAggregator(),
                "Sym-Sum": AdditiveAggregator(),
                "Bradley-Terry": BradleyTerryAggregator(**aggregation_kwargs),
                "KwikSort": KwikSortAggregator(**aggregation_kwargs)
            }[aggregation]
        except KeyError:
            raise ValueError("invalid aggregation specified: {}".format(aggregation))

    def build(self, dataset_name: str, shared_task: str, pointwise_depth: int = 1000, pairwise_depth: int = 50):
        if not pt.started():
            pt.init()

        self.shared_task_ = shared_task
        self.dataset_ = pt.get_dataset(dataset_name)
        self.pipeline_ = pt.BatchRetrieve(self.dataset.get_index('terrier_stemmed'), wmodel=self.preranking) % pointwise_depth >> \
                         pt.text.get_text(self.dataset, "text") >> \
                         self.pointwise() % pairwise_depth >> \
                         pt.text.get_text(self.dataset, "text") >> \
                         self.pairwise

        return self

    def persist(self, out_dir):
        if self.dataset_ is None or self.pipeline_ is None or self.shared_task_ is None:
            raise
        qids = list(self.dataset_.get_qrels(self.shared_task_).qid.unique())
        topics = self.dataset_.get_topics(self.shared_task_)
        topics = topics[topics['qid'].isin(qids)]
        ret = self.pipeline_(topics)
        pt.io.write_results(ret, out_dir + self.identifier + '-run.txt')
