import math
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
import torch
from torch.nn import functional as F
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from pyterrier.transformer import TransformerBase


class DuoT5ReRankerWithCache(TransformerBase):
    def __init__(self, sampler=None, aggregator=None, cache=None,
                 tok_model: str = 't5-base', model: str = 'castorini/duot5-base-msmarco',
                 batch_size: int = 4, text_field: str = 'text', verbose=True):
        self.sampler = sampler
        self.aggregator = aggregator
        assert self.aggregator is not None
        assert self.sampler is not None
        self.cache = cache
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def __str__(self):
        return f"DuoT5({self.model_name})"

    def transform(self, run):
        queries, texts = run['query'], run[self.text_field]
        pairwise_scores = []
        all_queries = set(queries.unique())
        prompts = self.tokenizer.batch_encode_plus([f'Relevant:' for _ in range(self.batch_size)], return_tensors='pt',
                                                   padding='longest')
        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]
        served_everything_from_cache = True
        batches = 0
        for batch in self._iter_duo_batches(run):
            batches += 1
            served_everything_from_cache = False
            enc_query = self.tokenizer.batch_encode_plus([f'Query: {q}' for q in batch['query']], return_tensors='pt',
                                                         padding='longest')
            enc_text0 = self.tokenizer.batch_encode_plus([f'Query: {q}' for q in batch['text0']], return_tensors='pt',
                                                         padding='longest')
            enc_text1 = self.tokenizer.batch_encode_plus([f'Query: {q}' for q in batch['text1']], return_tensors='pt',
                                                         padding='longest')
            enc = {}
            for key in enc_query:
                query = enc_query[key][:, :-1]  # chop off end of sequence token-- this will be added with the prompt
                text0 = enc_text0[key][:, :-1]  # chop off end of sequence token-- this will be added with the prompt
                text1 = enc_text1[key][:, :-1]  # chop off end of sequence token-- this will be added with the prompt
                # Do we need to truncate? If so, how many tokens per document?
                if query.shape[1] + text0.shape[1] + text1.shape[1] > max_vlen:
                    tokens_to_truncate = query.shape[1] + text0.shape[1] + text1.shape[1] - max_vlen
                    tokens_to_truncate_per_doc = math.ceil(tokens_to_truncate / 2)
                    text0 = text0[:, :-tokens_to_truncate_per_doc]
                    text1 = text1[:, :-tokens_to_truncate_per_doc]
                # Combine the components:
                enc[key] = torch.cat([query, text0, text1, prompts[key][:query.shape[0]]], dim=1)
            enc['decoder_input_ids'] = torch.full(
                (len(batch['ids']), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                result = self.model(**enc).logits
            result = result[:, 0, (self.REL, self.NREL)]
            result = F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()

            for (qid, did1, did2), score in zip(batch['ids'], result):
                pairwise_scores.append((qid, did1, did2, score))
                if self.cache:
                    self.cache.add(qid, did1, did2, score)

            if self.cache and (batches % 500) == 0:
                self.cache.ensure_new_pairs_are_persisted()

        if self.cache and not served_everything_from_cache:
            self.cache.ensure_new_pairs_are_persisted()
            raise ValueError('This method does not really work if we serve from cache and calculate scores. Just rerun')

        for qid, _, _, _, did0, did1 in self._iter_duo_pairs(run):
            pairwise_scores.append((qid, did0, did1, self.cache[(qid, did0, did1)]))

        scores = (
            pd.DataFrame(pairwise_scores, columns=["qid", "id_a", "id_b", "score"])
            .groupby("qid")
            .apply(lambda group: self.aggregator(group.loc[:, ["id_a", "id_b", "score"]]))
            .reset_index()
            .drop("level_1", axis=1)
        )

        run = pd.merge(
            run.drop(columns=['score', 'rank'], errors='ignore'),
            scores,
            how="left",
            on=["qid", "docno"]
        )
        run = add_ranks(run)
        return run

    def _iter_duo_pairs(self, run):
        groups = run.groupby('qid')
        if self.verbose:
            groups = pt.tqdm(groups, desc='duoT5', unit='queries')
        for qid, group in groups:
            for _, (id_a, id_b) in self.sampler(group).iterrows():
                row1 = group[group["docno"] == id_a].iloc[0, :]
                row2 = group[group["docno"] == id_b].iloc[0, :]
                yield (
                    qid,
                    row1.query,
                    getattr(row1, self.text_field),
                    getattr(row2, self.text_field),
                    row1.docno,
                    row2.docno
                )

    def _iter_duo_batches(self, run):
        batch = {'ids': [], 'query': [], 'text0': [], 'text1': []}
        for qid, query, text0, text1, did0, did1 in self._iter_duo_pairs(run):
            if self.cache:
                if (qid, did0, did1) in self.cache:
                    continue
            batch['ids'].append((qid, did0, did1))
            batch['query'].append(query)
            batch['text0'].append(text0)
            batch['text1'].append(text1)
            if len(batch['ids']) == self.batch_size:
                yield batch
                for v in batch.values():
                    v.clear()
        if len(batch['ids']) > 0:
            yield batch
