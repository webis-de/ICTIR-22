#!/usr/bin/env python3
from trectools import TrecRun, TrecQrel
from glob import glob
from tqdm import tqdm
import sys
import gzip

def trec_run(file_name):
    if file_name.endswith('s.gz'):
        return TrecRun(gzip.open(file_name)).run_data.docid.unique()
    else:
        return 

def all_runs(year):
    ret = set()
    for f in tqdm(glob('runs/trec-' + str(year) +'-web.adhoc/*')):
        ret.update(TrecRun(f).run_data.docid.unique())

    return ret

def all_qrels(year):
    TOPICS = {
    	18: '1-50',
    	19: '51-100',
    	20: '101-150',
    	21: '151-200',
    	22: '201-250',
    	23: '251-300',
    }
    qrels = TrecQrel('topics-and-qrels/qrels.web.' + TOPICS[year] + '.txt').qrels_data

    return set(qrels.docid.unique())

if __name__ == '__main__':
    ids = []
    for year in [18, 19, 20, 21, 22, 23]:
        ids += list(all_qrels(year)) + list(all_runs(year))

    ids = list(ids)
    ids = sorted(ids) 
    with open('all-important-docs-to-include.txt', 'w') as f:
            for docid in ids:
                f.write(docid + '\n')

    ids = []
    for year in [18, 19, 20, 21, 22, 23]:
        ids += list(all_qrels(year))

    ids = list(ids)
    ids = sorted(ids) 
    with open('docs-from-qrels-to-include.txt', 'w') as f:
            for docid in ids:
                f.write(docid + '\n')

