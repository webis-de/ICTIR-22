class FileSystemCache:
    def __init__(self, directory):
        self.__directory = directory
        self.__cache__ = None

    def __most_recent_file(self):
        from glob import glob
        ret = glob(self.__directory + '/cache_v_*.json')
        if len(ret) == 0:
            return None
        else:
            ret = sorted(ret, key=lambda i: int(i.split('cache_v_')[1].split('.json')[0]), reverse=True)
            return ret[0]

    def __getitem__(self, item):
        return self.__all_items()[item]

    def __iter__(self):
        return self.__all_items().items().__iter__()

    def __contains__(self, item):
        return item in self.__all_items()

    def add(self, qid, doc1, doc2, score):
        self.__all_items()
        self.__cache__[(qid, doc1, doc2)] = score

    def ensure_new_pairs_are_persisted(self):
        import json
        self.__all_items()
        f = self.__most_recent_file()
        if not f:
            f = '/cache_v_0.json'

        f = int(f.split('/cache_v_')[1].split('.json')[0])

        with open(self.__directory + '/cache_v_' + str(f + 1) + '.json', 'w') as f:
            f.write(json.dumps({json.dumps(k): v for k, v in self}))

    def __all_items(self):
        if self.__cache__:
            return self.__cache__
        else:
            self.__cache__ = self.__load_cache()
            return self.__cache__

    def __load_cache(self):
        f = self.__most_recent_file()
        import json
        if f:
            ret = json.load(open(f))
            return {tuple(json.loads(k)): v for k, v in ret.items()}
        else:
            return {}
