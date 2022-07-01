import unittest

from retrieval_pipelines.FileSystemCache import FileSystemCache
import json


class FileSystemCacheTest(unittest.TestCase):

    def test_insertion_of_persisted_example(self):
        tmp_dir = self.temp_dir()
        cache = FileSystemCache(tmp_dir)
        self.write_to_file(tmp_dir + '/cache_v_0.json', 'no-valid-json')
        self.write_to_file(tmp_dir + '/cache_v_1.json', 'no-valid-json')
        self.write_to_file(tmp_dir + '/cache_v_2.json', json.dumps({'[1, "d1", "d2"]': 0, '[1, "d1", "d3"]': 1}))

        self.assertEquals(0, cache[(1, 'd1', 'd2')])
        self.assertEquals(1, cache[(1, 'd1', 'd3')])

    def test_iteration_of_persisted_example(self):
        tmp_dir = self.temp_dir()
        cache = FileSystemCache(tmp_dir)
        self.write_to_file(tmp_dir + '/cache_v_0.json', 'no-valid-json')
        self.write_to_file(tmp_dir + '/cache_v_1.json', 'no-valid-json')
        self.write_to_file(tmp_dir + '/cache_v_2.json', json.dumps({'[1, "d1", "d2"]': 0, '[1, "d1", "d3"]': 1}))

        expected = [
            "(1, 'd1', 'd2') -> 0",
            "(1, 'd1', 'd3') -> 1"
        ]
        actual = []

        for (qid, did1, did2), score in cache:
            actual += ['(' + str(qid) + ', \'' + did1 + '\', \'' + did2 + '\') -> ' + str(score)]

        self.assertEqual(expected, actual)

    def test_single_step_usage_of_cache(self):
        tmp_dir = self.temp_dir()
        cache = FileSystemCache(tmp_dir)

        self.assertEquals(0, len([i for i in cache]))

        cache.add(123, "did11", "did22", 1)
        cache.add(123, "did33", "did44", 3)
        cache.add(124, "did22", "did11", 2)

        cache.ensure_new_pairs_are_persisted()
        cache = FileSystemCache(tmp_dir)

        self.assertEquals(1, cache[(123, "did11", "did22")])
        self.assertTrue((123, "did11", "did22") in cache)
        self.assertEquals(3, cache[(123, "did33", "did44")])
        self.assertTrue((123, "did33", "did44") in cache)

        self.assertEquals(2, cache[(124, "did22", "did11")])
        self.assertTrue((124, "did22", "did11") in cache)

    def test_two_step_usage_of_cache(self):
        tmp_dir = self.temp_dir()
        cache = FileSystemCache(tmp_dir)

        # Dummy entry for empty cache
        self.assertEquals(0, len([i for i in cache]))

        cache.add(123, "did11", "did22", 1)
        cache.add(123, "did33", "did44", 3)
        cache.add(124, "did22", "did11", 2)

        cache.ensure_new_pairs_are_persisted()
        cache = FileSystemCache(tmp_dir)

        self.assertEquals(1, cache[(123, "did11", "did22")])
        self.assertEquals(3, cache[(123, "did33", "did44")])
        self.assertEquals(2, cache[(124, "did22", "did11")])

        cache.add(1111, 'd_1', 'd_2', 22)

        cache.ensure_new_pairs_are_persisted()
        cache = FileSystemCache(tmp_dir)

        self.assertEquals(1, cache[(123, "did11", "did22")])
        self.assertEquals(3, cache[(123, "did33", "did44")])
        self.assertEquals(2, cache[(124, "did22", "did11")])
        self.assertEquals(22, cache[(1111, "d_1", "d_2")])

    @staticmethod
    def write_to_file(file_name, content):
        with open(file_name, 'w') as f:
            f.write(content)

    @staticmethod
    def temp_dir():
        import tempfile
        return str(tempfile.mkdtemp())
