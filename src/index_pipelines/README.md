# Indexing ClueWeb09/ClueWeb12

```
java -cp /mnt/ceph/storage/data-tmp/2021/kibi9872/romcir22-keyqueries/retrieval-with-anserini/target/romcir22-1.0-SNAPSHOT-jar-with-dependencies.jar io.anserini.index.IndexCollection \
    -collection ClueWeb09Collection \
    -input /mnt/ceph/storage/corpora/corpora-thirdparty/corpus-clueweb09/ \
    -index indexes/lucene-index.cw09 \
    -generator DefaultLuceneDocumentGenerator \
    -whitelist /mnt/ceph/storage/data-tmp/2021/kibi9872/sigir22-pairwise-ranking/data/trec-web-tracks/docs-from-qrels-to-include.txt \
    -threads 80 -storePositions -storeDocvectors -storeRaw -storeContents |tee indexes-c4noclean.logs
```

```
java -cp /mnt/ceph/storage/data-tmp/2021/kibi9872/romcir22-keyqueries/retrieval-with-anserini/target/romcir22-1.0-SNAPSHOT-jar-with-dependencies.jar io.anserini.index.IndexCollection \
    -collection ClueWeb12Collection \
    -input /mnt/ceph/storage/corpora/corpora-thirdparty/corpus-clueweb12/parts/ \
    -index indexes/lucene-index.cw12 \
    -generator DefaultLuceneDocumentGenerator \
    -whitelist /mnt/ceph/storage/data-tmp/2021/kibi9872/sigir22-pairwise-ranking/data/trec-web-tracks/docs-from-qrels-to-include.txt \
    -threads 80 -storePositions -storeDocvectors -storeRaw -storeContents |tee indexes-c4noclean.logs
```

