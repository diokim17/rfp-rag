# Scenario A local embedding evaluation

Dense retrieval only; local token counts are not billable API usage.

## experiment

```json
{
  "experiment_id": "chy-0030",
  "owner": "chy",
  "commit": "2c3136996f477eb66d8f7fed800cec1094fb667c",
  "dirty": true,
  "measured_at_utc": "2026-10-12T04:20:15.478522+00:00",
  "documents_sha256": "97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a",
  "evaluation_sha256": "3f1f37b5eb5a06ed0827d4368dad73c99a6bd48ab4048334804bd6cd6d2f2b00",
  "module_sha256": "67744ea6c44236a31af1bb43160ed8df6cf8705ed92e3e97f438dbbcdd91b296",
  "runner_sha256": "01c19563fa841033ecdfb8943baf01fb750702ad6c5c55137c7a11fd158b59f6",
  "trace_id": "fa22b25bb6497c8db84d7bc73ca81045",
  "index_sha256": "bc2e509f8cb38e0ec1a541a28040e2a3c38428b6d85414f7ece79556bdae789f",
  "chunks_sha256": "874aee869ed0fc4b2242f37ae7ede49517c1307ec421ef503794ee36e0496a0c"
}
```

## model

```json
{
  "repo_id": "nlpai-lab/KURE-v1",
  "revision": "8b418a58414668e75532ed045c22d9ca018ae2b2",
  "manifest_sha256": "eaca68994c2563f5b0c117900d6667fb2f78d7d0291351fdc66184aef6592f27",
  "files": {
    "1_Pooling/config.json": "13e69897522ee8255104483ed9f219465d1be3936654a54a318758738052789e",
    "config.json": "852d42e020c7f989c2acaf30fc683b7f768e8c6d1ab17166e835442162bd825d",
    "config_sentence_transformers.json": "8c73684c2160f2209f3706d6cbe0e933287fe6945d9546042be4c105cdb5252e",
    "model.safetensors": "c18156e80caf8ff45eb84a24a853130c3bca03087ccb41b051f86e7556bae02c",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "eb9b44b13c0f52a3b3685c3b1cbdea1ba8b04bea123b98f61610048940776eb1",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "fb3c3b93c46fd5a8634e262e1b7de7da11a18b527aa2282b312952b692781dfd",
    "tokenizer_config.json": "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
    "trainer_state.json": "034d53976146a1922e238cf779dbbe33e861a771d50b73f1f18f1ae82c01b06a"
  }
}
```

## settings

```json
{
  "embedding_model": "chy-kure-v1-8b418a584146",
  "dimension": 1024,
  "chunk_count": 10218,
  "chunk_size": 1000,
  "chunk_overlap": 150,
  "chunking_strategy": "fixed",
  "chunking_version": 1,
  "embedding_context": "none",
  "embedding_context_version": 1,
  "chunk_metadata_version": 1,
  "documents_sha256": "97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a",
  "rerank": "none",
  "hybrid": false,
  "max_per_doc": "none",
  "rewrite": "off",
  "within_doc": "off",
  "expand": 0,
  "top_k": 5,
  "repeat": 3,
  "documents": 100,
  "cases": 65,
  "device": "cuda:0",
  "cpu_threads": 2,
  "dtype": "torch.float16",
  "max_length": 8192,
  "packages": {
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "sentence-transformers": "5.1.1",
    "numpy": "2.4.3",
    "faiss-cpu": "1.13.2"
  }
}
```

## summary

```json
{
  "recall_at_5": 0.9261538461538461,
  "mrr_at_5": 0.9628205128205128,
  "mean_seconds": 0.03536258237794615,
  "median_seconds": 0.03404464898630977,
  "p95_seconds": 0.045993118267506364,
  "failures": 0
}
```

## timing

```json
{
  "load_seconds": 18.054343858966604,
  "build_seconds": 137.85358210792765,
  "peak_gpu_allocated_bytes": 1325149696
}
```

## stats

```json
{
  "build": {
    "calls": 320,
    "texts": 10218,
    "input_tokens": 5718975,
    "truncated_texts": 0,
    "oom_retries": 0,
    "failures": 0
  },
  "warmup": {
    "calls": 1,
    "texts": 1,
    "input_tokens": 25,
    "truncated_texts": 0,
    "oom_retries": 0,
    "failures": 0
  },
  "evaluation": {
    "calls": 195,
    "texts": 195,
    "input_tokens": 4869,
    "truncated_texts": 0,
    "oom_retries": 0,
    "failures": 0
  },
  "final_batch_size": 8
}
```

## Misses (first measured pass)

| ID | Recall@5 | MRR@5 | Error |
| --- | ---: | ---: | --- |
| h10 | 0.6667 | 1.0000 |  |
| h14 | 0.3333 | 1.0000 |  |
| g02 | 0.5000 | 1.0000 |  |
| g03 | 0.6667 | 1.0000 |  |
| g04 | 0.6667 | 1.0000 |  |
| g06 | 0.5000 | 0.3333 |  |
| n06 | 0.6667 | 1.0000 |  |
| g07 | 0.2000 | 1.0000 |  |
| g09 | 0.0000 | 0.0000 |  |
