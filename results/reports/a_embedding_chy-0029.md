# Scenario A local embedding evaluation

Dense retrieval only; local token counts are not billable API usage.

## experiment

```json
{
  "experiment_id": "chy-0029",
  "owner": "chy",
  "commit": "2c3136996f477eb66d8f7fed800cec1094fb667c",
  "dirty": true,
  "measured_at_utc": "2026-10-12T04:17:23.310816+00:00",
  "documents_sha256": "97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a",
  "evaluation_sha256": "3f1f37b5eb5a06ed0827d4368dad73c99a6bd48ab4048334804bd6cd6d2f2b00",
  "module_sha256": "67744ea6c44236a31af1bb43160ed8df6cf8705ed92e3e97f438dbbcdd91b296",
  "runner_sha256": "01c19563fa841033ecdfb8943baf01fb750702ad6c5c55137c7a11fd158b59f6",
  "trace_id": "e8f8e06b6a6f6736cca713b5d2388fff",
  "index_sha256": "9786f9e6570ebe27ce47f491a775ddb379d70841695599d8c1ec04df7a46e32a",
  "chunks_sha256": "874aee869ed0fc4b2242f37ae7ede49517c1307ec421ef503794ee36e0496a0c"
}
```

## model

```json
{
  "repo_id": "BAAI/bge-m3",
  "revision": "5617a9f61b028005a4858fdac845db406aefb181",
  "manifest_sha256": "c08690d7741b680b8a382f643588ba1909e6ae6e7cffee7c030afe15f4879611",
  "files": {
    "1_Pooling/config.json": "e54c164a07274f2eb45bb724f54a79d1efcc90c41573887cd9a29aeee0597352",
    "config.json": "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
    "config_sentence_transformers.json": "1eef72430e7194a1e59680e635aed81ffa083f05668dbc5bb1c56c04c0999c38",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "eb9b44b13c0f52a3b3685c3b1cbdea1ba8b04bea123b98f61610048940776eb1",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
    "tokenizer_config.json": "a62b2b6784f990259fddef5f16388693a8043be4f69179e6a5257eeb3f9abac4",
    "pytorch_model.bin": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
  }
}
```

## settings

```json
{
  "embedding_model": "chy-bge-m3-5617a9f61b02",
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
  "recall_at_5": 0.9351282051282052,
  "mrr_at_5": 0.9512820512820512,
  "mean_seconds": 0.03504166721055905,
  "median_seconds": 0.033533799927681684,
  "p95_seconds": 0.04534572795964777,
  "failures": 0
}
```

## timing

```json
{
  "load_seconds": 10.244372791144997,
  "build_seconds": 136.18499291688204,
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
| h05 | 0.0000 | 0.0000 |  |
| h14 | 0.6667 | 1.0000 |  |
| g02 | 0.7500 | 1.0000 |  |
| g06 | 0.5000 | 0.3333 |  |
| n06 | 0.6667 | 1.0000 |  |
| g07 | 0.2000 | 1.0000 |  |
| g09 | 0.5000 | 1.0000 |  |
| g10 | 0.5000 | 0.2500 |  |
