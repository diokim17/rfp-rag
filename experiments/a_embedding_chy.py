"""Prepare pinned local models, then compare dense retrieval without generation/API calls.

Run from the owner's checkout with .venv/bin/python. Each run allocates its ID
using the existing counter, snapshots its inputs, and creates fresh output paths.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.server.server_embedding import build_index, load_index
from tracking.tracking_ids import next_experiment_id, owner_initials
from tracking.tracking_reports import file_hash, text_file_hash
from observability.observability_trace import Trace, code_version
from parsing.parsing_documents import read_json, write_json
from pipelines.server.server_retrieval import retrieve
from pipelines.server.server_client import ScenarioAClient
from experiments.a_retrieval_metrics import score_case

from config.config_paths import model_directory

MODEL_ROOT = model_directory("embedding")
MODELS = {"bge-m3": "BAAI/bge-m3", "kure-v1": "nlpai-lab/KURE-v1"}
OPTIONS = dict(rerank="none", hybrid=False, max_per_doc="none", rewrite="off",
               within_doc="off", expand=0)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    from huggingface_hub import HfApi, snapshot_download
    repo = MODELS[args.model]
    revision = HfApi().model_info(repo, revision=args.revision).sha
    path = MODEL_ROOT / f"chy-{args.model}-{revision[:12]}"
    # Never modify a previously prepared or partially downloaded model folder.
    path.mkdir(exist_ok=False)
    snapshot_download(repo, revision=revision, local_dir=path,
                      allow_patterns=["*.json", "*.safetensors", "pytorch_model*.bin", "*.model", "*.txt"],
                      ignore_patterns=["onnx/*", "openvino/*"], max_workers=2)
    if not (path / "modules.json").is_file():
        raise ValueError("Downloaded model lacks SentenceTransformer modules.json")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise ValueError("Downloaded model lacks weights")
    manifest = {"repo_id": repo, "revision": revision,
                "files": {str(p.relative_to(path)): sha256(p)
                          for p in sorted(path.rglob("*"))
                          if p.is_file() and ".cache" not in p.relative_to(path).parts}}
    write_json(path / "local_manifest.json", manifest)
    print(json.dumps({"model_folder": path.name, "repo_id": repo, "revision": revision}))


def validate_inputs(documents, cases):
    if not documents or not cases:
        raise ValueError("Documents and evaluation cases must not be empty")
    ids = {doc["doc_id"] for doc in documents}
    if len(ids) != len(documents):
        raise ValueError("Duplicate document IDs")
    case_ids = [case["id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate evaluation IDs")
    for case in cases:
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("Evaluation question is missing")
        gold = case.get("expected_doc_ids")
        if not isinstance(gold, list) or not gold or not set(gold) <= ids:
            raise ValueError("Evaluation contains missing gold documents")


def verify_model(path):
    manifest = read_json(path / "local_manifest.json")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()) or sha256(target) != expected:
            raise ValueError("Model manifest verification failed")
    return manifest


def measured_cases(cases, client, index, chunks, config, repeat):
    records = []
    for repetition in range(repeat):
        for case in cases:
            start = time.perf_counter()
            error = None
            try:
                hits = retrieve(case["question"], client, index, chunks, config,
                                5, case.get("filters"), **OPTIONS)
            except Exception as exc:
                hits, error = [], type(exc).__name__
            metrics = score_case(case["expected_doc_ids"], [h["doc_id"] for h in hits], 5)
            records.append({"id": case["id"], "repetition": repetition,
                            "seconds": time.perf_counter() - start, **metrics,
                            "error": error, "hits": [{k: h[k] for k in
                            ("doc_id", "chunk_id", "score")} for h in hits]})
        print(f"evaluation repetition {repetition + 1}/{repeat} complete", flush=True)
    return records


def summarize(records):
    import numpy as np
    times = [row["seconds"] for row in records]
    return dict(recall_at_5=statistics.mean(r["recall"] for r in records),
                mrr_at_5=statistics.mean(r["mrr"] for r in records),
                mean_seconds=statistics.mean(times), median_seconds=statistics.median(times),
                p95_seconds=float(np.percentile(times, 95)),
                failures=sum(r["error"] is not None for r in records))


def report(result, path):
    # An explicit allowlist: no question, chunk text, filename, filter values or secrets.
    lines = ["# Scenario A local embedding evaluation", "",
             "Dense retrieval only; local token counts are not billable API usage.", ""]
    for title in ("experiment", "model", "settings", "summary", "timing", "stats"):
        lines += [f"## {title}", "", "```json",
                  json.dumps(result[title], ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Misses (first measured pass)", "", "| ID | Recall@5 | MRR@5 | Error |",
              "| --- | ---: | ---: | --- |"]
    for row in result["records"]:
        if row["repetition"] == 0 and (row["recall"] < 1 or row["error"]):
            lines.append(f"| {row['id']} | {row['recall']:.4f} | {row['mrr']:.4f} | {row['error'] or ''} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    from dotenv import load_dotenv
    import torch
    load_dotenv(ROOT / ".env.a", override=False)
    owner = owner_initials(os.getenv("EXPERIMENT_OWNER", ""))
    if Path(args.model_folder).name != args.model_folder or args.model_folder in {".", ".."}:
        raise ValueError("model-folder must be a single directory name")
    if args.repeat < 1:
        raise ValueError("repeat must be positive")
    documents, cases = read_json(args.documents), read_json(args.eval_file)
    validate_inputs(documents, cases)
    manifest = verify_model(MODEL_ROOT / args.model_folder)
    experiment_id = next_experiment_id(owner, ROOT / ".experiment-state/a")
    paths = {name: ROOT / base / "a-embedding" / experiment_id for name, base in
             (("processed", "data/processed/a"), ("index", "indexes/a"), ("results", "results/a"))}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    # Snapshot bytes, then read the snapshot used for both hashing and execution.
    for source, name in ((args.documents, "documents.json"), (args.eval_file, "eval.json")):
        (paths["processed"] / name).write_bytes(source.read_bytes())
    documents = read_json(paths["processed"] / "documents.json")
    cases = read_json(paths["processed"] / "eval.json")
    validate_inputs(documents, cases)
    experiment = {"experiment_id": experiment_id, "owner": owner, **code_version(ROOT),
                  "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                  "documents_sha256": text_file_hash(paths["processed"] / "documents.json"),
                  "evaluation_sha256": text_file_hash(paths["processed"] / "eval.json"),
                  "module_sha256": text_file_hash(ROOT / "pipelines/server/server_local_embedding.py"),
                  "runner_sha256": text_file_hash(Path(__file__))}
    settings = {"scenario": "A", "A_EMBEDDING_MODEL": args.model_folder,
                "A_GENERATION_MODEL": "unused-retrieval-only"}
    os.environ.update(RFP_CHUNKING_STRATEGY="fixed", RFP_EMBEDDING_CONTEXT="none",
                      RETRIEVAL_REWRITE="off")
    trace = Trace.from_env(experiment)
    print(f"experiment_id={experiment_id}, documents={len(documents)}, cases={len(cases)}", flush=True)
    with trace.run("local-embedding-comparison"):
        start = time.perf_counter()
        client = ScenarioAClient(settings, MODEL_ROOT, MODEL_ROOT, need_generation=False)
        load_seconds = time.perf_counter() - start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        config = build_index(documents, client, paths["index"], args.model_folder, 1000, 150,
                             documents_sha256=experiment["documents_sha256"])
        build_seconds = time.perf_counter() - start
        build_stats = dict(client.embedding.stats)
        index, chunks, config = load_index(paths["index"])
        print(f"built {len(chunks)} chunks in {build_seconds:.2f}s", flush=True)
        # Warm-up excluded from measured latency, but counted explicitly in stats.
        retrieve(cases[0]["question"], client, index, chunks, config, 5,
                 cases[0].get("filters"), **OPTIONS)
        warm_stats = dict(client.embedding.stats)
        records = measured_cases(cases, client, index, chunks, config, args.repeat)
        summary = summarize(records)
        trace.scores(summary)
        experiment.update({"trace_id": trace.trace_id,
                           "index_sha256": file_hash(paths["index"] / "index.faiss"),
                           "chunks_sha256": text_file_hash(paths["index"] / "chunks.json")})
        result = {"experiment": experiment,
                  "model": {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                            "manifest_sha256": file_hash(MODEL_ROOT / args.model_folder / "local_manifest.json"),
                            "files": manifest["files"]},
                  "settings": {**config, **OPTIONS, "top_k": 5, "repeat": args.repeat,
                               "documents": len(documents), "cases": len(cases),
                               "device": str(client.embedding.model.device),
                               "cpu_threads": torch.get_num_threads(),
                               "dtype": str(client.embedding.model.dtype), "max_length": client.embedding.model.max_seq_length,
                               "packages": {p: importlib.metadata.version(p) for p in
                                            ("torch", "transformers", "sentence-transformers", "numpy", "faiss-cpu")}},
                  "summary": summary, "timing": {"load_seconds": load_seconds, "build_seconds": build_seconds,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0},
                  "stats": {"build": build_stats,
                            "warmup": {k: warm_stats[k] - build_stats[k] for k in warm_stats},
                            "evaluation": {k: v - warm_stats[k] for k, v in client.embedding.stats.items()},
                            "final_batch_size": client.embedding.batch_size},
                  "token_usage": trace.usage, "records": records}
    write_json(paths["results"] / "evaluation.json", result)
    report_path = ROOT / "results/a/reports" / f"a_embedding_{experiment_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report(result, report_path)
    print(json.dumps(summary))
    print(f"result={paths['results'] / 'evaluation.json'}\nreport={report_path}")
    if summary["failures"]:
        raise RuntimeError("Evaluation completed with failures; do not select this model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", choices=MODELS, required=True)
    prep.add_argument("--revision", default="main", help="Resolved to an immutable commit before download")
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--model-folder", required=True)
    evaluate.add_argument("--documents", type=Path, default=ROOT / "data/processed/a/documents.json")
    evaluate.add_argument("--eval-file", type=Path, default=ROOT / "data/eval_v2.json")
    evaluate.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
