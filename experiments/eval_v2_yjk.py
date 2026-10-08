"""김연주: 평가셋 eval_v2로 검색 고도화 후보를 비교 (답변 생성 없음).

  python experiments/eval_v2_yjk.py --eval-file data/eval_v2.json --index-dir indexes/eval-v2-structured-yjk
  python experiments/eval_v2_yjk.py ... --settings hybrid-ce-c50-cap2 prefix-ce-c50-cap2 --allow-api

지표·집계는 team_eval_yjk.py와 같습니다(doc_recall@5, 정답 문서/정답 청크 첫 순위). 순위는 --top-k(기본 5) 안에서만 셉니다.
비교하는 변형 두 가지(둘 다 이후 retrieval.py 기본 동작으로 반영):
- prefix: BM25 색인 텍스트 앞에 사업명·발주 기관을 붙입니다(retrieve의 bm25_prefix). 접두 없는 설정은
  bm25_prefix=False를 명시합니다. 반환 청크 본문과 채점은 원문 그대로입니다.
- fuzzy: retrieval._resolve_filters가 기본으로 같은 일을 하므로 이제는 결과가 같습니다. 예전 정의: 필터 값을 공백·기호를 지우고 '서울특별시→서울시'로 맞춘 뒤 메타데이터 값에 포함되는지로 찾아
  하나로 정해지면 그 값으로 바꿔 exact-match 필터에 넘깁니다(예: '한국철도공사' → '한국철도공사 (용역)').
"""

import argparse
import hashlib
import math
import os
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing import read_json, write_json  # noqa: E402
from retrieval_eval_yjk import CachedEmbeddingClient  # noqa: E402
from team_eval_yjk import file_hash, relative, report_lines, score_case, summarize_groups  # noqa: E402

BASE = {"vector": {"rerank": "none", "hybrid": False, "max_per_doc": "none"}}
for prefix in ("", "prefix-"):
    HYBRID = {"hybrid": True, "hybrid_vector_k": 100, "hybrid_bm25_k": 100, "bm25_prefix": prefix == "prefix-"}
    BASE[f"{prefix}hybrid"] = {"rerank": "none", "candidates": 100, "max_per_doc": "none", **HYBRID}
    BASE[f"{prefix}hybrid-cap2"] = {"rerank": "none", "candidates": 100, "max_per_doc": 2, **HYBRID}
    for candidates in (50, 100):
        BASE[f"{prefix}hybrid-ce-c{candidates}"] = {"rerank": "cross-encoder", "candidates": candidates,
                                                   "max_per_doc": "none", **HYBRID}
        BASE[f"{prefix}hybrid-ce-c{candidates}-cap2"] = {"rerank": "cross-encoder", "candidates": candidates,
                                                        "max_per_doc": 2, **HYBRID}
        # lexical: 후보 안 글자 2-gram BM25(사업명·발주 기관 접두)와 코사인을 표준점수로 더해 재정렬(CPU)
        BASE[f"{prefix}hybrid-lex-c{candidates}"] = {"rerank": "lexical", "candidates": candidates,
                                                    "max_per_doc": "none", **HYBRID}
        BASE[f"{prefix}hybrid-lex-c{candidates}-cap2"] = {"rerank": "lexical", "candidates": candidates,
                                                         "max_per_doc": 2, **HYBRID}
# 이름에 'fuzzy'가 붙으면 필터 값 정규화를 함께 씁니다.
SETTINGS = {**BASE, **{f"{name}+fuzzy": options for name, options in BASE.items()}}
# '+wd'는 문서 안 재선택(residual+full), '+wdr'은 남은 질문만(residual) — retrieve의 within_doc
SETTINGS.update({f"{name}{suffix}": {**options, "within_doc": mode} for name, options in BASE.items()
                 for suffix, mode in (("+wd", "residual+full"), ("+wdr", "residual"))})
# '+x5'·'+x10'·'+x20'은 재정렬 후보 확장(retrieve의 expand) — 재정렬하는 설정에만
SETTINGS.update({f"{name}+x{count}": {**options, "expand": count} for name, options in BASE.items()
                 if options["rerank"] != "none" for count in (5, 10, 20)})


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


def org_key(value):
    text = nfc(value).replace("서울특별시", "서울시")
    return "".join(ch for ch in text if ch.isalnum()).casefold()


def resolve_filters(filters, chunks):
    """필터 값마다 정규화 포함 관계로 메타데이터 값을 찾고, 하나로 정해질 때만 바꿉니다."""
    if not filters:
        return filters
    resolved = {}
    for key, value in filters.items():
        known = {nfc(chunk["metadata"][key]) for chunk in chunks if key in chunk["metadata"]}
        matches = sorted(v for v in known if org_key(value) in org_key(v))
        resolved[key] = matches[0] if nfc(value) not in known and len(matches) == 1 else value
    return resolved


def cached_cross_encoder(retrieval, path):
    """(모델, 질문, 문단) 해시별 cross-encoder 점수를 파일에 캐시합니다. 설정끼리 겹치는 후보를 다시 계산하지 않습니다."""
    original = retrieval._cross_encoder_scores
    cache = read_json(path) if Path(path).is_file() else {}

    def scores(question, passages, model_name):
        keys = [hashlib.sha256(f"{model_name}\n{question}\n{p}".encode()).hexdigest() for p in passages]
        todo = [i for i, key in enumerate(keys) if key not in cache]
        if todo:
            for i, value in zip(todo, original(question, [passages[i] for i in todo], model_name)):
                cache[keys[i]] = value
        return [cache[key] for key in keys]
    return original, scores, lambda: write_json(path, cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--cache-file", type=Path, default=ROOT / "results/cache/query_embeddings.json")
    parser.add_argument("--top-k", type=int, default=5, help="반환 청크 수이자 지표를 세는 순위 범위(팀 기준 5로 고정)")
    parser.add_argument("--ce-cache-file", type=Path, default=ROOT / "results/cache/cross_encoder_scores.json")
    parser.add_argument("--allow-api", action="store_true", help="캐시에 없는 질문 임베딩 API 호출을 허용")
    parser.add_argument("--fresh-query-embeddings", action="store_true",
                        help="질문 임베딩 캐시를 읽지 않고 모두 API로 새로 받습니다(캐시 파일도 갱신하지 않음)")
    parser.add_argument("--allow-cpu-rerank", action="store_true",
                        help="GPU가 없을 때 cross-encoder를 CPU로 돌리는 것을 허용(기본은 중단)")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from embedding import embed_texts, load_index
    from experiment_ids import next_experiment_id, owner_initials
    import retrieval

    cases = read_json(args.eval_file)
    index, chunks, config = load_index(args.index_dir)
    if any(SETTINGS[name]["rerank"] == "cross-encoder" for name in args.settings) and not args.allow_cpu_rerank:
        import torch
        if not torch.cuda.is_available():
            parser.error("cross-encoder 설정은 GPU에서만 돌립니다. CUDA가 없습니다(--allow-cpu-rerank로 CPU 허용).")
    client = CachedEmbeddingClient(args.cache_file)
    if args.fresh_query_embeddings:
        client.cache, client.save = {}, lambda: None
        args.allow_api = True
    model = config["embedding_model"]
    questions = sorted({case["question"] for case in cases})
    missing = [q for q in questions if hashlib.sha256(f"{model}\n{q}".encode()).hexdigest() not in client.cache]
    if missing and not args.allow_api:
        parser.error(f"캐시에 없는 질문 {len(missing)}건은 임베딩 API를 호출합니다. 확인 후 --allow-api를 붙이세요.")
    load_dotenv(ROOT / ".env", override=False)
    experiment_id = next_experiment_id(owner_initials(os.getenv("EXPERIMENT_OWNER", "")), ROOT / ".experiment-state")
    print(f"실험 ID: {experiment_id} / 설정 {len(args.settings)}개 / 질문 {len(cases)}건 / 임베딩할 질문 {len(missing)}건")
    for start in range(0, len(missing), 64):
        embed_texts(missing[start:start + 64], client, model)
    client.save()

    ce_original, ce_cached, ce_save = cached_cross_encoder(retrieval, args.ce_cache_file)
    retrieval._cross_encoder_scores = ce_cached
    ce_load_ms, rerank_device = 0.0, None
    if any(SETTINGS[name]["rerank"] == "cross-encoder" for name in args.settings):
        start = time.perf_counter()
        retrieval.retrieve(cases[0]["question"], client, index, chunks, config, 1, None,
                           rerank="cross-encoder", candidates=1)
        ce_load_ms = (time.perf_counter() - start) * 1000
        rerank_device = next(iter(retrieval._cross_encoders.values()))[2]

    # eval_v2는 확장 문항에 bm25_reference가 없어 기준값 열은 싣지 않습니다.
    results = []
    for name in args.settings:
        rows, times = [], []
        for case in cases:
            filters = resolve_filters(case.get("filters"), chunks) if name.endswith("+fuzzy") else case.get("filters")
            start = time.perf_counter()
            hits = retrieval.retrieve(case["question"], client, index, chunks, config, args.top_k, filters,
                                      **SETTINGS[name])
            times.append((time.perf_counter() - start) * 1000)
            rows.append({"id": case["id"], **score_case(case, hits), "hits": [hit["chunk_id"] for hit in hits]})
        ordered = sorted(times)
        results.append((name, {"options": {**retrieval.retrieval_options(**SETTINGS[name]),
                                           "fuzzy_filter": name.endswith("+fuzzy")},
                               "rows": rows, "mean_ms": statistics.fmean(times),
                               "p95_ms": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)]}))
        ce_save()
        print(f"{name}: 완료 ({statistics.fmean(times):.0f}ms/질문)", flush=True)
    retrieval._cross_encoder_scores = ce_original
    for _, result in results:
        result["summary"] = summarize_groups(cases, result["rows"])
    if client.api_calls:
        client.save()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"experiment_id": experiment_id, "measured_at_utc": stamp, "eval_file": relative(args.eval_file),
            "evaluation_sha256": file_hash(args.eval_file), "cases": len(cases), "index_dir": relative(args.index_dir),
            "chunks_sha256": file_hash(Path(args.index_dir) / "chunks.json"), "chunk_count": len(chunks),
            "chunking_strategy": config.get("chunking_strategy"),
            "embedding_context": config.get("embedding_context"), "embedding_model": model, "top_k": args.top_k,
            "cross_encoder_model": os.getenv("RETRIEVAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            "cross_encoder_load_ms": round(ce_load_ms), "embedding_api_calls": client.api_calls,
            "fresh_query_embeddings": args.fresh_query_embeddings, "rerank_device": rerank_device}
    write_json(Path(args.results_dir) / f"eval_v2_{experiment_id}_{stamp}.json", {"meta": meta, "results": dict(results)})
    lines = report_lines(meta, cases, results)
    lines[0] = "# eval_v2 검색 지표: 하이브리드·리랭킹·BM25 메타 접두·필터 정규화"
    lines = [line.replace("상위 50개 안에서만", f"상위 {args.top_k}개 안에서만") for line in lines]
    report_path = Path(args.reports_dir) / f"eval_v2_yjk_{experiment_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"보고서: {relative(report_path)}")


if __name__ == "__main__":
    main()
