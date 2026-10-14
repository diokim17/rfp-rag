"""김연주: 평가셋(eval_v2·eval_v4)으로 검색 고도화 후보를 비교 (답변 생성 없음).

  python experiments/eval_v2_yjk.py --eval-file data/eval_v2.json --index-dir indexes/eval-v2-structured-yjk
  python experiments/eval_v2_yjk.py ... --settings hybrid-ce-c50-cap2 prefix-ce-c50-cap2 --allow-api
  .venv-a/bin/python experiments/eval_v2_yjk.py --eval-file data/eval_v4.json --index-dir indexes/scenario-a --settings default

.env의 RFP_SCENARIO=A면 질문을 서버의 A 임베딩 모델(A_EMBEDDING_MODEL)로 임베딩합니다(API·질문 캐시 없음, 생성 모델은 안 불러옴).
인덱스도 같은 A 모델로 만든 것이어야 합니다. 리랭커도 A 공용 폴더(A_RERANK_MODEL)의 모델을 씁니다.
eval_v4의 답 없는 문항(answerable=false)은 검색 지표에서 빼고, 설정마다 빈 결과로 돌아온 수만 따로 봅니다.

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
import platform
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
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
# 'default'는 retrieve 기본값 그대로(회귀 점검용). '+rw'는 질문 재작성 both(원문 + 재작성 RRF), '+rwo'는 only.
SETTINGS["default"] = {}
SETTINGS.update({f"{name}{suffix}": {**options, "rewrite": mode} for name, options in
                 [("default", {}), ("prefix-hybrid-ce-c50-cap2+x5", SETTINGS["prefix-hybrid-ce-c50-cap2+x5"]),
                  ("prefix-hybrid-lex-c50-cap2+x5", SETTINGS["prefix-hybrid-lex-c50-cap2+x5"])]
                 for suffix, mode in (("+rw", "both"), ("+rwo", "only"))})
# '+dd'는 중복 처리(retrieve의 dedup): 본문이 같은 문서는 한 문서로 상한을 세고, 같은 본문 청크는 한 번만
SETTINGS["default+dd"] = {"dedup": True}
# '+sec'는 재정렬 입력에 절 제목 추가(retrieve의 rerank_section)
SETTINGS["default+sec"] = {"rerank_section": True}


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
    parser.add_argument("--rewrite-cache", type=Path, default=ROOT / "results/cache/query_rewrites.json",
                        help="질문 재작성 결과 캐시(같은 질문은 다시 호출하지 않음)")
    parser.add_argument("--allow-cpu-rerank", action="store_true",
                        help="GPU가 없을 때 cross-encoder를 CPU로 돌리는 것을 허용(기본은 중단)")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from embedding import embed_texts, load_index
    from experiment_ids import next_experiment_id, owner_initials
    import retrieval

    load_dotenv(ROOT / ".env", override=False)
    scenario_a = os.getenv("RFP_SCENARIO", "B").strip().upper() == "A"
    cases = read_json(args.eval_file)
    index, chunks, config = load_index(args.index_dir)
    uses_cross_encoder = any(retrieval.retrieval_options(**SETTINGS[name])["rerank"] == "cross-encoder"
                             for name in args.settings)
    if uses_cross_encoder and not args.allow_cpu_rerank:
        import torch
        if not torch.cuda.is_available():
            parser.error("cross-encoder 설정은 GPU에서만 돌립니다. CUDA가 없습니다(--allow-cpu-rerank로 CPU 허용).")
    model = config["embedding_model"]
    if scenario_a:
        # 질문은 retrieve가 부를 때마다 A 모델로 바로 임베딩하므로 미리 채울 캐시가 없습니다.
        from run import A_EMBEDDING_MODEL_DIR, A_GENERATION_MODEL_DIR, read_scenario_settings
        from scenario_a import ScenarioAClient
        client = ScenarioAClient(read_scenario_settings(ROOT / ".env"), A_GENERATION_MODEL_DIR, A_EMBEDDING_MODEL_DIR,
                                 need_generation=False)
        client.api_calls, client.save = 0, lambda: None  # 아래 B용 코드(API 호출 수 기록·캐시 저장)를 그대로 쓰기 위한 값
        missing = []
    else:
        client = CachedEmbeddingClient(args.cache_file)
        if args.fresh_query_embeddings:
            client.cache, client.save = {}, lambda: None
            args.allow_api = True
        questions = sorted({case["question"] for case in cases})
        missing = [q for q in questions if hashlib.sha256(f"{model}\n{q}".encode()).hexdigest() not in client.cache]
        if missing and not args.allow_api:
            parser.error(f"캐시에 없는 질문 {len(missing)}건은 임베딩 API를 호출합니다. 확인 후 --allow-api를 붙이세요.")
    experiment_id = next_experiment_id(owner_initials(os.getenv("EXPERIMENT_OWNER", "")), ROOT / ".experiment-state")
    print(f"실험 ID: {experiment_id} / 설정 {len(args.settings)}개 / 질문 {len(cases)}건 / 임베딩할 질문 {len(missing)}건")
    for start in range(0, len(missing), 64):
        embed_texts(missing[start:start + 64], client, model)
    client.save()

    # 점수 캐시가 이미 있으면 리랭커를 다시 돌리지 않아 처리 시간이 짧게 나옵니다(지표는 같음). 보고서에 남겨 둡니다.
    ce_cache_existed = Path(args.ce_cache_file).is_file()
    ce_original, ce_cached, ce_save = cached_cross_encoder(retrieval, args.ce_cache_file)
    retrieval._cross_encoder_scores = ce_cached
    ce_load_ms, rerank_device = 0.0, None
    if uses_cross_encoder:
        # 점수 캐시를 거치지 않고 모델을 직접 불러옵니다(캐시에 있으면 모델이 안 올라와 장치·로드 시간을 못 잼).
        model_name = retrieval.rerank_model_source()  # A면 공용 폴더 경로, B면 HF 모델 ID
        start = time.perf_counter()
        ce_original(cases[0]["question"], ["모델 로드 확인"], model_name)
        ce_load_ms = (time.perf_counter() - start) * 1000
        rerank_device = retrieval._cross_encoders[model_name][2]

    # 질문 재작성 결과(cache·api·fallback-*)를 설정마다 세어, 재작성이 실제로 쓰였는지 보고서에 남깁니다.
    import query_rewrite
    rewrite_status = Counter()
    rewrite_original = query_rewrite.rewrite_question

    def counted_rewrite(*call_args, **call_kwargs):
        text, info = rewrite_original(*call_args, **call_kwargs)
        rewrite_status[info["status"]] += 1
        return text, info
    query_rewrite.rewrite_question = counted_rewrite

    # eval_v2는 확장 문항에 bm25_reference가 없어 기준값 열은 싣지 않습니다.
    results = []
    for name in args.settings:
        rows, times = [], []
        rewrite_status.clear()
        for case in cases:
            filters = resolve_filters(case.get("filters"), chunks) if name.endswith("+fuzzy") else case.get("filters")
            start = time.perf_counter()
            hits = retrieval.retrieve(case["question"], client, index, chunks, config, args.top_k, filters,
                                      rewrite_cache=args.rewrite_cache, **SETTINGS[name])
            times.append((time.perf_counter() - start) * 1000)
            # dup_texts: 상위 결과 중 앞 순위와 본문이 똑같아 자리만 차지한 청크 수(중복 문서·같은 서식 문구)
            rows.append({"id": case["id"], **score_case(case, hits), "hits": [hit["chunk_id"] for hit in hits],
                         "dup_texts": len(hits) - len({hit["text"] for hit in hits})})
        ordered = sorted(times)
        results.append((name, {"options": {**retrieval.retrieval_options(**SETTINGS[name]),
                                           "fuzzy_filter": name.endswith("+fuzzy"),
                                           "rewrite_status": dict(rewrite_status)},
                               "rows": rows, "mean_ms": statistics.fmean(times),
                               "p95_ms": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)]}))
        ce_save()
        print(f"{name}: 완료 ({statistics.fmean(times):.0f}ms/질문)", flush=True)
    retrieval._cross_encoder_scores = ce_original
    query_rewrite.rewrite_question = rewrite_original
    for _, result in results:
        result["summary"] = summarize_groups(cases, result["rows"])
    if client.api_calls:
        client.save()

    # 같은 조건에서 돌렸는지 나중에 확인할 수 있도록 코드·인덱스·모델·실행 환경을 남깁니다.
    # 코드: 커밋 해시. 커밋하지 않은 수정이 있으면 뒤에 +dirty를 붙입니다.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip():
        commit += "+dirty"
    # 모델: 폴더 이름만으로는 안에 든 모델(bge-m3, e5 등)이 바뀌어도 알 수 없어 가중치 파일 해시를 남깁니다.
    # A만 해당합니다(B 임베딩은 OpenAI 모델 이름, B 리랭커는 HF 모델 ID로 이미 구분됨).
    model_dirs = {"embedding": client.embedding_path} if scenario_a else {}
    if scenario_a and uses_cross_encoder:
        model_dirs["cross_encoder"] = Path(retrieval.rerank_model_source())
    weights_sha256 = {"embedding": None, "cross_encoder": None}
    for kind, folder in model_dirs.items():
        digest = hashlib.sha256()
        for path in sorted(p for p in folder.rglob("*") if p.suffix in (".safetensors", ".bin", ".pt", ".onnx")):
            with path.open("rb") as f:
                digest.update(path.relative_to(folder).as_posix().encode() + hashlib.file_digest(f, "sha256").digest())
        weights_sha256[kind] = digest.hexdigest()
    # 실행 환경: torch는 리랭커나 A 임베딩을 쓸 때만 설치돼 있으므로 그때만 적습니다.
    environment = {"host": socket.gethostname(), "python": platform.python_version()}
    if uses_cross_encoder or scenario_a:
        import torch
        import transformers
        environment.update({"torch": torch.__version__, "transformers": transformers.__version__,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"git_commit": commit, "experiment_id": experiment_id, "measured_at_utc": stamp, "eval_file": relative(args.eval_file),
            "evaluation_sha256": file_hash(args.eval_file), "cases": len(cases), "index_dir": relative(args.index_dir),
            "chunks_sha256": file_hash(Path(args.index_dir) / "chunks.json"), "chunk_count": len(chunks),
            # 임베딩 모델을 바꿔 다시 만들면 청크는 같아도 벡터가 달라지므로 index.faiss도 해시합니다.
            "index_faiss_sha256": hashlib.sha256((Path(args.index_dir) / "index.faiss").read_bytes()).hexdigest(),
            "documents_sha256": config.get("documents_sha256"),
            "chunking_strategy": config.get("chunking_strategy"),
            "embedding_context": config.get("embedding_context"), "embedding_model": model, "top_k": args.top_k,
            "scenario": "A" if scenario_a else "B",
            "cross_encoder_model": retrieval.rerank_model_source() if uses_cross_encoder else None,
            "embedding_weights_sha256": weights_sha256["embedding"],
            "cross_encoder_weights_sha256": weights_sha256["cross_encoder"],
            "cross_encoder_score_cache_used": ce_cache_existed, **environment,
            "cross_encoder_load_ms": round(ce_load_ms), "embedding_api_calls": client.api_calls,
            "fresh_query_embeddings": args.fresh_query_embeddings, "rerank_device": rerank_device}
    write_json(Path(args.results_dir) / f"eval_v2_{experiment_id}_{stamp}.json", {"meta": meta, "results": dict(results)})
    lines = report_lines(meta, cases, results)
    lines[0] = f"# {Path(args.eval_file).stem} 검색 지표: 하이브리드·리랭킹·BM25 메타 접두·필터 정규화"
    lines = [line.replace("상위 50개 안에서만", f"상위 {args.top_k}개 안에서만") for line in lines]
    # 정답 지표에는 안 드러나는 '같은 내용이 자리를 겹쳐 차지했는지'를 따로 봅니다(dedup 비교용).
    lines += ["", f"## 상위 {args.top_k}개 안 중복 본문", "",
              "| 설정 | 중복 본문 자리 합계 | 중복이 있는 질문 수 | 평균 반환 수 |", "| --- | ---: | ---: | ---: |"]
    for name, result in results:
        rows = result["rows"]
        lines.append(f"| {name} | {sum(row['dup_texts'] for row in rows)} | {sum(row['dup_texts'] > 0 for row in rows)} | "
                     f"{statistics.fmean(row['returned'] for row in rows):.2f} |")
    # 답 없는 문항: 검색 지표에서 빼고, 빈 결과(min_score 차단 등)로 돌아온 수만 봅니다.
    unanswerable = [i for i, case in enumerate(cases) if case.get("answerable") is False]
    if unanswerable:
        lines += ["", f"## 답 없는 문항 ({len(unanswerable)}문항, 검색 지표에서 제외)", "",
                  "| 설정 | 빈 결과로 돌아온 문항 | min_score |", "| --- | ---: | --- |"]
        for name, result in results:
            empty = sum(result["rows"][i]["returned"] == 0 for i in unanswerable)
            lines.append(f"| {name} | {empty} / {len(unanswerable)} | {result['options']['min_score']} |")
    report_path = Path(args.reports_dir) / f"{Path(args.eval_file).stem}_yjk_{experiment_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"보고서: {relative(report_path)}")


if __name__ == "__main__":
    main()
