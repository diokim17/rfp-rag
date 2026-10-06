"""김연주: 리랭킹 없음 / lexical / 하이브리드 / 질문 재작성 / 하이브리드+재작성 검색 비교 (답변 생성 없음).

  python experiments/hybrid_eval_yjk.py --eval-file data/eval_retrieval_yjk.json                 # API 없는 설정만
  python experiments/hybrid_eval_yjk.py --eval-file data/eval_retrieval_yjk.json --plan-rewrite  # 재작성 호출 수만 계산
  python experiments/hybrid_eval_yjk.py --eval-file ... --pilot 10 --allow-api        # 재작성 10건만 시험 호출
  python experiments/hybrid_eval_yjk.py --eval-file ... --settings rewrite-only-lexical --allow-api

- 지표: Recall@k·MRR@k (retrieval_metrics_yjk.py, 문서 단위 정답), K=3·5·10.
  설정마다 top_k=10으로 한 번 검색해 앞에서 3·5·10개를 잘라 계산합니다(top_k가 순위를 바꾸지 않으므로 같은 값).
- 그룹: 사업명 있음(A·E) / 일부(B) / 없음(C·D)과 전체.
- 시간: 질문당 검색 시간의 평균과 상위 5%(p95). 질문 임베딩·재작성은 측정 전에 캐시에 채우고,
  BM25 색인 생성 시간은 따로 기록합니다.
- 결과에는 실험 ID, 설정별로 retrieve가 실제 적용한 옵션(retrieval_options), 청크·인덱스·평가셋 해시를 남깁니다.
"""

import argparse
import hashlib
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing import read_json, write_json  # noqa: E402
from retrieval_eval_yjk import CachedEmbeddingClient  # noqa: E402
from retrieval_metrics_yjk import score_case  # noqa: E402

KS = (3, 5, 10)
NAME_GROUPS = {"A_full_name": "사업명 있음", "E_filter_named": "사업명 있음", "B_short_name": "사업명 일부",
               "C_no_name": "사업명 없음", "D_filter_agency": "사업명 없음"}
GROUPS = ("사업명 있음", "사업명 일부", "사업명 없음", "전체")
# 이름 → retrieve 옵션. 재작성이 들어간 설정은 API가 필요합니다.
SETTINGS = {
    "none": {"rerank": "none"},
    "lexical-c100": {"rerank": "lexical", "candidates": 100},
    "hybrid": {"rerank": "none", "hybrid": True, "candidates": 100},
    "hybrid-lexical-c100": {"rerank": "lexical", "candidates": 100, "hybrid": True},
    "hybrid-lexical-c100-v50-b50": {"rerank": "lexical", "candidates": 100, "hybrid": True,
                                     "hybrid_vector_k": 50, "hybrid_bm25_k": 50},
    "hybrid-lexical-c200": {"rerank": "lexical", "candidates": 200, "hybrid": True,
                            "hybrid_vector_k": 100, "hybrid_bm25_k": 100},
    # cross-encoder 비교: 하이브리드 후보(벡터 100 + BM25 100, RRF)는 같게 두고 상위 candidates개만 재정렬
    **{f"hybrid-ce-c{n}": {"rerank": "cross-encoder", "candidates": n, "hybrid": True,
                           "hybrid_vector_k": 100, "hybrid_bm25_k": 100} for n in (30, 50, 100)},
    "hybrid-lexical-c100-cap2": {"rerank": "lexical", "candidates": 100, "hybrid": True, "max_per_doc": 2},
    "rewrite-only-lexical": {"rerank": "lexical", "candidates": 100, "rewrite": "only"},
    "rewrite-both-lexical": {"rerank": "lexical", "candidates": 100, "rewrite": "both"},
    "hybrid-rewrite-only-lexical": {"rerank": "lexical", "candidates": 100, "hybrid": True, "rewrite": "only"},
    "hybrid-rewrite-both-lexical": {"rerank": "lexical", "candidates": 100, "hybrid": True, "rewrite": "both"},
}
NO_API = [name for name, options in SETTINGS.items() if options.get("rewrite", "off") == "off"
          and options.get("rerank") != "cross-encoder"]  # cross-encoder는 GPU에서 오래 걸려 직접 지정
DISCLAIMER = ("평가셋은 개인이 CSV 메타데이터로 자동 구성한 문서 단위 정답 325문항이며 팀 공통 평가셋이 아닙니다. "
              "답변 생성과 답변 사실성은 측정하지 않습니다.")


class EvalClient(CachedEmbeddingClient):
    """질문 임베딩은 파일 캐시, 재작성(responses)은 실제 API (재시도 1회)."""

    def __init__(self, cache_path, allow_api):
        super().__init__(cache_path)
        self.allow_api, self._responses = allow_api, None
        self.responses = SimpleNamespace(create=self.respond)
        self.rewrite_calls = 0

    def client(self):
        if not self.allow_api:
            raise RuntimeError("API 호출이 허용되지 않았습니다 (--allow-api).")
        return super().client()

    def respond(self, **kwargs):
        if self._responses is None:
            from openai import OpenAI
            self.client()  # .env 로드와 허용 여부 확인
            self._responses = OpenAI(timeout=60.0, max_retries=1).responses
        self.rewrite_calls += 1
        return self._responses.create(**kwargs)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def plan_rewrite(cases, model, cache_path):
    """API를 부르지 않고 재작성 호출 수와 토큰 한도를 계산합니다."""
    from query_rewrite import MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, RewriteCache, cache_key, estimate_input_tokens

    cache = RewriteCache.open(cache_path)
    questions = sorted({case["question"] for case in cases})
    todo = [q for q in questions if cache_key(q, model) not in cache.items]
    estimates = [estimate_input_tokens(q) for q in todo]
    callable_ = [e for e in estimates if e <= MAX_INPUT_TOKENS]
    return {"model": model, "unique_questions": len(questions), "cached": len(questions) - len(todo),
            "rewrite_calls": len(callable_), "over_input_limit": len(estimates) - len(callable_),
            "estimated_input_tokens_max": max(estimates, default=0),
            "estimated_input_tokens_mean": round(statistics.fmean(estimates), 1) if estimates else 0,
            "estimated_input_tokens_total": sum(callable_),
            "output_tokens_cap_total": len(callable_) * MAX_OUTPUT_TOKENS,
            "embedding_calls_for_rewrites": math.ceil(len(callable_) / 64),
            "limits": {"input_tokens_per_call": MAX_INPUT_TOKENS, "output_tokens_per_call": MAX_OUTPUT_TOKENS}}


def prefill_rewrites(cases, client, config, model, cache_path):
    """재작성과 재작성 질문 임베딩을 측정 전에 채웁니다. 반환: 호출 기록."""
    from embedding import embed_texts
    from query_rewrite import rewrite_question

    statuses, usage, rewritten = {}, {"input_tokens": 0, "output_tokens": 0}, []
    start = time.perf_counter()
    for question in sorted({case["question"] for case in cases}):
        text, info = rewrite_question(question, client, model, cache_path)
        statuses[info["status"]] = statuses.get(info["status"], 0) + 1
        for key in usage:
            usage[key] += (info.get("usage") or {}).get(key) or 0
        rewritten.append(text)
    rewrite_ms = (time.perf_counter() - start) * 1000
    before = client.api_calls
    missing = sorted(set(rewritten))
    for begin in range(0, len(missing), 64):
        embed_texts(missing[begin:begin + 64], client, config["embedding_model"])
    client.save()
    return {"model": model, "statuses": statuses, "rewrite_api_requests": client.rewrite_calls,
            "usage": usage, "rewrite_ms": round(rewrite_ms), "embedding_api_calls": client.api_calls - before}


def run_setting(name, options, cases, client, index, chunks, config, rewrite_cache):
    from retrieval import retrieval_options, retrieve

    rows, times = [], []
    for case in cases:
        start = time.perf_counter()
        hits = retrieve(case["question"], client, index, chunks, config, max(KS), case.get("filters"),
                        rewrite_cache=rewrite_cache, **options)
        times.append((time.perf_counter() - start) * 1000)
        docs = [hit["doc_id"] for hit in hits]
        rows.append({"id": case.get("id", str(len(rows))), "group": case.get("group"),
                     "name_group": NAME_GROUPS.get(case.get("group"), "기타"), "hits": docs,
                     **{f"k{k}": score_case(case["expected_doc_ids"], docs, k) for k in KS}})
    ordered = sorted(times)
    summary = {}
    for group in GROUPS:
        members = [row for row in rows if group == "전체" or row["name_group"] == group]
        if members:
            summary[group] = {"cases": len(members), **{
                f"{metric}@{k}": statistics.fmean(row[f"k{k}"][metric] for row in members)
                for k in KS for metric in ("recall", "mrr")}}
    return {"setting": name, "options": retrieval_options(**options), "summary": summary,
            "mean_ms": statistics.fmean(times), "p95_ms": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
            "max_ms": ordered[-1]}, rows


def report_lines(meta, results, rewrite_info):
    lines = ["# 하이브리드 검색·질문 재작성 비교 (검색 단계)", "", DISCLAIMER, "",
             "| 항목 | 값 |", "| --- | --- |", *(f"| {k} | `{v}` |" for k, v in meta.items()), ""]
    for group in GROUPS:
        lines += [f"## {group}", "", "| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for result in results:
            row = result["summary"].get(group)
            if row:
                lines.append(f"| {result['setting']} | " + " | ".join(
                    f"{row[f'recall@{k}']:.3f}" for k in KS) + " | " + " | ".join(f"{row[f'mrr@{k}']:.3f}" for k in KS) + " |")
        lines.append("")
    lines += ["## 처리 시간과 적용 옵션", "", "| 설정 | 평균(ms) | 상위 5%(ms) | 적용 옵션 |", "| --- | ---: | ---: | --- |"]
    for result in results:
        enabled = {k: v for k, v in result["options"].items() if v not in (None, False, "off", "none")}
        lines.append(f"| {result['setting']} | {result['mean_ms']:.1f} | {result['p95_ms']:.1f} | `{enabled}` |")
    if rewrite_info:
        lines += ["", "## 질문 재작성 호출", "", f"`{rewrite_info}`"]
    lines += ["", "처리 시간은 retrieve() 한 번의 시간으로 벡터 검색·필터·BM25·RRF·리랭킹·문서당 상한을 포함합니다. 질문 임베딩·재작성은 캐시를 썼으므로 API 지연은 빠져 있고, BM25 색인 생성과 cross-encoder 모델 로드 시간도 제외했습니다.", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes/parsing-v2-yjk")
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=NO_API)
    parser.add_argument("--cache-file", type=Path, default=ROOT / "results/cache/query_embeddings.json")
    parser.add_argument("--rewrite-cache", type=Path, default=ROOT / "results/cache/query_rewrite_gpt-5-mini.json")
    parser.add_argument("--rewrite-model", default="gpt-5-mini")
    parser.add_argument("--allow-api", action="store_true", help="재작성·임베딩 API 호출을 허용")
    parser.add_argument("--plan-rewrite", action="store_true", help="API 없이 재작성 예상 호출 수만 계산")
    parser.add_argument("--pilot", type=int, help="앞쪽 N개 질문만 재작성해 실제 토큰 사용량을 확인하고 종료")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    args = parser.parse_args()

    cases = read_json(args.eval_file)
    if args.plan_rewrite:
        for key, value in plan_rewrite(cases, args.rewrite_model, args.rewrite_cache).items():
            print(f"{key}: {value}")
        return
    if args.pilot:
        if not args.allow_api:
            parser.error("--pilot은 재작성 API를 호출합니다. 확인 후 --allow-api를 붙이세요.")
        from query_rewrite import MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, rewrite_question

        client = EvalClient(args.cache_file, True)
        questions = list(dict.fromkeys(case["question"] for case in cases))[:args.pilot]
        rows = [rewrite_question(q, client, args.rewrite_model, args.rewrite_cache)[1] for q in questions]
        used = [row.get("usage") or {} for row in rows]
        print(f"시험 재작성 {len(rows)}건 / API 요청 {client.rewrite_calls}회 / 상태 "
              f"{ {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}} }")
        for key, limit in (("input_tokens", MAX_INPUT_TOKENS), ("output_tokens", MAX_OUTPUT_TOKENS)):
            values = [u[key] for u in used if u.get(key) is not None]
            if values:
                print(f"{key}: 최대 {max(values)}, 평균 {statistics.fmean(values):.0f} (한도 {limit}), "
                      f"한도 초과 {sum(v > limit for v in values)}건")
        return
    needs_api = [name for name in args.settings if SETTINGS[name].get("rewrite", "off") != "off"]
    if needs_api and not args.allow_api:
        parser.error(f"재작성 설정({', '.join(needs_api)})은 API를 호출합니다. 확인 후 --allow-api를 붙이세요.")
    for name in needs_api:
        SETTINGS[name]["rewrite_model"] = args.rewrite_model

    from dotenv import load_dotenv
    from embedding import load_index
    from experiment_ids import next_experiment_id, owner_initials
    import retrieval

    load_dotenv(ROOT / ".env", override=False)  # 키는 환경 변수로만 읽고 결과에 남기지 않습니다.
    experiment_id = next_experiment_id(owner_initials(os.getenv("EXPERIMENT_OWNER", "")), ROOT / ".experiment-state")
    index, chunks, config = load_index(args.index_dir)
    client = EvalClient(args.cache_file, args.allow_api)
    print(f"실험 ID: {experiment_id} / 설정 {len(args.settings)}개 / 질문 {len(cases)}건")

    rewrite_info = None
    if needs_api:
        rewrite_info = prefill_rewrites(cases, client, config, args.rewrite_model, args.rewrite_cache)
        print(f"재작성 준비: {rewrite_info}")
    bm25_ms = 0.0
    if any(SETTINGS[name].get("hybrid") for name in args.settings):
        start = time.perf_counter()
        retrieval._bm25_index(chunks)
        bm25_ms = (time.perf_counter() - start) * 1000
        print(f"BM25 색인 생성: {bm25_ms:.0f}ms")

    ce_load_ms = 0.0
    if any(SETTINGS[name].get("rerank") == "cross-encoder" for name in args.settings):
        start = time.perf_counter()  # 모델 로드·GPU 초기화를 측정에서 분리
        retrieval.retrieve(cases[0]["question"], client, index, chunks, config, 1, None,
                           rerank="cross-encoder", candidates=1)
        ce_load_ms = (time.perf_counter() - start) * 1000
        print(f"cross-encoder 모델 준비: {ce_load_ms:.0f}ms")

    embedding_calls_before = client.api_calls
    results, records = [], []
    for name in args.settings:
        result, rows = run_setting(name, dict(SETTINGS[name]), cases, client, index, chunks, config, args.rewrite_cache)
        results.append(result)
        records += [{"setting": name, **row} for row in rows]
        row = result["summary"]["전체"]
        print(f"{name}: Recall@3/5/10 {row['recall@3']:.3f}/{row['recall@5']:.3f}/{row['recall@10']:.3f} "
              f"MRR@5 {row['mrr@5']:.3f} / 평균 {result['mean_ms']:.1f}ms, p95 {result['p95_ms']:.1f}ms")
    if client.api_calls != embedding_calls_before:
        sys.exit("측정 중에 임베딩 API가 호출되었습니다. 캐시를 확인하세요.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"experiment_id": experiment_id, "measured_at_utc": stamp, "eval_file": args.eval_file.name,
            "evaluation_sha256": file_hash(args.eval_file), "cases": len(cases), "index_dir": relative(args.index_dir),
            "chunks_file": "chunks.json", "chunks_sha256": file_hash(args.index_dir / "chunks.json"),
            "index_sha256": file_hash(args.index_dir / "index.faiss"), "chunk_count": config["chunk_count"],
            "embedding_model": config["embedding_model"], "ks": list(KS), "bm25_build_ms": round(bm25_ms),
            "cross_encoder_model": os.getenv("RETRIEVAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3") if ce_load_ms else None,
            "cross_encoder_load_ms": round(ce_load_ms)}
    output = args.results_dir / f"hybrid_eval_{experiment_id}_{stamp}.json"
    write_json(output, {"settings": meta, "disclaimer": DISCLAIMER, "rewrite": rewrite_info,
                        "results": results, "records": records})
    report = args.reports_dir / f"hybrid_eval_yjk_{experiment_id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(report_lines(meta, results, rewrite_info)), encoding="utf-8")
    print(f"결과 저장: {output}\n보고서: {report}")


if __name__ == "__main__":
    main()
