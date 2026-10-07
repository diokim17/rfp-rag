"""김연주: 검색 단계만 비교하는 개인 실험 도구 (생성 API 호출 없음).

  python experiments/retrieval_eval_yjk.py make-evalset --processed-dir ... --eval-file ...
  python experiments/retrieval_eval_yjk.py run --index-dir ... --eval-file ... --results-dir ... --label baseline

평가셋은 CSV 메타데이터로 정답 문서가 확정되는 질문만 자동 구성합니다(팀 공통 평가셋 아님).
질문 임베딩은 results-dir에 캐시하여 같은 질문으로 설정만 바꿔 비교할 때 API를 다시 호출하지 않습니다.
"""

import argparse
import hashlib
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parsing import read_json, write_json  # noqa: E402

ASPECTS = ["사업 예산은 얼마인가요?", "사업 기간은 어떻게 되나요?", "입찰 참가 자격은 무엇인가요?",
           "제안서 평가 방법은 어떻게 되나요?", "주요 요구사항은 무엇인가요?", "추진 배경과 목적은 무엇인가요?"]


def make_evalset(processed_dir, eval_file, seed=20261001):
    documents = read_json(Path(processed_dir) / "documents.json")
    rng = random.Random(seed)
    by_name, by_agency = defaultdict(list), defaultdict(list)
    for doc in documents:
        by_name[doc["metadata"]["사업명"]].append(doc["doc_id"])
        by_agency[doc["metadata"]["발주 기관"]].append(doc["doc_id"])
    cases = []
    for i, doc in enumerate(documents):
        meta, expected = doc["metadata"], sorted(by_name[doc["metadata"]["사업명"]])
        aspect = ASPECTS[i % len(ASPECTS)]
        # A: 사업명 전체를 언급한 질문
        cases.append({"group": "A_full_name", "question": f"{meta['사업명']} 사업의 {aspect}",
                      "expected_doc_ids": expected})
        # B: 기관명 + 사업명 뒤쪽 일부만 언급 (축약 표현)
        words = re.sub(r"[\[\]()<>「」『』‘’“”\"'·,]", " ", meta["사업명"]).split()
        cases.append({"group": "B_short_name", "question": f"{meta['발주 기관']}의 {' '.join(words[-3:])} 관련 {aspect}",
                      "expected_doc_ids": expected})
        # C: 사업명·기관명 없이 사업 요약의 한 줄로만 묘사
        lines = [re.sub(r"^[-•\s]+", "", line).strip() for line in meta.get("사업 요약", "").splitlines()]
        lines = [re.sub(r"^[^:]{1,12}:\s*", "", line) for line in lines[1:]]
        lines = [line for line in lines if len(line) >= 25 and meta["사업명"][:8] not in line]
        if lines:
            cases.append({"group": "C_no_name", "question": f"다음 내용의 사업에서 {aspect} {rng.choice(lines)}",
                          "expected_doc_ids": [doc["doc_id"]]})
    for agency, ids in sorted(by_agency.items()):
        if len(ids) < 2:
            continue
        # D: 같은 기관의 문서가 여럿일 때 기관 필터 + 일반 질문 → 해당 기관 문서 전부
        cases.append({"group": "D_filter_agency", "question": "이 기관이 발주한 사업들의 주요 요구사항은 무엇인가요?",
                      "expected_doc_ids": sorted(ids), "filters": {"발주 기관": agency}})
        # E: 기관 필터 + 특정 사업 질문
        for doc in documents:
            if doc["doc_id"] in ids:
                cases.append({"group": "E_filter_named", "question": f"{doc['metadata']['사업명']} 사업의 {rng.choice(ASPECTS)}",
                              "expected_doc_ids": sorted(by_name[doc["metadata"]["사업명"]]),
                              "filters": {"발주 기관": agency}})
    write_json(eval_file, cases)
    counts = defaultdict(int)
    for case in cases:
        counts[case["group"]] += 1
    print(f"평가셋 저장: {eval_file} ({len(cases)}건) {dict(counts)}")


class CachedEmbeddingClient:
    """질문 임베딩만 캐시합니다. 그 외 호출(리랭킹 등)은 실제 client에 위임합니다."""

    def __init__(self, cache_path):
        self.path, self.real = Path(cache_path), None
        self.cache = read_json(self.path) if self.path.is_file() else {}
        self.api_calls = 0
        self.embeddings = SimpleNamespace(create=self.embed)

    def client(self):
        if self.real is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(ROOT / ".env", override=False)
            self.real = OpenAI(timeout=60.0, max_retries=2)
        return self.real

    def embed(self, model, input):
        keys = [hashlib.sha256(f"{model}\n{text}".encode()).hexdigest() for text in input]
        missing = [i for i, key in enumerate(keys) if key not in self.cache]
        if missing:
            response = self.client().embeddings.create(model=model, input=[input[i] for i in missing])
            self.api_calls += 1
            for item in response.data:
                self.cache[keys[missing[item.index]]] = item.embedding
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=self.cache[key])
                                     for i, key in enumerate(keys)], usage=None)

    def save(self):
        write_json(self.path, self.cache)


def run(index_dir, eval_file, results_dir, reports_dir, label, top_k):
    from embedding import embed_texts, load_index
    from retrieval import retrieve

    cases = read_json(eval_file)
    index, chunks, config = load_index(index_dir)
    results_dir = Path(results_dir)
    client = CachedEmbeddingClient(results_dir / "query_embedding_cache.json")
    questions = sorted({case["question"] for case in cases})
    for start in range(0, len(questions), 64):  # 캐시 채우기: 측정 시간에서 API 지연을 분리
        embed_texts(questions[start:start + 64], client, config["embedding_model"])
    client.save()
    # 검색 기본값이 바뀌어도 이 실험은 예전 기본값(코사인 검색, 리랭킹은 RETRIEVAL_RERANK로만 켬)을 유지
    base = {"rerank": os.getenv("RETRIEVAL_RERANK") or "none", "hybrid": False, "max_per_doc": "none"}
    retrieve(cases[0]["question"], client, index, chunks, config, top_k, cases[0].get("filters"), **base)  # 모델 로드 등 준비

    groups, records = defaultdict(list), []
    for case in cases:
        start = time.perf_counter()
        hits = retrieve(case["question"], client, index, chunks, config, top_k, case.get("filters"), **base)
        seconds = time.perf_counter() - start
        expected = set(case["expected_doc_ids"])
        docs = [hit["doc_id"] for hit in hits]
        first = next((rank for rank, doc in enumerate(docs, 1) if doc in expected), None)
        violations = sum(any(str(hit["metadata"].get(key)) != str(value) for key, value in (case.get("filters") or {}).items())
                         for hit in hits)
        record = {"group": case["group"], "recall_at_k": len(expected & set(docs)) / len(expected),
                  "mrr": 1 / first if first else 0.0,
                  "doc_precision": sum(doc in expected for doc in docs) / len(docs) if docs else 0.0,
                  "top1": float(bool(docs) and docs[0] in expected), "filter_violations": violations,
                  "hit_count": len(hits), "seconds": seconds}
        groups[case["group"]].append(record)
        groups["ALL"].append(record)
        records.append({**record, "question": case["question"], "expected_doc_ids": sorted(expected),
                        "hits": [{k: hit.get(k) for k in ("doc_id", "chunk_id", "score", "rerank_score")} for hit in hits]})

    settings = {"label": label, "top_k": top_k, **config,
                **{key: os.getenv(key) for key in ("RETRIEVAL_RERANK", "RETRIEVAL_CANDIDATES", "RETRIEVAL_RERANK_MODEL")}}
    summary = {}
    for name in sorted(groups):
        rows = groups[name]
        summary[name] = {"count": len(rows),
                         **{metric: sum(r[metric] for r in rows) / len(rows)
                            for metric in ("recall_at_k", "mrr", "top1", "doc_precision")},
                         "filter_violations": sum(r["filter_violations"] for r in rows),
                         "median_ms": statistics.median(r["seconds"] for r in rows) * 1000,
                         "max_ms": max(r["seconds"] for r in rows) * 1000}
    write_json(results_dir / f"retrieval_eval_{label}.json", {"settings": settings, "summary": summary, "records": records})

    # 공유용 요약: 질문·원문·필터 값 없이 집계 지표만 기록
    lines = [f"# 검색 단계 비교 요약: {label}", "", "| 설정 | 값 |", "| --- | --- |"]
    lines += [f"| {key} | `{value}` |" for key, value in settings.items()]
    lines += [f"| evaluation_sha256 | `{hashlib.sha256(Path(eval_file).read_bytes()).hexdigest()}` |",
              f"| query_embedding_api_calls | `{client.api_calls}` |", "",
              "| 그룹 | 건수 | Recall@k | MRR@k | Top-1 | 문서 정밀도@k | 필터 위반 | 중앙 지연(ms) | 최대 지연(ms) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in summary.items():
        lines.append(f"| {name} | {row['count']} | {row['recall_at_k']:.3f} | {row['mrr']:.3f} | {row['top1']:.3f} | "
                     f"{row['doc_precision']:.3f} | {row['filter_violations']} | {row['median_ms']:.1f} | {row['max_ms']:.1f} |")
    lines += ["", "지연은 질문 임베딩 캐시를 사용한 검색·리랭킹 처리 시간이며 임베딩 API 지연은 제외됩니다.",
              "평가셋은 CSV 메타데이터로 자동 구성한 문서 단위 정답이며 청크 단위 적합성과 답변 사실성은 측정하지 않습니다.", ""]
    report = Path(reports_dir) / f"retrieval_eval_yjk_{label}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-len(summary) - 5:-3]))
    print(f"요약: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["make-evalset", "run"])
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "make-evalset":
        make_evalset(args.processed_dir, args.eval_file)
    else:
        run(args.index_dir, args.eval_file, args.results_dir, args.reports_dir, args.label, args.top_k)
