"""김연주: 같은 문서 안에서 청크를 다시 고르는 2단계 검색 실험 (eval_v2, 답변 생성 없음).

  python experiments/within_doc_reselect_yjk.py --eval-file data/eval_v2.json --index-dir indexes/eval-v2-structured-yjk

1단계: retrieve(하이브리드 + BM25 접두, 리랭킹 없음)로 상위 50개를 받습니다.
2단계: 순위에서 문서가 차지한 자리는 그대로 두고, 그 자리에 들어갈 청크만 그 문서의 전체 청크에서 다시 고릅니다.
  따라서 문서 단위 지표(doc_recall@5, 정답 문서 순위)는 1단계와 같고 정답 청크 지표만 달라집니다.
  다시 고를 때는 질문에서 그 문서의 사업명·발주 기관과 겹치는 단어를 뺀 '남은 질문'을 씁니다.
  - residual: 남은 질문의 BM25(문서 안) + 남은 질문 벡터 코사인을 RRF로 합침
  - residual+full: 위 둘에 원래 질문의 BM25(문서 안)·벡터 코사인을 더해 RRF로 합침
질문·남은 질문 임베딩은 캐시 없이 API로 새로 받습니다(CPU 작업만 있어 GPU는 쓰지 않습니다).
"""

import argparse
import os
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing import read_json, write_json  # noqa: E402
from retrieval_eval_yjk import CachedEmbeddingClient  # noqa: E402
from team_eval_yjk import TOP_K, file_hash, relative, report_lines, score_case, summarize_groups  # noqa: E402

BASES = {
    "prefix-hybrid": {"rerank": "none", "candidates": 100, "max_per_doc": "none"},
    "prefix-hybrid-cap2": {"rerank": "none", "candidates": 100, "max_per_doc": 2},
}
HYBRID = {"hybrid": True, "hybrid_vector_k": 100, "hybrid_bm25_k": 100, "bm25_prefix": True}
METHODS = ("residual", "residual+full")
PROJECT_FIELDS = ("사업명", "발주 기관")


def residual_question(question, metadata, bigrams):
    """사업명·발주 기관과 2-gram이 절반 이상 겹치는 질문 단어를 뺍니다. 다 빠지면 원래 질문을 씁니다."""
    project = set(bigrams(" ".join(str(metadata.get(field, "")) for field in PROJECT_FIELDS)))
    kept = []
    for word in unicodedata.normalize("NFC", question).split():
        grams = bigrams(word)
        if not grams or sum(gram in project for gram in grams) / len(grams) < .5:
            kept.append(word)
    return " ".join(kept) or question


def rrf(rankings, constant=60):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            fused[i] += 1. / (constant + rank)
    return [i for i, _ in sorted(fused.items(), key=lambda item: -item[1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from embedding import embed_texts, load_index
    from experiment_ids import next_experiment_id, owner_initials
    import retrieval

    load_dotenv(ROOT / ".env", override=False)
    cases = read_json(args.eval_file)
    index, chunks, config = load_index(args.index_dir)
    model = config["embedding_model"]
    client = CachedEmbeddingClient(ROOT / "results/cache/unused.json")
    client.cache, client.save = {}, lambda: None  # 캐시 없이 이번 실행 안에서만 재사용
    experiment_id = next_experiment_id(owner_initials(os.getenv("EXPERIMENT_OWNER", "")), ROOT / ".experiment-state")

    vectors = index.reconstruct_n(0, index.ntotal)
    by_doc = defaultdict(list)
    for i, chunk in enumerate(chunks):
        by_doc[chunk["doc_id"]].append(i)
    number = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}

    # 1단계 검색과 문서별 남은 질문
    first = {}
    for name, options in BASES.items():
        first[name] = [retrieval.retrieve(case["question"], client, index, chunks, config, TOP_K, case.get("filters"),
                                          **options, **HYBRID) for case in cases]
    texts = set()
    residuals = {}
    for i, case in enumerate(cases):
        for hits in (first[name][i] for name in BASES):
            for hit in hits:
                key = (i, hit["doc_id"])
                if key not in residuals:
                    residuals[key] = residual_question(case["question"], hit["metadata"], retrieval._bigrams)
                    texts.add(residuals[key])
        texts.add(case["question"])
    texts = sorted(texts)
    print(f"실험 ID: {experiment_id} / 질문 {len(cases)}건 / 임베딩할 문장 {len(texts)}건 (남은 질문 포함)")
    embedded = {}
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embedded.update(zip(batch, embed_texts(batch, client, model)))

    def reselect(case_number, question, doc_id, method):
        """문서 안 청크 번호를 다시 매긴 순서."""
        members = by_doc[doc_id]
        rest = residuals[(case_number, doc_id)]
        scores = vectors[members] @ embedded[rest]
        rankings = [retrieval._bm25_ranking(rest, chunks, set(members), len(members)),
                    [members[j] for j in np.argsort(-scores, kind="stable")]]
        if method == "residual+full":
            full = vectors[members] @ embedded[question]
            rankings += [retrieval._bm25_ranking(question, chunks, set(members), len(members)),
                         [members[j] for j in np.argsort(-full, kind="stable")]]
        return rrf(rankings)

    results = []
    for name in BASES:
        rows = [{"id": case["id"], **score_case(case, hits)} for case, hits in zip(cases, first[name])]
        results.append((name, {"rows": rows, "mean_ms": 0.0, "p95_ms": 0.0,
                               "options": {**retrieval.retrieval_options(**BASES[name], **HYBRID)}}))
        for method in METHODS:
            rows = []
            for i, (case, hits) in enumerate(zip(cases, first[name])):
                queues = {}
                reordered = []
                for hit in hits:
                    if hit["doc_id"] not in queues:
                        queues[hit["doc_id"]] = iter(reselect(i, case["question"], hit["doc_id"], method))
                    reordered.append({**chunks[next(queues[hit["doc_id"]])], "score": hit["score"]})
                rows.append({"id": case["id"], **score_case(case, reordered)})
            results.append((f"{name} > {method}", {"rows": rows, "mean_ms": 0.0, "p95_ms": 0.0,
                                                    "options": {"within_doc": method}}))
        print(f"{name}: 완료", flush=True)
    for _, result in results:
        result["summary"] = summarize_groups(cases, result["rows"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"experiment_id": experiment_id, "measured_at_utc": stamp, "eval_file": relative(args.eval_file),
            "evaluation_sha256": file_hash(args.eval_file), "cases": len(cases), "index_dir": relative(args.index_dir),
            "chunks_sha256": file_hash(Path(args.index_dir) / "chunks.json"), "chunk_count": len(chunks),
            "chunking_strategy": config.get("chunking_strategy"), "embedding_model": model, "top_k": TOP_K,
            "embedding_api_calls": client.api_calls, "fresh_query_embeddings": True}
    write_json(Path(args.results_dir) / f"within_doc_{experiment_id}_{stamp}.json", {"meta": meta, "results": dict(results)})
    lines = report_lines(meta, cases, results)
    lines[0] = "# eval_v2: 같은 문서 안에서 청크 다시 고르기 (2단계 검색)"
    # 처리 시간 표는 이 실험에서 재지 않으므로 뺍니다.
    lines = lines[:next(n for n, line in enumerate(lines) if line.startswith("## 처리 시간"))]
    presentation = [i for i, case in enumerate(cases) if "발표" in case["type"]]
    lines += ["## 본문-발표 유형만", "", "| 설정 | 문항 | 정답청크 5위 내 | 정답청크 MRR | 정답청크 없음 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in results:
        ranks = [result["rows"][i]["first_answer_chunk_rank"] for i in presentation]
        lines.append(f"| {name} | {len(ranks)} | {statistics.fmean(r is not None and r <= 5 for r in ranks):.3f} | "
                     f"{statistics.fmean(1 / r if r else 0. for r in ranks):.3f} | {sum(r is None for r in ranks)} |")
    report_path = Path(args.reports_dir) / f"within_doc_yjk_{experiment_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"보고서: {relative(report_path)}")


if __name__ == "__main__":
    main()
