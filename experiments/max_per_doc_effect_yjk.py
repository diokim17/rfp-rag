"""김연주: 문서당 청크 상한(max_per_doc)의 효과를 검색 단계에서만 비교하는 개인 실험 도구 (생성 API 호출 없음).

  python experiments/max_per_doc_effect_yjk.py --eval-file data/eval_retrieval_yjk.json --index-dir indexes

상한(없음·2·3)과 리랭킹(none·lexical 후보 50·lexical 후보 100)의 모든 조합을 같은 질문으로 비교합니다.
문서 단위 Recall·MRR로는 보이지 않는 차이를 보려고 top-k 구성 지표(고유 문서 수, 정답 외 청크 비율 등)를 함께 계산합니다.
질문 임베딩은 results/cache에 저장하며, 캐시에 없는 질문이 있으면 --allow-api를 주기 전에는 API를 호출하지 않고 멈춥니다.
"""

import argparse
import hashlib
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing import read_json, write_json  # noqa: E402
from retrieval_eval_yjk import CachedEmbeddingClient  # noqa: E402

CAPS = (None, 2, 3)
RERANKS = (("none", None), ("lexical", 50), ("lexical", 100))
# 이전 실험(retrieval_eval_yjk.py)이 만든 캐시. 있으면 처음 한 번 가져와 API 재호출을 피합니다.
SEED_CACHE = ROOT / "results/retrieval-rerank-yjk/query_embedding_cache.json"


def question_id(model, question):
    """평가셋에 질문 ID가 없으므로 (모델, 질문)의 해시를 ID로 씁니다. retrieval_eval_yjk.py와 같은 규칙입니다."""
    return hashlib.sha256(f"{model}\n{question}".encode()).hexdigest()


def measure(case, hits, top_k):
    """질문 하나의 top-k 지표. gold는 정답 문서 ID 집합입니다."""
    gold = set(case["expected_doc_ids"])
    docs = [hit["doc_id"] for hit in hits[:top_k]]
    first = next((rank for rank, doc in enumerate(docs, 1) if doc in gold), None)
    gold_chunks = sum(doc in gold for doc in docs)
    return {
        "recall_any": float(first is not None),            # exp_max_per_doc.py 방식: 정답 문서가 하나라도 있으면 1
        "recall_frac": len(gold & set(docs)) / len(gold),  # retrieval_eval_yjk.py 방식: 찾은 정답 문서 비율
        "mrr": 1 / first if first else 0.0,
        "hit_count": len(docs),
        "unique_docs": len(set(docs)),
        "non_gold_share": (len(docs) - gold_chunks) / len(docs) if docs else 0.0,
        "gold_chunks": gold_chunks,
        "gold_chunks_ge2": float(gold_chunks >= 2),
    }


def label(cap, mode, candidates):
    return f"{mode if mode == 'none' else f'{mode}-c{candidates}'} / {'상한 없음' if cap is None else f'상한 {cap}'}"


def run(args):
    from embedding import embed_texts, load_index
    from retrieval import retrieve

    cases = read_json(args.eval_file)
    index, chunks, config = load_index(args.index_dir)
    model = config["embedding_model"]
    cache_path = args.cache_dir / "query_embeddings.json"
    if not cache_path.is_file() and SEED_CACHE.is_file():
        write_json(cache_path, read_json(SEED_CACHE))
    client = CachedEmbeddingClient(cache_path)
    questions = sorted({case["question"] for case in cases})
    missing = [q for q in questions if question_id(model, q) not in client.cache]
    print(f"질문 {len(cases)}건 (고유 {len(questions)}개) / 캐시에 없는 질문 {len(missing)}개 "
          f"→ 필요한 임베딩 API 호출 {-(-len(missing) // 64)}회")
    if missing and not args.allow_api:
        sys.exit("캐시에 없는 질문이 있습니다. API 호출을 허용하려면 --allow-api를 추가하세요.")
    for start in range(0, len(missing), 64):
        embed_texts(missing[start:start + 64], client, model)
    if missing:
        client.save()

    summary, records = {}, []
    for mode, candidates in RERANKS:
        for cap in CAPS:
            name, groups = label(cap, mode, candidates), defaultdict(list)
            for number, case in enumerate(cases):
                hits = retrieve(case["question"], client, index, chunks, config, args.top_k, case.get("filters"),
                                rerank=mode, candidates=candidates, hybrid=False,  # 검색 기본값이 바뀌어도 이 실험의 조건(하이브리드·상한 꺼짐)을 유지
                                max_per_doc="none" if cap is None else cap)
                row = measure(case, hits, args.top_k)
                groups["ALL"].append(row)
                groups[case["group"]].append(row)
                records.append({"setting": name, "case": number, "group": case["group"], **row,
                                "hits": [[hit["doc_id"], hit["chunk_id"]] for hit in hits]})
            summary[name] = {group: {"count": len(rows),
                                     **{key: statistics.fmean(row[key] for row in rows) for key in rows[0]},
                                     "gold_chunks_ge2_count": sum(row["gold_chunks_ge2"] for row in rows),
                                     "short_count": sum(row["hit_count"] < args.top_k for row in rows)}
                             for group, rows in groups.items()}
    if client.api_calls:
        sys.exit("캐시를 채운 뒤에도 임베딩 API가 호출되었습니다. 캐시 키 규칙을 확인하세요.")

    k = args.top_k
    print(f"\n| 설정 | Recall@{k}(하나라도) | Recall@{k}(비율) | MRR | 고유 문서 수 | 정답 외 청크 비율 | "
          f"정답 청크 2개↑ 질문 | {k}개 미만 반환 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, groups in summary.items():
        row = groups["ALL"]
        print(f"| {name} | {row['recall_any']:.3f} | {row['recall_frac']:.3f} | {row['mrr']:.3f} | "
              f"{row['unique_docs']:.2f} | {row['non_gold_share']:.3f} | {row['gold_chunks_ge2_count']:.0f} | "
              f"{row['short_count']} |")
    filtered = sorted(group for group in next(iter(summary.values())) if "filter" in group)
    if filtered:
        print("\n| 설정 | " + " | ".join(f"{group} Recall(비율) / 고유 문서" for group in filtered) + " |\n| --- |"
              + " ---: |" * len(filtered))
        for name, groups in summary.items():
            print(f"| {name} | " + " | ".join(
                f"{groups[group]['recall_frac']:.3f} / {groups[group]['unique_docs']:.2f}" for group in filtered) + " |")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.results_dir / f"max_per_doc_effect_{stamp}.json"
    write_json(output, {"settings": {**config, "top_k": k, "index_dir": str(args.index_dir),
                                     "evaluation_sha256": hashlib.sha256(args.eval_file.read_bytes()).hexdigest(),
                                     "query_embedding_api_calls": client.api_calls},
                        "summary": summary, "records": records})
    print(f"\n결과 저장: {output}")
    return summary, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/cache")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-api", action="store_true", help="캐시에 없는 질문의 임베딩 API 호출을 허용")
    run(parser.parse_args())
