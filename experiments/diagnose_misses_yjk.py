"""김연주: 검색에서 정답 문서를 놓친 질문을 모아 단계(1차 검색/리랭킹)와 원인을 분류합니다 (API 호출 없음).

  python experiments/diagnose_misses_yjk.py --eval-file data/eval_retrieval_yjk.json

기준 설정은 lexical 리랭킹 후보 100, top-k 5입니다 (--rerank, --candidates, --top-k로 변경).
질문 임베딩은 results/cache/query_embeddings.json 캐시만 쓰며, 캐시에 없으면 멈춥니다.

단계 구분: 필터를 적용한 코사인 순위에서 정답 문서의 첫 청크 순위가
  후보 수 이내 → 리랭킹 문제 (후보에는 있었으나 리랭킹 뒤 top-k 밖으로 밀림)
  후보 수 밖  → 1차 검색 누락 (후보를 늘려야 들어옴)

원인 분류 (자동 규칙, 위에서부터 먼저 걸리는 것):
  유사한 다른 사업에 밀려서: top-k가 다른 정답 문서의 청크로 다 찼거나(기관 필터 질문),
                            정답 밖 1위 문서가 같은 기관이거나 사업명이 비슷하고 질문 내용도 그만큼 들어 있음
  정확한 용어가 달라서: 질문의 고유어(영문이 섞인 단어, 사업명 일부 질문의 기관명·사업명 조각)가
                       정답 문서 본문에 없음 (띄어쓰기와 조사 차이는 무시)
  청크에 내용이 없어서: 질문의 개수·수치 표현("56개")이 본문에 없거나,
                       질문 내용의 글자 2-gram이 정답 문서의 어느 청크에도 절반 이상 모여 있지 않음
  질문이 일반적이라 구분 불가: 질문 내용이 정답 청크에 있지만, 다른 사업 3개 이상에도 비슷하게(정답 비율 -0.2 이내, 최소 0.75) 들어 있음
                              (예: "업무 효율성 향상, 사용자 편의성"). 4분류에 맞지 않아 따로 둡니다.
  표현이 달라서: 그 외. 질문 내용이 정답 청크에 있지만 리랭킹·임베딩 점수가 다른 문서보다 낮음
자동 분류는 대표 사례를 직접 읽어 확인해야 합니다. 원문은 보고서에 넣지 않습니다.
"""

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing import read_json, write_json  # noqa: E402
from retrieval_eval_yjk import ASPECTS, CachedEmbeddingClient  # noqa: E402

# 평가셋 그룹 → 사업명 포함 여부
NAME_GROUPS = {"A_full_name": "사업명 있음", "E_filter_named": "사업명 있음", "B_short_name": "사업명 일부",
               "C_no_name": "사업명 없음", "D_filter_agency": "사업명 없음"}
TEMPLATE = re.compile(r"다음 내용의 사업에서|이 기관이 발주한 사업들의|관련|사업의")
JOSA = re.compile(r"(에서|으로|에게|까지|부터|이며|이다|의|은|는|이|가|을|를|에|로|과|와|도|만)$")
CAUSES = ("정확한 용어가 달라서", "표현이 달라서", "유사한 다른 사업에 밀려서", "청크에 내용이 없어서", "질문이 일반적이라 구분 불가")
# 검색 기본값이 바뀌어도 이 진단의 조건(하이브리드·상한 꺼짐)을 유지
BASE = {"hybrid": False, "max_per_doc": "none"}


def question_content(question):
    """평가셋 질문 틀(측면 질문, 연결어)을 지운 내용 부분."""
    for aspect in ASPECTS:
        question = question.replace(aspect, " ")
    return " ".join(TEMPLATE.sub(" ", question).split())


def squash(text):
    return re.sub(r"\s+", "", text)


COUNT = re.compile(r"^\d+(개|건|명|종|식|회|년|월|일)?$")


def question_terms(content, case_group):
    """(고유어, 개수·수치 표현). 고유어는 영문이 섞인 단어, B 그룹은 기관명·사업명 조각(조사 제거) 전체."""
    words = [JOSA.sub("", word) for word in re.findall(r"[\w()·-]+", content)]
    words = [word for word in words if len(word) >= 2]
    counts = [word for word in words if COUNT.match(word)]
    if case_group == "B_short_name":
        return [word for word in words if word not in counts], counts
    return [word for word in words if re.search(r"[A-Za-z]", word)], counts


def coverage(grams, chunk_grams):
    """질문 내용 2-gram 중 한 청크에 함께 들어 있는 최대 비율."""
    return max((len(grams & item) / len(grams) for item in chunk_grams), default=0.0) if grams else 0.0


def name_similarity(bigrams_fn, left, right):
    a, b = set(bigrams_fn(left)), set(bigrams_fn(right))
    return len(a & b) / len(a | b) if a | b else 0.0


def classify(case, gold_doc, top_doc, chunks_by_doc, grams_by_doc, bigrams_fn):
    """반환: (원인, 근거 수치). top_doc이 None이면 top-k가 다른 정답 문서로 찬 경우입니다."""
    content = question_content(case["question"])
    grams = set(bigrams_fn(content))
    text = squash(" ".join(chunk["text"] for chunk in chunks_by_doc[gold_doc["doc_id"]]))
    terms, counts = question_terms(content, case["group"])
    missing_terms = [term for term in terms if squash(term) not in text]
    missing_counts = [term for term in counts if term not in text]
    gold_cov = coverage(grams, grams_by_doc[gold_doc["doc_id"]])
    top_cov = coverage(grams, grams_by_doc[top_doc["doc_id"]]) if top_doc else 0.0
    # 정답 문서만큼 질문 내용을 담은 다른 사업 수 (질문이 얼마나 일반적인지)
    rivals = sum(doc_id != gold_doc["doc_id"] and coverage(grams, items) >= max(0.75, gold_cov - 0.2)
                 for doc_id, items in grams_by_doc.items()) if grams else 0
    similar = top_doc is not None and (
        name_similarity(bigrams_fn, gold_doc["사업명"], top_doc["사업명"]) >= 0.25
        or gold_doc["발주 기관"] == top_doc["발주 기관"])
    evidence = {"missing_terms": len(missing_terms), "missing_counts": len(missing_counts),
                "gold_chunk_coverage": round(gold_cov, 3), "top_chunk_coverage": round(top_cov, 3),
                "rival_docs": rivals, "similar_top": similar, "filled_by_other_gold": top_doc is None}
    if top_doc is None or (similar and top_cov >= gold_cov - 0.1):
        return "유사한 다른 사업에 밀려서", evidence
    if missing_terms:
        return "정확한 용어가 달라서", evidence
    if missing_counts or gold_cov < 0.5:
        return "청크에 내용이 없어서", evidence
    if rivals >= 3:
        return "질문이 일반적이라 구분 불가", evidence
    return "표현이 달라서", evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes/parsing-v2-yjk")
    parser.add_argument("--cache-file", type=Path, default=ROOT / "results/cache/query_embeddings.json")
    parser.add_argument("--rerank", default="lexical")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    args = parser.parse_args()

    from embedding import embed_texts, load_index
    from retrieval import _bigrams, retrieve

    cases = read_json(args.eval_file)
    index, chunks, config = load_index(args.index_dir)
    client = CachedEmbeddingClient(args.cache_file)
    chunks_by_doc = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["doc_id"]].append(chunk)
    meta_by_doc = {doc_id: {"doc_id": doc_id, **items[0]["metadata"]} for doc_id, items in chunks_by_doc.items()}
    grams_by_doc = {doc_id: [set(_bigrams(chunk["text"])) for chunk in items] for doc_id, items in chunks_by_doc.items()}

    misses = []
    for number, case in enumerate(cases):
        gold = set(case["expected_doc_ids"])
        filters = case.get("filters") or {}
        hits = retrieve(case["question"], client, index, chunks, config, args.top_k, filters,
                        rerank=args.rerank, candidates=args.candidates, **BASE)
        found = gold & {hit["doc_id"] for hit in hits}
        if found == gold:
            continue
        vector = embed_texts([case["question"]], client, config["embedding_model"])
        _, ids = index.search(vector, index.ntotal)
        order = [int(i) for i in ids[0] if all(chunks[int(i)]["metadata"].get(k) == v for k, v in filters.items())]
        first_rank = {}
        for rank, i in enumerate(order, 1):
            first_rank.setdefault(chunks[i]["doc_id"], rank)
        top_doc = next((meta_by_doc[hit["doc_id"]] for hit in hits if hit["doc_id"] not in gold), None)
        # 후보를 늘렸을 때 들어오는지: 리랭킹 후보 50·100·200, top-k 5·10으로 다시 검색
        recovered = {}
        for candidates in (50, 100, 200):
            for top_k in (args.top_k, 10):
                more = retrieve(case["question"], client, index, chunks, config, top_k, filters,
                                rerank=args.rerank, candidates=candidates, **BASE)
                recovered[f"c{candidates}@{top_k}"] = len(gold & {hit["doc_id"] for hit in more}) / len(gold)
        for doc_id in sorted(gold - found):
            rank = first_rank.get(doc_id)
            cause, evidence = classify(case, meta_by_doc[doc_id], top_doc, chunks_by_doc, grams_by_doc, _bigrams)
            misses.append({
                "case": number, "group": case["group"], "name_group": NAME_GROUPS.get(case["group"], case["group"]),
                "gold_doc_id": doc_id, "cosine_rank": rank,
                "stage": "리랭킹 문제" if rank is not None and rank <= args.candidates else "1차 검색 누락",
                "within_50": rank is not None and rank <= 50, "within_100": rank is not None and rank <= 100,
                "cause": cause, **evidence, "recovered": recovered,
                "top_doc_id": top_doc["doc_id"] if top_doc else None,
            })
    if client.api_calls:
        sys.exit("임베딩 API가 호출되었습니다. 캐시를 확인하세요.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = {"rerank": args.rerank, "candidates": args.candidates, "top_k": args.top_k,
                "index_dir": str(args.index_dir.resolve().relative_to(ROOT)) if args.index_dir.resolve().is_relative_to(ROOT)
                else args.index_dir.name,
                "chunk_count": config["chunk_count"],
                "chunks_sha256": hashlib.sha256((args.index_dir / "chunks.json").read_bytes()).hexdigest(),
                "evaluation_sha256": hashlib.sha256(args.eval_file.read_bytes()).hexdigest(),
                "cases": len(cases), "missed_cases": len({m["case"] for m in misses}), "missed_gold_docs": len(misses)}
    output = args.results_dir / f"diagnose_misses_{stamp}.json"
    write_json(output, {"settings": settings, "misses": misses})

    groups = ("사업명 있음", "사업명 일부", "사업명 없음")
    total_by_group = Counter(NAME_GROUPS.get(case["group"], case["group"]) for case in cases)
    lines = ["# 실패 질문 진단 (검색 단계)", "",
             "평가셋: 개인이 CSV 메타데이터로 자동 구성한 문서 단위 정답 325문항(팀 공통 평가셋 아님). 답변 사실성은 측정하지 않습니다.",
             "원인은 자동 규칙으로 1차 분류한 것이며 대표 사례는 직접 확인했습니다. 질문·원문은 넣지 않았습니다.", "",
             "| 설정 | 값 |", "| --- | --- |", *(f"| {k} | `{v}` |" for k, v in settings.items()), "",
             "## 단계별 (놓친 정답 문서 기준)", "",
             "| 사업명 | 전체 질문 | 놓친 정답 문서 | 코사인 50위 이내 | 51~100위 | 100위 밖 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in groups:
        rows = [m for m in misses if m["name_group"] == group]
        lines.append(f"| {group} | {total_by_group[group]} | {len(rows)} | {sum(m['within_50'] for m in rows)} | "
                     f"{sum(m['within_100'] and not m['within_50'] for m in rows)} | {sum(not m['within_100'] for m in rows)} |")
    lines += ["", "## 원인별", "", "| 원인 | " + " | ".join(groups) + " | 합계 |", "| --- |" + " ---: |" * (len(groups) + 1)]
    for cause in CAUSES:
        counts = [sum(m["cause"] == cause and m["name_group"] == group for m in misses) for group in groups]
        lines.append(f"| {cause} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    lines += ["", "## 후보·top-k를 늘렸을 때 찾은 정답 문서 비율 (놓친 질문 기준 평균)", "",
              "| 설정 | " + " | ".join(groups) + " | 전체 |", "| --- |" + " ---: |" * (len(groups) + 1)]
    for key in misses[0]["recovered"] if misses else []:
        cells = []
        for group in (*groups, None):
            rows = [m for m in misses if group is None or m["name_group"] == group]
            cells.append(f"{sum(m['recovered'][key] for m in rows) / len(rows):.2f}" if rows else "-")
        lines.append(f"| {args.rerank} {key.replace('c', '후보 ').replace('@', ', top-')} | " + " | ".join(cells) + " |")
    lines += ["", f"상세: `{output.name}` (Git 제외)", ""]
    report = args.reports_dir / f"diagnose_misses_yjk_{args.eval_file.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"결과 저장: {output}\n보고서: {report}")


if __name__ == "__main__":
    main()
