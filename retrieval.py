"""김연주: 질문 → 코사인 유사도 검색 → (선택) 리랭킹. 기본값은 리랭킹 없는 baseline입니다."""

from collections import Counter
import math
import os
import re
import unicodedata

import numpy as np

from embedding import embed_texts
from observability import observed, record_retrieval

RERANK_MODES = ("none", "lexical", "cross-encoder")
# 청크 본문에는 사업명이 거의 없으므로 리랭킹 입력 앞에 붙여 어느 사업의 청크인지 구분합니다.
RERANK_FIELDS = ("사업명", "발주 기관")
_cross_encoders = {}


def _bigrams(text):
    words = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    return [word[i:i + 2] for word in words for i in range(max(len(word) - 1, 1))]


def _standardize(values):
    mean = sum(values) / len(values)
    deviation = (sum((value - mean) ** 2 for value in values) / len(values)) ** .5
    return [(value - mean) / deviation if deviation else 0. for value in values]


def _lexical_scores(question, passages):
    """후보 집합 안에서 계산한 문자 2-gram BM25. 형태소 분석기 없이 한국어 부분 일치를 반영합니다."""
    counts = [Counter(_bigrams(passage)) for passage in passages]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) or 1
    frequency = Counter(gram for count in counts for gram in count)
    weights = {gram: math.log(1 + (len(passages) - frequency[gram] + .5) / (frequency[gram] + .5))
               for gram in set(_bigrams(question)) if gram in frequency}
    return [sum(weight * count[gram] * 2.2 / (count[gram] + 1.2 * (.25 + .75 * length / average))
                for gram, weight in weights.items()) for count, length in zip(counts, lengths)]


def _cross_encoder_scores(question, passages, model_name):
    """로컬 cross-encoder 점수. torch·transformers는 이 방식을 선택했을 때만 필요하며 본문을 외부로 보내지 않습니다."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("cross-encoder 리랭킹에는 개인 가상환경에 torch, transformers 설치가 필요합니다.") from exc
    if model_name not in _cross_encoders:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=torch.float16 if device == "cuda" else torch.float32)
        _cross_encoders[model_name] = AutoTokenizer.from_pretrained(model_name), model.to(device).eval(), device
    tokenizer, model, device = _cross_encoders[model_name]
    scores = []
    with torch.no_grad():
        for start in range(0, len(passages), 16):
            batch = passages[start:start + 16]
            encoded = tokenizer([question] * len(batch), batch, padding=True, truncation="only_second",
                                max_length=1024, return_tensors="pt").to(device)
            scores += model(**encoded).logits.view(-1).float().tolist()
    return scores


def _rerank(question, hits, mode):
    passages = [" ".join(str(hit["metadata"][key]) for key in RERANK_FIELDS if hit["metadata"].get(key))
                + "\n" + hit["text"] for hit in hits]
    if mode == "lexical":
        # 척도가 다른 두 점수를 후보 집합 내 표준점수로 맞춰 더합니다.
        values = [dense + lexical for dense, lexical in zip(
            _standardize([hit["score"] for hit in hits]), _standardize(_lexical_scores(question, passages)))]
    else:
        values = _cross_encoder_scores(question, passages,
                                       os.getenv("RETRIEVAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    if len(values) != len(hits) or not all(math.isfinite(value) for value in values):
        raise ValueError("리랭킹 점수가 후보와 맞지 않습니다.")
    # 안정 정렬: 리랭킹 점수가 같으면 코사인 유사도 순서를 유지합니다.
    return sorted(({**hit, "rerank_score": float(value)} for hit, value in zip(hits, values)),
                  key=lambda hit: -hit["rerank_score"])


REWRITE_MODES = ("off", "only", "both")
_OFF, _ON = {"", "0", "false", "off", "no", "none"}, {"1", "true", "on", "yes"}
_bm25_indexes = {}


def _flag(value, env):
    raw = os.getenv(env, "") if value is None else value
    if isinstance(raw, bool):
        return raw
    raw = str(raw).strip().lower()
    if raw in _OFF:
        return False
    if raw in _ON:
        return True
    raise ValueError(f"{env}는 on/off(true/false, 1/0) 중 하나여야 합니다.")


def _count(value, env, default, name):
    raw = os.getenv(env) if value is None else value
    number = default if raw is None or str(raw).strip() == "" else int(raw)
    if number < 1:
        raise ValueError(f"{name}는 1 이상이어야 합니다.")
    return number


def retrieval_options(*, rerank=None, candidates=None, max_per_doc=None, hybrid=None, hybrid_vector_k=None,
                      hybrid_bm25_k=None, rrf_k=None, rewrite=None, rewrite_model=None):
    """retrieve가 실제로 적용하는 옵션. 우선순위: 인자 > 환경 변수 > 기본값(새 기능은 모두 꺼짐).

    결과 파일에 이 값을 그대로 기록하면 설명과 실제 동작이 어긋나지 않습니다.
    candidates·벡터/BM25 후보 수·RRF 상수는 그 값을 쓰는 기능이 켜졌을 때만 채웁니다.
    """
    mode = (os.getenv("RETRIEVAL_RERANK", "") if rerank is None else rerank).strip().lower() or "none"
    if mode not in RERANK_MODES:
        raise ValueError(f"지원하는 리랭킹 방식: {', '.join(RERANK_MODES)}")
    if max_per_doc is not None and max_per_doc < 1:
        raise ValueError("문서당 최대 청크 수는 1 이상이어야 합니다.")
    use_hybrid = _flag(hybrid, "RETRIEVAL_HYBRID")
    rewrite_mode = (os.getenv("RETRIEVAL_REWRITE", "") if rewrite is None else rewrite).strip().lower() or "off"
    if rewrite_mode not in REWRITE_MODES:
        raise ValueError(f"지원하는 질문 재작성 방식: {', '.join(REWRITE_MODES)}")
    fused = use_hybrid or rewrite_mode == "both"
    if mode != "none" or fused:
        candidates = _count(candidates, "RETRIEVAL_CANDIDATES", 50, "리랭킹 후보 수")
    else:
        candidates = None
    return {
        "rerank": mode, "candidates": candidates, "max_per_doc": max_per_doc,
        "hybrid": use_hybrid,
        "hybrid_vector_k": _count(hybrid_vector_k, "RETRIEVAL_HYBRID_VECTOR_K", candidates, "벡터 후보 수") if fused else None,
        "hybrid_bm25_k": _count(hybrid_bm25_k, "RETRIEVAL_HYBRID_BM25_K", candidates, "BM25 후보 수") if use_hybrid else None,
        "rrf_k": _count(rrf_k, "RETRIEVAL_RRF_K", 60, "RRF 상수") if fused else None,
        "rewrite": rewrite_mode,
        "rewrite_model": (rewrite_model or os.getenv("RETRIEVAL_REWRITE_MODEL") or "gpt-5-mini")
        if rewrite_mode != "off" else None,
    }


def _bm25_index(chunks):
    """청크 본문 전체의 BM25 역색인 (토큰: _bigrams, 공식: _lexical_scores와 같은 k1=1.2·b=0.75).

    처음 한 번만 만들고 같은 chunks 목록에는 다시 씁니다. IDF는 후보가 아니라 전체 청크 기준입니다.
    """
    cached = _bm25_indexes.get(id(chunks))
    if cached is not None and cached["chunks"] is chunks:
        return cached
    vocabulary, terms, docs, counts = {}, [], [], []
    lengths = np.zeros(len(chunks), dtype="float32")
    for number, chunk in enumerate(chunks):
        grams = Counter(_bigrams(chunk["text"]))
        lengths[number] = sum(grams.values())
        for gram, count in grams.items():
            terms.append(vocabulary.setdefault(gram, len(vocabulary)))
            docs.append(number)
            counts.append(count)
    terms = np.asarray(terms, dtype="int64")
    order = np.argsort(terms, kind="stable")
    terms, docs = terms[order], np.asarray(docs, dtype="int64")[order]
    counts = np.asarray(counts, dtype="float32")[order]
    frequency = np.bincount(terms, minlength=len(vocabulary)).astype("float32")
    weights = np.log(1 + (len(chunks) - frequency + .5) / (frequency + .5)).astype("float32")
    average = float(lengths.mean()) or 1.
    norm = .25 + .75 * lengths[docs] / average
    built = {"chunks": chunks, "vocabulary": vocabulary, "docs": docs,
             "pointer": np.concatenate([[0], np.cumsum(frequency)]).astype("int64"),
             "weights": weights[terms] * counts * 2.2 / (counts + 1.2 * norm)}
    _bm25_indexes.clear()  # 인덱스는 하나만 메모리에 둡니다.
    _bm25_indexes[id(chunks)] = built
    return built


def _bm25_ranking(question, chunks, eligible, limit):
    """필터를 통과한 청크 중 BM25 점수가 0보다 큰 상위 limit개의 청크 번호."""
    built = _bm25_index(chunks)
    scores = np.zeros(len(chunks), dtype="float32")
    for gram in set(_bigrams(question)):
        term = built["vocabulary"].get(gram)
        if term is not None:
            start, end = built["pointer"][term], built["pointer"][term + 1]
            scores[built["docs"][start:end]] += built["weights"][start:end]
    mask = np.zeros(len(chunks), dtype=bool)
    mask[list(eligible)] = True
    found = np.flatnonzero(mask & (scores > 0))
    return [int(i) for i in found[np.argsort(-scores[found], kind="stable")][:limit]]


def _rrf(rankings, constant):
    """Reciprocal Rank Fusion: 점수 = Σ 1/(constant + 순위). 동점은 먼저 나온 목록의 순서를 따릅니다."""
    fused = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.) + 1. / (constant + rank)
    first_seen = {i: n for n, i in enumerate(dict.fromkeys(i for ranking in rankings for i in ranking))}
    return sorted(fused.items(), key=lambda item: (-item[1], first_seen[item[0]]))


@observed("retrieve")
def retrieve(question, client, index, chunks, config, top_k=5, filters=None, *, rerank=None, candidates=None,
             max_per_doc=None, hybrid=None, hybrid_vector_k=None, hybrid_bm25_k=None, rrf_k=None,
             rewrite=None, rewrite_model=None, rewrite_cache=None):
    """반환: Chunk에 score(float)를 추가한 목록. filters는 metadata 정확 일치.

    rerank: none(기본)·lexical·cross-encoder. 생략하면 환경 변수 RETRIEVAL_RERANK를 사용합니다.
    리랭킹은 필터를 통과한 코사인 상위 candidates개(기본 RETRIEVAL_CANDIDATES 또는 50)만 재정렬하여
    top_k개를 반환하고 rerank_score(float)를 추가합니다. score는 항상 코사인 유사도입니다.
    max_per_doc: 한 문서(doc_id)에서 반환할 최대 청크 수. 생략하면 제한하지 않습니다.
    리랭킹과 함께 쓰면 재정렬된 후보에 적용하므로 후보가 소수 문서에 몰리면 top_k보다 적게 반환할 수 있습니다.

    아래 옵션은 모두 기본 꺼짐이며, 꺼져 있으면 위의 기존 동작과 결과가 같습니다 (retrieval_options 참고).
    hybrid: 켜면 벡터 상위 hybrid_vector_k개와 BM25 상위 hybrid_bm25_k개(기본 둘 다 candidates)를
      RRF(상수 rrf_k, 기본 60)로 합쳐 후보를 만들고, 그 후보에 rerank를 적용합니다.
      합친 후보에는 rrf_score(float)가 추가되고 score는 여전히 질문과의 코사인 유사도입니다.
      환경 변수: RETRIEVAL_HYBRID, RETRIEVAL_HYBRID_VECTOR_K, RETRIEVAL_HYBRID_BM25_K, RETRIEVAL_RRF_K.
    rewrite: off(기본)·only·both. only는 재작성 질문만으로, both는 원문과 재작성 질문을 각각 검색해
      RRF로 합칩니다. 재작성은 query_rewrite.rewrite_question이 하며 실패하거나 비면 원문으로 돌아갑니다.
      rewrite_cache(파일 경로)를 주면 같은 질문은 다시 호출하지 않습니다. 환경 변수: RETRIEVAL_REWRITE,
      RETRIEVAL_REWRITE_MODEL(기본 gpt-5-mini). 리랭킹에는 검색에 쓴 질문들을 이어 붙인 문장을 씁니다.
    """
    if not question.strip() or top_k < 1:
        raise ValueError("비어 있지 않은 질문과 1 이상의 top_k가 필요합니다.")
    options = retrieval_options(rerank=rerank, candidates=candidates, max_per_doc=max_per_doc, hybrid=hybrid,
                                hybrid_vector_k=hybrid_vector_k, hybrid_bm25_k=hybrid_bm25_k, rrf_k=rrf_k,
                                rewrite=rewrite, rewrite_model=rewrite_model)
    mode, candidates = options["rerank"], options["candidates"]
    pool = top_k if mode == "none" else max(top_k, candidates)
    normalize = lambda value: unicodedata.normalize("NFC", str(value))
    eligible = {i for i, chunk in enumerate(chunks) if all(
        key in chunk["metadata"] and normalize(chunk["metadata"][key]) == normalize(value)
        for key, value in (filters or {}).items()
    )}
    if not eligible:
        record_retrieval([])
        return []
    if options["hybrid"] or options["rewrite"] != "off":
        hits = _fused_retrieve(question, client, index, chunks, config, top_k, pool, eligible, options, rewrite_cache)
        record_retrieval(hits)
        return hits
    vector = embed_texts([question], client, config["embedding_model"])
    if vector.shape[1] != index.d:
        raise ValueError("질문 임베딩 차원이 인덱스와 다릅니다.")
    # 소규모 baseline: 전체 순위를 얻고 필터를 적용하여 후보를 채웁니다.
    # 문서당 상한이 있으면 건너뛰는 청크만큼 더 아래 순위까지 봐야 합니다.
    scores, ids = index.search(vector, index.ntotal if filters or max_per_doc else min(pool, index.ntotal))
    per_doc = Counter()
    # 리랭킹을 할 때는 후보를 줄이지 않고 재정렬한 뒤에 상한을 적용합니다.
    within_limit = lambda doc_id: max_per_doc is None or per_doc[doc_id] < max_per_doc
    hits = []
    for score, i in zip(scores[0], ids[0]):
        if int(i) in eligible and (mode != "none" or within_limit(chunks[int(i)]["doc_id"])):
            per_doc[chunks[int(i)]["doc_id"]] += 1
            hits.append({**chunks[int(i)], "score": float(score)})
            if len(hits) == pool:
                break
    if mode != "none":
        # 후보는 이미 필터를 통과했으므로 재정렬해도 필터 밖 문서가 포함되지 않습니다.
        ranked, hits = observed(f"rerank-{mode}")(_rerank)(question, hits, mode), []
        per_doc.clear()
        for hit in ranked:
            if within_limit(hit["doc_id"]):
                per_doc[hit["doc_id"]] += 1
                hits.append(hit)
        hits = hits[:top_k]
    record_retrieval(hits)
    return hits


def _fused_retrieve(question, client, index, chunks, config, top_k, pool, eligible, options, rewrite_cache):
    """하이브리드·질문 재작성 경로. 질문마다 벡터(와 BM25) 순위를 만들고 RRF로 합친 뒤 기존과 같이 리랭킹합니다."""
    queries = [question]
    if options["rewrite"] != "off":
        from query_rewrite import rewrite_question  # 재작성을 켰을 때만 불러옵니다.

        rewritten, _ = rewrite_question(question, client, options["rewrite_model"], rewrite_cache)
        queries = [rewritten] if options["rewrite"] == "only" else list(dict.fromkeys([question, rewritten]))
    vector_k = options["hybrid_vector_k"] or pool
    rankings, cosine = [], None
    for number, query in enumerate(queries):
        vector = embed_texts([query], client, config["embedding_model"])
        if vector.shape[1] != index.d:
            raise ValueError("질문 임베딩 차원이 인덱스와 다릅니다.")
        scores, ids = index.search(vector, index.ntotal)
        if number == 0:  # score는 첫 질문(only면 재작성 질문, 그 외에는 원문)과의 코사인 유사도
            cosine = {int(i): float(s) for s, i in zip(scores[0], ids[0])}
        rankings.append([int(i) for i in ids[0] if int(i) in eligible][:vector_k])
        if options["hybrid"]:
            rankings.append(_bm25_ranking(query, chunks, eligible, options["hybrid_bm25_k"]))
    fused = _rrf(rankings, options["rrf_k"] or 60)
    max_per_doc, mode = options["max_per_doc"], options["rerank"]
    per_doc = Counter()
    within_limit = lambda doc_id: max_per_doc is None or per_doc[doc_id] < max_per_doc
    hits = []
    for i, rrf_score in fused:
        if mode != "none" or within_limit(chunks[i]["doc_id"]):
            per_doc[chunks[i]["doc_id"]] += 1
            hits.append({**chunks[i], "score": cosine[i], "rrf_score": rrf_score})
            if len(hits) == pool:
                break
    if mode != "none":
        ranked, hits = observed(f"rerank-{mode}")(_rerank)(" ".join(queries), hits, mode), []
        per_doc.clear()
        for hit in ranked:
            if within_limit(hit["doc_id"]):
                per_doc[hit["doc_id"]] += 1
                hits.append(hit)
        hits = hits[:top_k]
    return hits
