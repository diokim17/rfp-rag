"""김연주: 질문 → 코사인 유사도 검색 → (선택) 리랭킹. 기본값은 리랭킹 없는 baseline입니다."""

from collections import Counter
import math
import os
import re
import unicodedata

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


@observed("retrieve")
def retrieve(question, client, index, chunks, config, top_k=5, filters=None, *, rerank=None, candidates=None,
             max_per_doc=None):
    """반환: Chunk에 score(float)를 추가한 목록. filters는 metadata 정확 일치.

    rerank: none(기본)·lexical·cross-encoder. 생략하면 환경 변수 RETRIEVAL_RERANK를 사용합니다.
    리랭킹은 필터를 통과한 코사인 상위 candidates개(기본 RETRIEVAL_CANDIDATES 또는 50)만 재정렬하여
    top_k개를 반환하고 rerank_score(float)를 추가합니다. score는 항상 코사인 유사도입니다.
    max_per_doc: 한 문서(doc_id)에서 반환할 최대 청크 수. 생략하면 제한하지 않습니다.
    리랭킹과 함께 쓰면 재정렬된 후보에 적용하므로 후보가 소수 문서에 몰리면 top_k보다 적게 반환할 수 있습니다.
    """
    if not question.strip() or top_k < 1:
        raise ValueError("비어 있지 않은 질문과 1 이상의 top_k가 필요합니다.")
    if max_per_doc is not None and max_per_doc < 1:
        raise ValueError("문서당 최대 청크 수는 1 이상이어야 합니다.")
    mode = (os.getenv("RETRIEVAL_RERANK", "") if rerank is None else rerank).strip().lower() or "none"
    if mode not in RERANK_MODES:
        raise ValueError(f"지원하는 리랭킹 방식: {', '.join(RERANK_MODES)}")
    pool = top_k
    if mode != "none":
        candidates = int(os.getenv("RETRIEVAL_CANDIDATES") or 50 if candidates is None else candidates)
        if candidates < 1:
            raise ValueError("리랭킹 후보 수는 1 이상이어야 합니다.")
        pool = max(top_k, candidates)
    normalize = lambda value: unicodedata.normalize("NFC", str(value))
    eligible = {i for i, chunk in enumerate(chunks) if all(
        key in chunk["metadata"] and normalize(chunk["metadata"][key]) == normalize(value)
        for key, value in (filters or {}).items()
    )}
    if not eligible:
        record_retrieval([])
        return []
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
