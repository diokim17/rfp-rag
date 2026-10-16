"""
기본값은 eval_team_v1(yjk-0017)·eval_v2(yjk-0021, yjk-0025)·새 질문 19개(yjk-0024) 측정에서 가장 좋았던 조합이며,
인자나 환경 변수로 끌 수 있습니다. 설정 근거는 results/reports/b_retrieval_settings_yjk.md.
"""

from collections import Counter
from itertools import zip_longest
import importlib.util
import math
import os
from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np

SCENARIO = "B"

from b_embedding import embed_texts
from b_observability import observed, record_retrieval

RERANK_MODES = ("none", "lexical", "cross-encoder")
# 청크 본문에는 사업명이 거의 없으므로 리랭킹 입력 앞에 붙여 어느 사업의 청크인지 구분합니다.
RERANK_FIELDS = ("사업명", "발주 기관")
COMPARE_DOCS = 3  # 비교 질문에서 문서 전체를 재정렬할 최대 문서 수(재정렬 비용 상한)
# 기본값: 하이브리드(벡터 100 + BM25 100) → RRF 상위 50개를 cross-encoder로 재정렬 → 문서당 2개.
DEFAULT_RERANK, DEFAULT_HYBRID, DEFAULT_CANDIDATES, DEFAULT_HYBRID_K, DEFAULT_MAX_PER_DOC = "cross-encoder", True, 50, 100, 2
# 하이브리드의 BM25 색인에도 사업명·발주 기관을 본문 앞에 붙입니다(eval_v2 yjk-0010: 50위 밖 정답 청크 10→1건).
DEFAULT_BM25_PREFIX = True
# 재정렬 후보 확장: 후보에 처음 나온 문서 EXPAND_DOCS개에서 남은 질문 BM25 상위 청크를 더합니다.
# 기본 5(yjk-0024 새 질문 정답청크 5위 내 0.737→0.947, yjk-0025 eval_v2 0.968→0.984). 0이면 끔.
DEFAULT_EXPAND, EXPAND_DOCS = 5, 2
# 범위 밖 질문 차단: 질문과 가장 가까운 청크의 코사인이 이 값보다 낮으면 빈 결과를 돌려줘 생성 API를 부르지 않습니다.
# 기본 0.40(eval_v2 정상 질문 최소 0.423, 범위 밖 질문 24개 중 17개 차단). 나머지는 생성 프롬프트의 거절 규칙이 처리합니다.
# 점수는 인덱스(청킹)·임베딩 모델마다 달라지므로 바꾸면 experiments/offtopic_threshold_yjk.py로 다시 잽니다. none·off로 끔.
# 이 값은 B 임베딩(text-embedding-3-small)으로 잰 것이라, 시나리오 A는 A 모델로 검증하기 전까지 기본으로 끕니다
# (A에서도 RETRIEVAL_MIN_SCORE나 min_score 인자로 기준값을 주면 차단합니다).
DEFAULT_MIN_SCORE = 0.40
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# 시나리오 A는 리랭커를 각자 HF 캐시로 내려받지 않고, 서버의 이 공용 폴더에 받아 둔 모델을 함께 씁니다.
# .env의 A_RERANK_MODEL에는 이 폴더 아래 모델 폴더 이름만 적습니다 (예: bge-reranker-v2-m3).
A_RERANK_MODEL_DIR = Path("/home/spai1313/models/rerank")
_cross_encoders = {}


def _cross_encoder_available():
    """torch·transformers를 불러오지 않고 설치 여부만 확인합니다."""
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def rerank_model_source():
    """cross-encoder가 불러올 모델 위치를 돌려줍니다.

    B: RETRIEVAL_RERANK_MODEL (HF 모델 ID, 기본 BAAI/bge-reranker-v2-m3)
    A: 공용 폴더 A_RERANK_MODEL_DIR/<A_RERANK_MODEL>. 로컬 폴더 경로를 넘기면 transformers가 Hub에 접속하지 않습니다.
    """
    if SCENARIO != "A":
        return os.getenv("RETRIEVAL_RERANK_MODEL", DEFAULT_RERANK_MODEL)
    name = os.getenv("A_RERANK_MODEL", "").strip()
    if not name or not (A_RERANK_MODEL_DIR / name).is_dir():
        raise ValueError(f"시나리오 A 리랭커 모델 폴더가 없습니다: {A_RERANK_MODEL_DIR}/{name} "
                         "(.env의 A_RERANK_MODEL 확인, 모델 없이 실행하려면 RETRIEVAL_RERANK=lexical)")
    return str(A_RERANK_MODEL_DIR / name)


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
        raise RuntimeError("cross-encoder 리랭킹에는 개인 가상환경에 torch, transformers 설치가 필요합니다 "
                           "(pip install -r experiments/requirements-rerank.txt).") from exc
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


def _with_project(chunk):
    """본문 앞에 사업명·발주 기관을 붙인 텍스트. 리랭킹 입력과 BM25 접두 색인에 씁니다."""
    return " ".join(str(chunk["metadata"][key]) for key in RERANK_FIELDS if chunk["metadata"].get(key)) \
        + "\n" + chunk["text"]


def _rerank(question, hits, mode):
    passages = [_with_project(hit) for hit in hits]
    if mode == "lexical":
        # 척도가 다른 두 점수를 후보 집합 내 표준점수로 맞춰 더합니다.
        values = [dense + lexical for dense, lexical in zip(
            _standardize([hit["score"] for hit in hits]), _standardize(_lexical_scores(question, passages)))]
    else:
        values = _cross_encoder_scores(question, passages, rerank_model_source())
    if len(values) != len(hits) or not all(math.isfinite(value) for value in values):
        raise ValueError("리랭킹 점수가 후보와 맞지 않습니다.")
    # 안정 정렬: 리랭킹 점수가 같으면 코사인 유사도 순서를 유지합니다.
    return sorted(({**hit, "rerank_score": float(value)} for hit, value in zip(hits, values)),
                  key=lambda hit: -hit["rerank_score"])


REWRITE_MODES = ("off", "only", "both")
WITHIN_DOC_MODES = ("off", "residual", "residual+full")
_OFF, _ON = {"", "0", "false", "off", "no", "none"}, {"1", "true", "on", "yes"}
_bm25_indexes = {}
_doc_members = {}
_name_gram_cache = {}


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


def _max_per_doc(value):
    """None이면 환경 변수 RETRIEVAL_MAX_PER_DOC, 없으면 기본값. none·off는 상한 없음(None)."""
    raw = os.getenv("RETRIEVAL_MAX_PER_DOC") if value is None else value
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_MAX_PER_DOC
    if isinstance(raw, str) and raw.strip().lower() in ("none", "off"):
        return None
    if isinstance(raw, bool) or int(raw) < 1:
        raise ValueError("문서당 최대 청크 수는 1 이상이어야 합니다 (끄려면 none).")
    return int(raw)


def _expand_count(value):
    """None이면 환경 변수 RETRIEVAL_EXPAND, 없으면 기본값. 0 이상의 정수(0 = 끔)."""
    raw = os.getenv("RETRIEVAL_EXPAND") if value is None else value
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_EXPAND
    if isinstance(raw, bool) or int(raw) < 0:
        raise ValueError("재정렬 후보 확장 수는 0 이상이어야 합니다 (끄려면 0).")
    return int(raw)


def _min_score(value):
    """None이면 환경 변수 RETRIEVAL_MIN_SCORE, 없으면 기본값(B는 DEFAULT_MIN_SCORE, A는 끔). none·off는 끔(None)."""
    raw = os.getenv("RETRIEVAL_MIN_SCORE") if value is None else value
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None if SCENARIO == "A" else DEFAULT_MIN_SCORE
    if isinstance(raw, str) and raw.strip().lower() in ("none", "off"):
        return None
    return float(raw)


def retrieval_options(*, rerank=None, candidates=None, max_per_doc=None, hybrid=None, hybrid_vector_k=None,
                      hybrid_bm25_k=None, rrf_k=None, rewrite=None, rewrite_model=None, bm25_prefix=None,
                      within_doc=None, expand=None, dedup=None, min_score=None, uncap_named=None):
    """retrieve가 실제로 적용하는 옵션. 우선순위: 인자 > 환경 변수 > 기본값(하이브리드 + cross-encoder + 상한 2).

    결과 파일에 이 값을 그대로 기록하면 설명과 실제 동작이 어긋나지 않습니다.
    candidates·벡터/BM25 후보 수·RRF 상수는 그 값을 쓰는 기능이 켜졌을 때만 채웁니다.
    리랭킹 방식을 지정하지 않았는데 torch·transformers가 없거나, 시나리오 A에서 A_RERANK_MODEL이 비어 있으면
    cross-encoder 대신 lexical을 쓰고 경고합니다.
    """
    mode = (os.getenv("RETRIEVAL_RERANK", "") if rerank is None else rerank).strip().lower()
    if not mode:
        mode = DEFAULT_RERANK
        if mode == "cross-encoder" and not _cross_encoder_available():
            warnings.warn("torch·transformers가 없어 기본 리랭킹을 cross-encoder 대신 lexical로 실행합니다.",
                          RuntimeWarning, stacklevel=2)
            mode = "lexical"
        # A에서 리랭커 모델을 아직 공용 폴더에 받지 않았으면, 모델이 필요 없는 lexical로 실행합니다.
        elif (mode == "cross-encoder" and SCENARIO == "A"
              and not os.getenv("A_RERANK_MODEL", "").strip()):
            warnings.warn("시나리오 A에 A_RERANK_MODEL이 없어 기본 리랭킹을 cross-encoder 대신 lexical로 실행합니다.",
                          RuntimeWarning, stacklevel=2)
            mode = "lexical"
    if mode not in RERANK_MODES:
        raise ValueError(f"지원하는 리랭킹 방식: {', '.join(RERANK_MODES)}")
    max_per_doc = _max_per_doc(max_per_doc)
    use_hybrid = (_flag(hybrid, "RETRIEVAL_HYBRID") if hybrid is not None or os.getenv("RETRIEVAL_HYBRID", "").strip()
                  else DEFAULT_HYBRID)
    rewrite_mode = (os.getenv("RETRIEVAL_REWRITE", "") if rewrite is None else rewrite).strip().lower() or "off"
    if rewrite_mode not in REWRITE_MODES:
        raise ValueError(f"지원하는 질문 재작성 방식: {', '.join(REWRITE_MODES)}")
    within_mode = (os.getenv("RETRIEVAL_WITHIN_DOC", "") if within_doc is None else within_doc).strip().lower() or "off"
    if within_mode not in WITHIN_DOC_MODES:
        raise ValueError(f"지원하는 문서 안 재선택 방식: {', '.join(WITHIN_DOC_MODES)}")
    fused = use_hybrid or rewrite_mode == "both"
    if mode != "none" or fused:
        candidates = _count(candidates, "RETRIEVAL_CANDIDATES", DEFAULT_CANDIDATES, "리랭킹 후보 수")
    else:
        candidates = None
    return {
        "rerank": mode, "candidates": candidates, "max_per_doc": max_per_doc,
        "hybrid": use_hybrid,
        "hybrid_vector_k": _count(hybrid_vector_k, "RETRIEVAL_HYBRID_VECTOR_K",
                                  DEFAULT_HYBRID_K if use_hybrid else candidates, "벡터 후보 수") if fused else None,
        "hybrid_bm25_k": _count(hybrid_bm25_k, "RETRIEVAL_HYBRID_BM25_K", DEFAULT_HYBRID_K, "BM25 후보 수")
        if use_hybrid else None,
        "bm25_prefix": (_flag(bm25_prefix, "RETRIEVAL_BM25_PREFIX")
                        if bm25_prefix is not None or os.getenv("RETRIEVAL_BM25_PREFIX", "").strip()
                        else DEFAULT_BM25_PREFIX) if use_hybrid else None,
        "rrf_k": _count(rrf_k, "RETRIEVAL_RRF_K", 60, "RRF 상수") if fused else None,
        "rewrite": rewrite_mode,
        "rewrite_model": (rewrite_model or os.getenv("RETRIEVAL_REWRITE_MODEL") or "gpt-5-mini")
        if rewrite_mode != "off" else None,
        "within_doc": within_mode,
        "expand": _expand_count(expand) if mode != "none" else None,
        "dedup": _flag(dedup, "RETRIEVAL_DEDUP") if dedup is not None or os.getenv("RETRIEVAL_DEDUP", "").strip()
        else False,
        "min_score": _min_score(min_score),
        "uncap_named": _flag(uncap_named, "RETRIEVAL_UNCAP_NAMED")
        if uncap_named is not None or os.getenv("RETRIEVAL_UNCAP_NAMED", "").strip() else False,
    }


def _bm25_index(chunks, prefix=False):
    """청크 본문 전체의 BM25 역색인 (토큰: _bigrams, 공식: _lexical_scores와 같은 k1=1.2·b=0.75).

    prefix면 본문 앞에 사업명·발주 기관을 붙여 색인합니다(_with_project). 반환 청크 본문은 바꾸지 않습니다.
    처음 한 번만 만들고 같은 chunks 목록에는 다시 씁니다. IDF는 후보가 아니라 전체 청크 기준입니다.
    """
    cached = _bm25_indexes.get((id(chunks), prefix))
    if cached is not None and cached["chunks"] is chunks:
        return cached
    vocabulary, terms, docs, counts = {}, [], [], []
    lengths = np.zeros(len(chunks), dtype="float32")
    for number, chunk in enumerate(chunks):
        grams = Counter(_bigrams(_with_project(chunk) if prefix else chunk["text"]))
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
    # 같은 chunks 목록의 본문·접두 색인만 메모리에 둡니다.
    for key in [key for key, value in _bm25_indexes.items() if value["chunks"] is not chunks]:
        del _bm25_indexes[key]
    _bm25_indexes[(id(chunks), prefix)] = built
    return built


def _bm25_ranking(question, chunks, eligible, limit, prefix=False):
    """필터를 통과한 청크 중 BM25 점수가 0보다 큰 상위 limit개의 청크 번호. prefix는 _bm25_index 참고."""
    built = _bm25_index(chunks, prefix)
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


def _filter_key(value):
    """필터 비교용 표기: NFC, '서울특별시'→'서울시', 글자·숫자만 남기고 소문자."""
    text = unicodedata.normalize("NFC", str(value)).replace("서울특별시", "서울시")
    return "".join(ch for ch in text if ch.isalnum()).casefold()


def _resolve_filters(filters, chunks):
    """정확히 일치하는 메타데이터 값이 없는 필터 값만 정규화 포함 관계로 찾아 바꿉니다.

    예: '한국철도공사' → '한국철도공사 (용역)', '서울시여성가족재단' → '서울특별시 여성가족재단'.
    후보가 하나일 때만 바꾸고, 없거나 여럿이면 원래 값을 그대로 둡니다(결과 0건).
    """
    resolved = {}
    for key, value in (filters or {}).items():
        known = {unicodedata.normalize("NFC", str(chunk["metadata"][key]))
                 for chunk in chunks if key in chunk["metadata"]}
        wanted = _filter_key(value)
        matches = {v for v in known if wanted and wanted in _filter_key(v)}
        exact = unicodedata.normalize("NFC", str(value)) in known
        resolved[key] = matches.pop() if not exact and len(matches) == 1 else value
    return resolved


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
             rewrite=None, rewrite_model=None, rewrite_cache=None, bm25_prefix=None, within_doc=None,
             expand=None, dedup=None, min_score=None, uncap_named=None):
    """반환: Chunk에 score(float)를 추가한 목록. filters는 metadata 정확 일치.
    정확히 일치하는 값이 없을 때만 표기 차이(공백·기호, '서울특별시'/'서울시')를 무시하고 그 값을 포함하는
    메타데이터 값이 하나뿐이면 그 값으로 거릅니다(_resolve_filters).

    기본값은 하이브리드 + cross-encoder + 문서당 상한 2입니다 (retrieval_options 참고).
    예전 baseline(코사인 검색만)은 rerank="none", hybrid=False, max_per_doc="none"으로 얻습니다.

    rerank: none·lexical·cross-encoder(기본). 생략하면 환경 변수 RETRIEVAL_RERANK, 그것도 없으면 기본값을 씁니다.
      기본값인데 torch·transformers가 없으면 lexical로 대체합니다. cross-encoder는 로컬 모델이라 본문을 외부로 보내지 않습니다.
    리랭킹은 필터를 통과한 후보 상위 candidates개(기본 RETRIEVAL_CANDIDATES 또는 50)만 재정렬하여
    top_k개를 반환하고 rerank_score(float)를 추가합니다. score는 항상 코사인 유사도입니다.
    max_per_doc: 한 문서(doc_id)에서 반환할 최대 청크 수(기본 2, 환경 변수 RETRIEVAL_MAX_PER_DOC). none이면 제한 없음.
    리랭킹과 함께 쓰면 재정렬된 후보에 적용하므로 후보가 소수 문서에 몰리면 top_k보다 적게 반환할 수 있습니다.

    hybrid: 켜면(기본) 벡터 상위 hybrid_vector_k개와 BM25 상위 hybrid_bm25_k개(기본 둘 다 100)를
      RRF(상수 rrf_k, 기본 60)로 합쳐 후보를 만들고, 그 후보에 rerank를 적용합니다.
      합친 후보에는 rrf_score(float)가 추가되고 score는 여전히 질문과의 코사인 유사도입니다.
      환경 변수: RETRIEVAL_HYBRID, RETRIEVAL_HYBRID_VECTOR_K, RETRIEVAL_HYBRID_BM25_K, RETRIEVAL_RRF_K.
    bm25_prefix: 하이브리드의 BM25 색인 본문 앞에 사업명·발주 기관을 붙입니다(기본 켬, 환경 변수
      RETRIEVAL_BM25_PREFIX). 반환 청크의 text와 score는 그대로입니다. 하이브리드를 끄면 쓰이지 않습니다.
    rewrite: off(기본)·only·both. only는 재작성 질문만으로, both는 원문과 재작성 질문을 각각 검색해
      RRF로 합칩니다. 재작성은 query_rewrite.rewrite_question이 하며 실패하거나 비면 원문으로 돌아갑니다.
      rewrite_cache(파일 경로)를 주면 같은 질문은 다시 호출하지 않습니다. 환경 변수: RETRIEVAL_REWRITE,
      RETRIEVAL_REWRITE_MODEL(기본 gpt-5-mini). 리랭킹에는 검색에 쓴 질문들을 이어 붙인 문장을 씁니다.
    within_doc: off(기본)·residual·residual+full. 위 과정이 끝난 뒤 문서가 차지한 자리 수·순서는 그대로 두고,
      각 자리의 청크만 그 문서의 전체 청크에서 다시 고릅니다(_reselect_within_docs). 환경 변수 RETRIEVAL_WITHIN_DOC.
      켜면 임베딩 API를 한 번 더 호출합니다(원래 질문과 문서별 남은 질문을 한 번에).
    expand: 재정렬 후보 확장(기본 5, 0이면 끔, 환경 변수 RETRIEVAL_EXPAND). 재정렬할 때만 쓰며, 후보에 처음 나온
      문서 2개에서 남은 질문(_residual_question)의 문서 안 BM25 상위 expand개 청크를 후보에 더해 재정렬기가
      순서를 정하게 합니다. 사업명이 질문 대부분을 차지해 정답 청크가 후보에 못 드는 경우(발표 시간 등)를 위한 것.
      더한 청크는 expanded=True를 갖고 rrf_score는 없습니다. API 호출은 늘지 않습니다.
    dedup: 켜면(기본 끔, 환경 변수 RETRIEVAL_DEDUP) 최종 결과에서 같은 내용이 자리를 겹쳐 차지하지 않게 합니다.
      본문이 완전히 같은 문서(같은 공고가 다른 파일명으로 두 번 들어온 경우)는 한 문서로 보고 max_per_doc을 함께 세며,
      이미 고른 청크와 본문이 똑같은 청크(문서 사이의 같은 서식 문구 등)는 건너뜁니다. 먼저 나온(순위가 높은) 쪽을 남깁니다.
    uncap_named: 질문이 사업 하나를 이름으로 가리키면 그 사업의 문서당 상한을 풉니다(기본 끔, RETRIEVAL_UNCAP_NAMED).
    min_score: 범위 밖 질문 차단(기본 B 0.40·A 끔, none·off로 끔, 환경 변수 RETRIEVAL_MIN_SCORE). 필터를 통과한 청크 중 질문과의 코사인이
      가장 높은 값도 이보다 낮으면 RFP와 무관한 질문으로 보고 빈 목록을 돌려줍니다. 그러면 generate_answer가
      생성 API를 부르지 않고 안내 문구를 답합니다. 질문 임베딩 호출은 그대로 한 번 일어납니다.
    """
    if not question.strip() or top_k < 1:
        raise ValueError("비어 있지 않은 질문과 1 이상의 top_k가 필요합니다.")
    options = retrieval_options(rerank=rerank, candidates=candidates, max_per_doc=max_per_doc, hybrid=hybrid,
                                hybrid_vector_k=hybrid_vector_k, hybrid_bm25_k=hybrid_bm25_k, rrf_k=rrf_k,
                                rewrite=rewrite, rewrite_model=rewrite_model, bm25_prefix=bm25_prefix,
                                within_doc=within_doc, expand=expand, dedup=dedup, min_score=min_score,
                                uncap_named=uncap_named)
    mode, candidates, max_per_doc = options["rerank"], options["candidates"], options["max_per_doc"]
    pool = top_k if mode == "none" else max(top_k, candidates)
    normalize = lambda value: unicodedata.normalize("NFC", str(value))
    filters = _resolve_filters(filters, chunks)
    eligible = {i for i, chunk in enumerate(chunks) if all(
        key in chunk["metadata"] and normalize(chunk["metadata"][key]) == normalize(value)
        for key, value in (filters or {}).items()
    )}
    if not eligible:
        record_retrieval([])
        return []
    if options["hybrid"] or options["rewrite"] != "off":
        hits = _fused_retrieve(question, client, index, chunks, config, top_k, pool, eligible, options, rewrite_cache)
        return _finish(question, hits, client, index, chunks, config, eligible, options)
    vector = embed_texts([question], client, config["embedding_model"])
    if vector.shape[1] != index.d:
        raise ValueError("질문 임베딩 차원이 인덱스와 다릅니다.")
    # 소규모 baseline: 전체 순위를 얻고 필터를 적용하여 후보를 채웁니다.
    # 문서당 상한·중복 제거가 있으면 건너뛰는 청크만큼 더 아래 순위까지 봐야 합니다.
    scores, ids = index.search(vector, index.ntotal if filters or max_per_doc or options["dedup"]
                               else min(pool, index.ntotal))
    best = next(float(score) for score, i in zip(scores[0], ids[0]) if int(i) in eligible)
    if options["min_score"] is not None and best < options["min_score"]:
        record_retrieval([])
        return []
    # 리랭킹을 할 때는 후보를 줄이지 않고 재정렬한 뒤에 상한을 적용합니다.
    take = _selector(chunks, options)
    hits = []
    for score, i in zip(scores[0], ids[0]):
        if int(i) in eligible and (mode != "none" or take(chunks[int(i)])):
            hits.append({**chunks[int(i)], "score": float(score)})
            if len(hits) == pool:
                break
    if mode != "none":
        hits = _expand_candidates(question, hits, chunks, eligible, options["expand"],
                                  lambda number: float(index.reconstruct(number) @ vector[0]))
        # 후보는 이미 필터를 통과했으므로 재정렬해도 필터 밖 문서가 포함되지 않습니다.
        ranked = observed(f"rerank-{mode}")(_rerank)(question, hits, mode)
        ranked = _aspect_slot(question, ranked, mode)
        ranked, picked = _named_slot(question, ranked, chunks, eligible, mode,
                                     lambda number: float(index.reconstruct(number) @ vector[0]), options)
        take = _selector(chunks, picked)
        hits = [hit for hit in ranked if take(hit)][:top_k]
    return _finish(question, hits, client, index, chunks, config, eligible, options)


def _finish(question, hits, client, index, chunks, config, eligible, options):
    """두 검색 경로의 공통 마무리: 문서 안 재선택(켰을 때)과 결과 기록."""
    if options["within_doc"] != "off" and hits:
        hits = observed("within-doc")(_reselect_within_docs)(question, hits, client, index, chunks, config,
                                                             eligible, options["within_doc"], options["dedup"])
    record_retrieval(hits)
    return hits


def with_next_chunk(hits, chunks):
    """1위 청크의 바로 다음 청크(같은 문서)를 맨 뒤에 덧붙입니다. 이미 결과에 있으면 그대로 둡니다.

    목록·표가 청크 경계에서 잘리면 뒷부분 항목이 빠집니다(eval_v4 yA048: 입찰 참가 자격 목록이 다음 청크로 이어짐).
    덧붙인 청크는 neighbor=True이고 score는 1위 청크 값을 그대로 씁니다(질문과 따로 비교하지 않음).
    retrieve()의 top_k 결과는 그대로 두고, b_run이 생성 직전에 부릅니다.
    """
    if not hits:
        return hits
    numbers = _members(chunks)[hits[0]["doc_id"]]
    ids = [chunks[number]["chunk_id"] for number in numbers]
    if hits[0]["chunk_id"] not in ids:
        return hits
    after = ids.index(hits[0]["chunk_id"]) + 1
    if after == len(ids) or any(hit["chunk_id"] == ids[after] for hit in hits):
        return hits
    return hits + [{**chunks[numbers[after]], "score": hits[0]["score"], "neighbor": True}]


def _members(chunks):
    """doc_id별 청크 번호. 같은 chunks 목록에는 다시 만들지 않습니다."""
    cached = _doc_members.get(id(chunks))
    if cached is None or cached[0] is not chunks:
        groups = {}
        for number, chunk in enumerate(chunks):
            groups.setdefault(chunk["doc_id"], []).append(number)
        _doc_members.clear()
        _doc_members[id(chunks)] = cached = (chunks, groups)
    return cached[1]


def _residual_question(question, metadata):
    """질문에서 그 문서의 사업명·발주 기관과 글자 2-gram이 절반 이상 겹치는 단어를 뺀 '남은 질문'. 다 빠지면 원래 질문."""
    project = set(_bigrams(" ".join(str(metadata.get(key, "")) for key in RERANK_FIELDS)))
    kept = []
    for word in unicodedata.normalize("NFC", question).split():
        grams = _bigrams(word)
        if not grams or sum(gram in project for gram in grams) / len(grams) < .5:
            kept.append(word)
    return " ".join(kept) or question


def _aspect_sentence(question, matched_text):
    """여러 문장 질문에서 이미 찾은 청크와 가장 덜 겹치는 문장(묻는 항목). 한 문장이면 None.

    예: '다음 내용의 사업에서 입찰 참가 자격은 무엇인가요? <사업 설명>'에서 사업 설명은 찾은 청크와 겹치므로
    '…입찰 참가 자격은 무엇인가요?'를 고릅니다.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[?？.!])\s+|\n+", unicodedata.normalize("NFC", question)) if s.strip()]
    if len(sentences) < 2:
        return None
    grams = set(_bigrams(matched_text))
    return min(sentences, key=lambda s: sum(g in grams for g in _bigrams(s)) / max(len(_bigrams(s)), 1))


def _aspect_slot(question, ranked, mode):
    """여러 문장 질문에서 1위 문서의 후보를 묻는 문장(_aspect_sentence)으로 다시 재정렬해 그 순서로 맨 앞에 둡니다.

    사업 설명이 질문 대부분이면 재정렬이 설명과 닮은 개요 청크만 올려 문서당 상한을 채우므로,
    1위 문서의 청크는 묻는 항목에 맞는 순서로 바꿉니다(사업명은 메타데이터로 함께 전달됨). 다른 문서 순서와 한 문장 질문은 그대로입니다.
    """
    aspect = _aspect_sentence(question, ranked[0]["text"]) if ranked else None
    if not aspect:
        return ranked
    same = [hit for hit in ranked if hit["doc_id"] == ranked[0]["doc_id"]]
    order = [hit["chunk_id"] for hit in _rerank(aspect, same, mode)]
    by_id = {hit["chunk_id"]: hit for hit in same}
    return [*(by_id[c] for c in order), *(hit for hit in ranked if hit["doc_id"] != ranked[0]["doc_id"])]


def _name_grams(chunks):
    """doc_id별 사업명의 '그 사업만의' 글자 2-gram. 여러 사업명에 흔한 2-gram(정보·시스템·구축 등, 전체 문서의 5% 초과)은 뺍니다."""
    cached = _name_gram_cache.get(id(chunks))
    if cached is None or cached[0] is not chunks:
        names = {doc_id: set(_bigrams(str(chunks[numbers[0]]["metadata"].get("사업명", ""))))
                 for doc_id, numbers in _members(chunks).items()}
        df = Counter(gram for grams in names.values() for gram in grams)
        common = max(1, len(names) * .05)
        _name_gram_cache.clear()
        _name_gram_cache[id(chunks)] = cached = (chunks, {d: {g for g in grams if df[g] <= common}
                                                          for d, grams in names.items()})
    return cached[1]


def _named_docs(question, ranked, chunks, strict=False):
    """후보 문서 중 질문이 이름으로 가리킨 사업, 후보 순서대로. 사업명 고유 2-gram이 3개 이상 질문에 나오고
    그 절반 이상이 나오거나(strict면 이것만) 5개 이상 나오면 가리킨 것으로 봅니다.

    2개만 겹쳐도 되면 '컨설팅'·'ERP' 한 단어로 다른 사업이 잡힙니다(평가 질문 247개 중 오탐 4건 → 3개 이상이면 0건).
    5개 이상 조건은 '대전대학교 MILE 전산시스템'처럼 긴 사업명을 줄여 쓴 비교 질문을 잡기 위한 것입니다.
    """
    asked, own = set(_bigrams(question)), _name_grams(chunks)
    named = []
    for hit in ranked:
        grams = own.get(hit["doc_id"], set())
        count = len(grams & asked)
        if hit["doc_id"] not in named and count >= 3 and (count / len(grams) >= .5 or (count >= 5 and not strict)):
            named.append(hit["doc_id"])
    return named


def _named_slot(question, ranked, chunks, eligible, mode, score_of, options):
    """질문이 이름으로 가리킨 사업에 맞춰 순서와 문서당 상한을 정합니다. 반환: (ranked, options).

    - 둘 이상(비교 질문): 그 문서들(COMPARE_DOCS개까지)의 전체 청크를 재정렬해 문서별로 번갈아 맨 앞에 둡니다.
      사업 개요 청크가 후보를 차지해 묻는 항목(예: 제안서 부수)의 청크가 후보 50개에 들지 못하기 때문입니다.
    - 하나(엄격한 기준으로도 하나)이고 uncap_named(기본 끔, RETRIEVAL_UNCAP_NAMED)면: 문서당 상한을 풉니다. 다른 사업 청크로 남은 자리를 채우는 대신 그 사업의 청크를 더 넣어
      한 문서 안에 서로 다르게 적힌 값(예: 하자담보 3년과 1년)을 함께 보게 합니다.
    - 그 밖(이름 없음·목록 질문 등)은 그대로입니다.
    """
    named = _named_docs(question, ranked, chunks)[:COMPARE_DOCS]
    if len(named) == 1:
        lift = options.get("uncap_named") and _named_docs(question, ranked, chunks, strict=True) == named
        return ranked, ({**options, "max_per_doc": None} if lift else options)
    if len(named) < 2:
        return ranked, options
    members, known = _members(chunks), {hit["chunk_id"]: hit for hit in ranked}
    queues = []
    for doc_id in named:
        hits = [known.get(chunks[n]["chunk_id"]) or {**chunks[n], "score": score_of(n)}
                for n in members.get(doc_id, []) if n in eligible]
        queues.append(_rerank(question, hits, mode))
    front = [hit for row in zip_longest(*queues) for hit in row if hit]
    seen = {hit["chunk_id"] for hit in front}
    return [*front, *(hit for hit in ranked if hit["chunk_id"] not in seen)], options


def _expand_candidates(question, hits, chunks, eligible, count, score_of):
    """재정렬 후보 확장: 후보에 처음 나온 문서 EXPAND_DOCS개에서 남은 질문의 문서 안 BM25 상위 count개를 더합니다.

    이미 후보인 청크는 건너뛰고 문서마다 최대 count개를 뒤에 붙입니다. 순서는 이어지는 재정렬이 정합니다.
    score_of(청크 번호)는 질문과의 코사인을 돌려줍니다.
    """
    if not count or not hits:
        return hits
    members = _members(chunks)
    seen = {hit["chunk_id"] for hit in hits}
    extra = []
    for doc_id in list(dict.fromkeys(hit["doc_id"] for hit in hits))[:EXPAND_DOCS]:
        ids = [number for number in members.get(doc_id, []) if number in eligible]
        first = next(hit for hit in hits if hit["doc_id"] == doc_id)
        rest = _aspect_sentence(question, first["text"]) or _residual_question(question, chunks[ids[0]]["metadata"])
        added = 0
        for number in _bm25_ranking(rest, chunks, set(ids), len(ids)):
            if added == count:
                break
            if chunks[number]["chunk_id"] not in seen:
                seen.add(chunks[number]["chunk_id"])
                extra.append({**chunks[number], "score": score_of(number), "expanded": True})
                added += 1
    return hits + extra


def _reselect_within_docs(question, hits, client, index, chunks, config, eligible, mode, dedup=False):
    """같은 문서 안에서 청크 다시 고르기 (eval_v2 yjk-0013).

    문서별로 남은 질문의 BM25(문서 안, 접두 없음)와 벡터 코사인 순위를 RRF(60)로 합치고, residual+full이면
    원래 질문의 두 순위도 더합니다. 문서가 차지한 자리마다 그 순서대로 청크를 넣으며 within_doc_rank(1부터)를 붙입니다.
    같은 청크가 다시 뽑히면 원래 Hit(rerank_score·rrf_score 포함)을 그대로 쓰고, 새 청크는 score(질문과의 코사인)만 가집니다.
    dedup이면 이미 고른 청크와 본문이 같은 청크는 건너뜁니다. 그 문서에 남은 청크가 모두 겹치면 그 자리는 비웁니다.
    """
    members = _members(chunks)
    residuals = {}
    for hit in hits:
        residuals.setdefault(hit["doc_id"], _residual_question(question, hit["metadata"]))
    texts = list(dict.fromkeys([question, *residuals.values()]))
    vectors = dict(zip(texts, embed_texts(texts, client, config["embedding_model"])))
    if vectors[question].shape[0] != index.d:
        raise ValueError("질문 임베딩 차원이 인덱스와 다릅니다.")
    original = {hit["chunk_id"]: hit for hit in hits}
    queues, reselected, taken = {}, [], set()
    for hit in hits:
        doc_id = hit["doc_id"]
        if doc_id not in queues:
            ids = [number for number in members.get(doc_id, []) if number in eligible]
            stored = np.vstack([index.reconstruct(number) for number in ids])
            cosine = stored @ vectors[question]
            rest = residuals[doc_id]
            rankings = [_bm25_ranking(rest, chunks, set(ids), len(ids)),
                        [ids[j] for j in np.argsort(-(stored @ vectors[rest]), kind="stable")]]
            if mode == "residual+full":
                rankings += [_bm25_ranking(question, chunks, set(ids), len(ids)),
                             [ids[j] for j in np.argsort(-cosine, kind="stable")]]
            scores = dict(zip(ids, cosine.tolist()))
            queues[doc_id] = iter([(rank, number, scores[number])
                                   for rank, (number, _) in enumerate(_rrf(rankings, 60), 1)])
        picked = next((item for item in queues[doc_id] if not dedup or chunks[item[1]]["text"] not in taken), None)
        if picked is None:
            continue
        rank, number, score = picked
        chunk = chunks[number]
        taken.add(chunk["text"])
        base = original.get(chunk["chunk_id"]) or {**chunk, "score": float(score)}
        reselected.append({**base, "within_doc_rank": rank})
    return reselected


def _fused_retrieve(question, client, index, chunks, config, top_k, pool, eligible, options, rewrite_cache):
    """하이브리드·질문 재작성 경로. 질문마다 벡터(와 BM25) 순위를 만들고 RRF로 합친 뒤 기존과 같이 리랭킹합니다."""
    queries = [question]
    if options["rewrite"] != "off":
        from b_query_rewrite import rewrite_question  # 재작성을 켰을 때만 불러옵니다.

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
            rankings.append(_bm25_ranking(query, chunks, eligible, options["hybrid_bm25_k"], options["bm25_prefix"]))
    if options["min_score"] is not None and max(cosine[i] for i in eligible) < options["min_score"]:
        return []
    fused = _rrf(rankings, options["rrf_k"] or 60)
    mode = options["rerank"]
    take = _selector(chunks, options)
    hits = []
    for i, rrf_score in fused:
        if mode != "none" or take(chunks[i]):
            hits.append({**chunks[i], "score": cosine[i], "rrf_score": rrf_score})
            if len(hits) == pool:
                break
    if mode != "none":
        hits = _expand_candidates(question, hits, chunks, eligible, options["expand"], lambda number: cosine[number])
        ranked = observed(f"rerank-{mode}")(_rerank)(" ".join(queries), hits, mode)
        ranked = _aspect_slot(question, ranked, mode)
        ranked, picked = _named_slot(question, ranked, chunks, eligible, mode, lambda number: cosine[number], options)
        take = _selector(chunks, picked)
        hits = [hit for hit in ranked if take(hit)][:top_k]
    return hits


def _selector(chunks, options):
    """최종 결과에 넣을 청크를 순위 순서대로 하나씩 받아 넣을지(True) 정하는 함수를 만듭니다.

    max_per_doc: 한 문서에서 이미 그만큼 골랐으면 뺍니다.
    dedup: 본문이 완전히 같은 문서들은 먼저 나온 doc_id 하나로 묶어 상한을 함께 세고,
      이미 고른 청크와 본문이 똑같은 청크는 뺍니다.
    """
    max_per_doc, dedup = options["max_per_doc"], options["dedup"]
    same_doc = {}
    if dedup:
        # 문서마다 청크 본문을 순서대로 이은 것이 같으면 같은 문서입니다.
        first = {}
        for doc_id, numbers in _members(chunks).items():
            same_doc[doc_id] = first.setdefault(tuple(chunks[n]["text"] for n in numbers), doc_id)
    per_doc, taken_texts = Counter(), set()

    def take(chunk):
        doc_id = same_doc.get(chunk["doc_id"], chunk["doc_id"])
        if max_per_doc is not None and per_doc[doc_id] >= max_per_doc:
            return False
        if dedup and chunk["text"] in taken_texts:
            return False
        per_doc[doc_id] += 1
        taken_texts.add(chunk["text"])
        return True
    return take
