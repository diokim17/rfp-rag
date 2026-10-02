"""유찬혁: Document → Chunk → OpenAI 임베딩 → FAISS 인덱스."""

import os
from pathlib import Path
import re

import faiss
import numpy as np

from parsing import read_json, write_json
from observability import observed, model_call


_BOUNDARIES = tuple(re.compile(pattern) for pattern in (
    r"\r?\n(?:[ \t]*\r?\n)+", r"\r?\n", r"[.!?。！？]\s+",
))


def _chunking_strategy():
    strategy = os.getenv("RFP_CHUNKING_STRATEGY", "fixed")
    if strategy not in {"fixed", "boundary"}:
        raise ValueError("RFP_CHUNKING_STRATEGY는 fixed 또는 boundary여야 합니다.")
    return strategy


def _boundary_end(text, start, end, chunk_size, chunk_overlap):
    minimum = start + max((chunk_size + 1) // 2, chunk_overlap + 1)
    for pattern in _BOUNDARIES:
        candidate = None
        for match in pattern.finditer(text, start, end):
            if match.end() >= minimum:
                candidate = match.end()
        if candidate is not None:
            return candidate
    return end


@observed("chunking")
def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    """문자 기준 청킹. RFP_CHUNKING_STRATEGY=boundary로 원문 경계를 우선합니다.

    기본 fixed 결과와 원문 문자 위치를 유지합니다. 반환: Chunk dict 목록.
    """
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("0 <= chunk_overlap < chunk_size 조건이 필요합니다.")
    strategy = _chunking_strategy()
    chunks = []
    for document in documents:
        text = document["text"]
        start, number = 0, 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if strategy == "boundary" and end < len(text):
                end = _boundary_end(text, start, end, chunk_size, chunk_overlap)
            if text[start:end].strip():
                chunks.append({
                    "chunk_id": f"{document['doc_id']}:{number}",
                    "doc_id": document["doc_id"], "text": text[start:end],
                    "metadata": {**document["metadata"], "start_char": start, "end_char": end},
                })
            if end == len(text):
                break
            start = end - chunk_overlap
            number += 1
    return chunks


def embed_texts(texts, client, model):
    """문서·질문에 공통으로 사용하는 정규화된 float32 벡터."""
    if not texts:
        raise ValueError("임베딩할 텍스트가 없습니다.")
    response = model_call("embedding", model,
                         lambda: client.embeddings.create(model=model, input=texts),
                         batch_size=len(texts))
    items = response.data
    indices = [item.index for item in items]
    if (any(type(i) is not int for i in indices)
            or sorted(indices) != list(range(len(texts)))):
        raise ValueError("임베딩 응답 index가 입력과 맞지 않습니다.")
    try:
        vectors = np.asarray([item.embedding for item in sorted(items, key=lambda x: x.index)], dtype="float32")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("임베딩 응답 벡터 형식이 올바르지 않습니다.") from exc
    if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("임베딩 응답이 입력과 맞지 않습니다.")
    # 먼저 스케일을 맞춰 float32 제곱합의 overflow/underflow를 방지합니다.
    scales = np.max(np.abs(vectors), axis=1)
    if np.any(scales == 0):
        raise ValueError("임베딩 응답에 영벡터가 있습니다.")
    vectors /= scales[:, None]
    faiss.normalize_L2(vectors)
    return vectors


@observed("build-index")
def build_index(documents, client, index_dir="indexes", model="text-embedding-3-small",
                chunk_size=1000, chunk_overlap=150):
    strategy = _chunking_strategy()
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("인덱스를 생성할 청크가 없습니다.")
    index = None
    for start in range(0, len(chunks), 32):
        vectors = embed_texts([c["text"] for c in chunks[start:start + 32]], client, model)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        elif vectors.shape[1] != index.d:
            raise ValueError("임베딩 배치 사이의 벡터 차원이 다릅니다.")
        index.add(vectors)
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    # 전체 임베딩 성공 후 저장합니다. 빌드 중에는 질의를 실행하지 마세요.
    faiss.write_index(index, str(path / "index.faiss"))
    write_json(path / "chunks.json", chunks)
    config = {"embedding_model": model, "dimension": index.d, "chunk_count": len(chunks),
              "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
              "chunking_strategy": strategy, "chunking_version": 1}
    write_json(path / "config.json", config)
    return config


def load_index(index_dir="indexes"):
    path = Path(index_dir)
    index = faiss.read_index(str(path / "index.faiss"))
    chunks, config = read_json(path / "chunks.json"), read_json(path / "config.json")
    if index.ntotal != len(chunks) or index.d != config["dimension"] or len(chunks) != config["chunk_count"]:
        raise ValueError("인덱스와 메타데이터 불일치: build를 다시 실행하세요.")
    return index, chunks, config
