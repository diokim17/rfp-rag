"""유찬혁: Document → Chunk → OpenAI 임베딩 → FAISS 인덱스."""

from pathlib import Path

import faiss
import numpy as np

from parsing import read_json, write_json
from observability import observed, model_call


@observed("chunking")
def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    """문자 수 기준 기본 청킹. 반환: [{chunk_id, doc_id, text, metadata}]."""
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("0 <= chunk_overlap < chunk_size 조건이 필요합니다.")
    chunks = []
    for document in documents:
        text = document["text"]
        for number, start in enumerate(range(0, len(text), chunk_size - chunk_overlap)):
            end = min(start + chunk_size, len(text))
            if text[start:end].strip():
                chunks.append({
                    "chunk_id": f"{document['doc_id']}:{number}",
                    "doc_id": document["doc_id"], "text": text[start:end],
                    "metadata": {**document["metadata"], "start_char": start, "end_char": end},
                })
            if end == len(text):
                break
    return chunks


def embed_texts(texts, client, model):
    """문서·질문에 공통으로 사용하는 정규화된 float32 벡터."""
    response = model_call("embedding", model,
                         lambda: client.embeddings.create(model=model, input=texts),
                         batch_size=len(texts))
    vectors = np.asarray([item.embedding for item in sorted(response.data, key=lambda x: x.index)], dtype="float32")
    if len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("임베딩 응답이 입력과 맞지 않습니다.")
    faiss.normalize_L2(vectors)
    return vectors


@observed("build-index")
def build_index(documents, client, index_dir="indexes", model="text-embedding-3-small",
                chunk_size=1000, chunk_overlap=150):
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("인덱스를 생성할 청크가 없습니다.")
    index = None
    for start in range(0, len(chunks), 32):
        vectors = embed_texts([c["text"] for c in chunks[start:start + 32]], client, model)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    # 전체 임베딩 성공 후 저장합니다. 빌드 중에는 질의를 실행하지 마세요.
    faiss.write_index(index, str(path / "index.faiss"))
    write_json(path / "chunks.json", chunks)
    config = {"embedding_model": model, "dimension": index.d, "chunk_count": len(chunks),
              "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    write_json(path / "config.json", config)
    return config


def load_index(index_dir="indexes"):
    path = Path(index_dir)
    index = faiss.read_index(str(path / "index.faiss"))
    chunks, config = read_json(path / "chunks.json"), read_json(path / "config.json")
    if index.ntotal != len(chunks) or index.d != config["dimension"] or len(chunks) != config["chunk_count"]:
        raise ValueError("인덱스와 메타데이터 불일치: build를 다시 실행하세요.")
    return index, chunks, config
