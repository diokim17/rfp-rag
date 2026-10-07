"""유찬혁: Document → Chunk → OpenAI 임베딩 → FAISS 인덱스."""

from bisect import bisect_right
import os
from pathlib import Path
import re
import unicodedata

import faiss
import numpy as np

from parsing import read_json, section_of, write_json
from observability import observed, model_call


_BOUNDARIES = tuple(re.compile(pattern) for pattern in (
    r"\r?\n(?:[ \t]*\r?\n)+", r"\r?\n", r"[.!?。！？]\s+",
))
_TABLE_MARKER = re.compile(r"<!-- (?P<close>/)?table(?::(?P<id>T\d+))? -->")
_TABLE_HEADER = re.compile(
    r"\|[^\r\n]*\|[ \t]*\r?\n[ \t]*\|(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*(?=\r?\n|$)"
)
_NEWLINE = re.compile("\n")
_NONSPACE = re.compile(r"\S")
_TOC_HEADING = re.compile(r"(?m)^[ \t]*(?:목[ \t]*차|<[ \t]*목[ \t]*차[ \t]*>)[ \t]*\r?$")
# project_blend v1의 고정 비중. 변경 시 embedding_context_version을 올립니다.
_PROJECT_CONTEXT_WEIGHT = 0.2


def _chunking_strategy():
    strategy = os.getenv("RFP_CHUNKING_STRATEGY", "fixed")
    if strategy not in {"fixed", "boundary", "structured"}:
        raise ValueError("RFP_CHUNKING_STRATEGY는 fixed, boundary 또는 structured여야 합니다.")
    return strategy


def _embedding_context():
    context = os.getenv("RFP_EMBEDDING_CONTEXT", "none")
    if context not in {"none", "project", "project_blend"}:
        raise ValueError("RFP_EMBEDDING_CONTEXT는 none, project 또는 project_blend여야 합니다.")
    return context


def _embedding_input(chunk, context):
    # 저장 본문/metadata는 그대로 두고 문서 임베딩 요청에만 문맥을 붙입니다.
    lines = []
    if context == "project":
        for key in ("사업명", "발주 기관"):
            value = chunk["metadata"].get(key)
            if isinstance(value, str):
                value = " ".join(unicodedata.normalize("NFC", value).split())
                if value:
                    lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n\n" + chunk["text"] if lines else chunk["text"]


def _document_vectors(chunks, client, model, context):
    """문서 전용 처리. blend는 본문 80% + 문맥 포함 입력 20% 후 재정규화.

    문맥이 있는 청크는 두 번 임베딩하므로 추가 토큰을 사용합니다.
    none/project와 질문 임베딩 경로는 유지합니다.
    """
    if context != "project_blend":
        return embed_texts([_embedding_input(c, context) for c in chunks], client, model)
    bodies = [c["text"] for c in chunks]
    vectors = embed_texts(bodies, client, model)
    inputs = [_embedding_input(c, "project") for c in chunks]
    positions = [i for i, text in enumerate(inputs) if text != bodies[i]]
    if positions:
        contextual = embed_texts([inputs[i] for i in positions], client, model)
        if contextual.shape[1] != vectors.shape[1]:
            raise ValueError("본문과 문맥 임베딩의 벡터 차원이 다릅니다.")
        mixed = ((1 - _PROJECT_CONTEXT_WEIGHT) * vectors[positions]
                 + _PROJECT_CONTEXT_WEIGHT * contextual)
        faiss.normalize_L2(mixed)
        vectors[positions] = mixed
    return vectors


def _table_spans(text):
    """짝이 맞는 비중첩 표만 사용합니다. 잘못된 표시는 일반 본문으로 처리."""
    spans, stack, invalid = [], [], False
    for match in _TABLE_MARKER.finditer(text):
        if not match["close"]:
            invalid = bool(stack)
            stack.append((match.start(), match["id"]))
        elif stack:
            start, table_id = stack.pop()
            invalid = invalid or table_id != match["id"]
            if not stack and not invalid:
                spans.append((start, match.end()))
    return spans


def _containing_span(pos, spans, starts):
    i = bisect_right(starts, pos) - 1
    if i >= 0 and spans[i][0] < pos < spans[i][1]:
        return spans[i]
    return None


def _table_entries(text, spans):
    """표 헤더는 본문에 복제하지 않고 원문 슬라이스와 문서 기준 좌표로 보관."""
    entries = []
    for start, end in spans:
        marker = _TABLE_MARKER.match(text, start)
        first = _NONSPACE.search(text, marker.end(), end)
        header = _TABLE_HEADER.match(text, first.start(), end) if first else None
        entries.append({
            "table_id": marker["id"],
            "start_char": start, "end_char": end,
            "header_text": header.group() if header else "",
            "header_start_char": header.start() if header else None,
            "header_end_char": header.end() if header else None,
        })
    return entries


def _chunk_tables(entries, ends, start, end):
    # 구간은 [start, end). 인접 표나 이전 표를 잘못 붙이지 않습니다.
    result = []
    i = bisect_right(ends, start)
    while i < len(entries) and entries[i]["start_char"] < end:
        result.append(dict(entries[i]))
        i += 1
    return result


def _section_entries(document):
    """구형/불완전한 sections를 원본 변경 없이 안전한 위치 목록으로 정리."""
    raw_sections = document.get("sections")
    if not isinstance(raw_sections, (list, tuple)):
        return []
    text = document["text"]
    return sorted((item for item in raw_sections
                   if isinstance(item, (list, tuple)) and len(item) == 2
                   and type(item[0]) is int and 0 <= item[0] < len(text)
                   and isinstance(item[1], str) and item[1].strip()
                   and (item[0] == 0 or text[item[0] - 1] == "\n")), key=lambda item: item[0])


def _structure(document, chunk_size, table_spans=None):
    """원문 좌표의 섹션 경계와 보호 구간(짧은 표/큰 표의 행/제목)을 준비."""
    text = document["text"]
    sections = sorted({pos for pos, _ in _section_entries(document)})
    tables = _table_spans(text) if table_spans is None else table_spans
    table_starts = [a for a, _ in tables]
    protected = []
    for start, end in tables:
        if end - start <= chunk_size:
            protected.append((start, end))
        else:
            # 긴 행도 구간에 포함: 행 자체가 제한을 넘을 때만 문자 분할합니다.
            row_start = start
            for match in _NEWLINE.finditer(text, start, end):
                row_end = match.end()
                protected.append((row_start, row_end))
                row_start = row_end
            if row_start < end:
                protected.append((row_start, end))
    for pos in sections:
        # 표 안의 제목처럼 보이는 텍스트는 표 규칙에 맡깁니다.
        i = bisect_right(table_starts, pos) - 1
        if i >= 0 and pos < tables[i][1]:
            continue
        line_end = text.find("\n", pos)
        next_text = _NONSPACE.search(text, line_end + 1) if line_end >= 0 else None
        body_end = text.find("\n", next_text.start()) if next_text else -1
        end = body_end + 1 if body_end >= 0 else len(text)
        # 제목 직후 표가 있더라도 표와 합쳐 큰 보호 구간을 만들지 않습니다.
        next_table = bisect_right(table_starts, pos)
        if next_table < len(tables):
            end = min(end, tables[next_table][0])
        if end > pos:
            protected.append((pos, end))
    merged = []
    for start, end in sorted(protected):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sections, merged, [a for a, _ in merged]


def _structured_end(text, start, size, overlap, structure):
    sections, spans, starts = structure
    limit = min(start + size, len(text))
    if limit == len(text):
        return limit
    minimum = start + max((size + 1) // 2, overlap + 1)
    i = bisect_right(sections, limit) - 1
    use_section = i >= 0 and sections[i] >= minimum
    if use_section and start == 0 and i == 0 and _TOC_HEADING.search(text, 0, sections[0]):
        # 목차만 담긴 첫 청크를 만들지 않도록 첫 개요의 문단 경계를 사용합니다.
        # 크기 제한과 아래 표/제목 보호 규칙은 그대로 적용합니다.
        use_section = False
    end = (sections[i] if use_section
           else _boundary_end(text, start, limit, size, overlap))
    span = _containing_span(end, spans, starts)
    if span:
        left, right = span
        if right <= limit:
            return right
        if left > start:
            return left
        return limit  # 행/제목 자체가 너무 길 때는 크기 제한을 지킵니다.
    return end


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
    """문자 기준 청킹. boundary는 문장, structured는 섹션/표 경계를 우선.

    structured의 실제 중복은 chunk_overlap 이하입니다. 원문 위치/기본 fixed 유지.
    반환: Chunk dict 목록.
    """
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("0 <= chunk_overlap < chunk_size 조건이 필요합니다.")
    strategy = _chunking_strategy()
    chunks = []
    for document in documents:
        text = document["text"]
        section_document = {**document, "sections": _section_entries(document)}
        table_spans = _table_spans(text)
        tables = _table_entries(text, table_spans)
        table_ends = [end for _, end in table_spans]
        structure = _structure(section_document, chunk_size, table_spans) if strategy == "structured" else None
        start, number = 0, 0
        previous_end = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if strategy == "boundary" and end < len(text):
                end = _boundary_end(text, start, end, chunk_size, chunk_overlap)
            elif strategy == "structured":
                end = _structured_end(text, start, chunk_size, chunk_overlap, structure)
                if end <= previous_end:
                    # 표 앞에 남은 overlap 때문에 같은 내용만 반복하지 않습니다.
                    start = previous_end
                    end = _structured_end(text, start, chunk_size, chunk_overlap, structure)
            if text[start:end].strip():
                chunks.append({
                    "chunk_id": f"{document['doc_id']}:{number}",
                    "doc_id": document["doc_id"], "text": text[start:end],
                    "metadata": {**document["metadata"], "start_char": start, "end_char": end,
                                 "section_path": section_of(section_document, start),
                                 "tables": _chunk_tables(tables, table_ends, start, end)},
                })
            if end == len(text):
                break
            if strategy == "structured":
                next_start = max(start + 1, end - chunk_overlap)
                span = _containing_span(next_start, structure[1], structure[2])
                start = min(span[1], end) if span else next_start
                previous_end = end
            else:
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
                chunk_size=1000, chunk_overlap=150, documents_sha256=None):
    """호출자가 계산한 documents.json 해시를 재계산·변환 없이 config에 저장합니다."""
    strategy = _chunking_strategy()
    context = _embedding_context()
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("인덱스를 생성할 청크가 없습니다.")
    index = None
    for start in range(0, len(chunks), 32):
        vectors = _document_vectors(chunks[start:start + 32], client, model, context)
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
              "chunking_strategy": strategy, "chunking_version": 2 if strategy == "structured" else 1,
              "embedding_context": context, "embedding_context_version": 1,
              "chunk_metadata_version": 1,
              "documents_sha256": documents_sha256}
    write_json(path / "config.json", config)
    return config


def load_index(index_dir="indexes"):
    path = Path(index_dir)
    index = faiss.read_index(str(path / "index.faiss"))
    chunks, config = read_json(path / "chunks.json"), read_json(path / "config.json")
    if index.ntotal != len(chunks) or index.d != config["dimension"] or len(chunks) != config["chunk_count"]:
        raise ValueError("인덱스와 메타데이터 불일치: build를 다시 실행하세요.")
    return index, chunks, config
