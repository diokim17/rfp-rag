"""김연주: 질문 → 코사인 유사도 검색. 리랭킹은 추후 추가합니다."""

import unicodedata

from embedding import embed_texts
from observability import observed, record_retrieval


@observed("retrieve")
def retrieve(question, client, index, chunks, config, top_k=5, filters=None):
    """반환: Chunk에 score(float)를 추가한 목록. filters는 metadata 정확 일치."""
    if not question.strip() or top_k < 1:
        raise ValueError("비어 있지 않은 질문과 1 이상의 top_k가 필요합니다.")
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
    # 소규모 baseline: 전체 순위를 얻고 필터를 적용하여 top_k를 채웁니다.
    scores, ids = index.search(vector, index.ntotal if filters else min(top_k, index.ntotal))
    hits = []
    for score, i in zip(scores[0], ids[0]):
        if int(i) in eligible:
            hits.append({**chunks[int(i)], "score": float(score)})
            if len(hits) == top_k:
                break
    record_retrieval(hits)
    return hits
