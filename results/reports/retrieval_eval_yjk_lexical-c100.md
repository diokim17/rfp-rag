# 검색 단계 비교 요약: lexical-c100

| 설정 | 값 |
| --- | --- |
| label | `lexical-c100` |
| top_k | `5` |
| embedding_model | `text-embedding-3-small` |
| dimension | `1536` |
| chunk_count | `2932` |
| chunk_size | `1000` |
| chunk_overlap | `150` |
| RETRIEVAL_RERANK | `lexical` |
| RETRIEVAL_CANDIDATES | `100` |
| RETRIEVAL_RERANK_MODEL | `None` |
| evaluation_sha256 | `e96b3f66b259559c0de902523c79abdcde3c2287dbfef5e0ea6c512348670bfa` |
| query_embedding_api_calls | `0` |

| 그룹 | 건수 | Recall@k | MRR@k | Top-1 | 문서 정밀도@k | 필터 위반 | 중앙 지연(ms) | 최대 지연(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 325 | 0.948 | 0.935 | 0.923 | 0.625 | 0 | 36.2 | 41.1 |
| A_full_name | 98 | 0.980 | 0.980 | 0.980 | 0.661 | 0 | 36.2 | 41.1 |
| B_short_name | 98 | 0.969 | 0.944 | 0.929 | 0.645 | 0 | 36.1 | 38.6 |
| C_no_name | 97 | 0.887 | 0.858 | 0.835 | 0.456 | 0 | 36.6 | 41.1 |
| D_filter_agency | 10 | 0.917 | 1.000 | 1.000 | 1.000 | 0 | 24.2 | 38.5 |
| E_filter_named | 22 | 1.000 | 1.000 | 1.000 | 0.945 | 0 | 24.8 | 38.8 |

지연은 질문 임베딩 캐시를 사용한 검색·리랭킹 처리 시간이며 임베딩 API 지연은 제외됩니다.
평가셋은 CSV 메타데이터로 자동 구성한 문서 단위 정답이며 청크 단위 적합성과 답변 사실성은 측정하지 않습니다.
