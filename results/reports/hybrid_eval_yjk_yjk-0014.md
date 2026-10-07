# 하이브리드 검색·질문 재작성 비교 (검색 단계)

평가셋은 개인이 CSV 메타데이터로 자동 구성한 문서 단위 정답 325문항이며 팀 공통 평가셋이 아닙니다. 답변 생성과 답변 사실성은 측정하지 않습니다.

| 항목 | 값 |
| --- | --- |
| experiment_id | `yjk-0014` |
| measured_at_utc | `20261006T065239Z` |
| eval_file | `eval_retrieval_yjk.json` |
| evaluation_sha256 | `e96b3f66b259559c0de902523c79abdcde3c2287dbfef5e0ea6c512348670bfa` |
| cases | `325` |
| index_dir | `indexes/parsing-v2-yjk` |
| chunks_file | `chunks.json` |
| chunks_sha256 | `8a8a4ba04af2fbaebe02663ef26d8164c4eb25ac46b3a7ad6ed2e881e181f474` |
| index_sha256 | `1c1815f6afde5f622d8558da2dab4348306b8a458adcfc08c7e29957a627e496` |
| chunk_count | `9789` |
| embedding_model | `text-embedding-3-small` |
| ks | `[3, 5, 10]` |
| bm25_build_ms | `3701` |
| cross_encoder_model | `BAAI/bge-reranker-v2-m3` |
| cross_encoder_load_ms | `9158` |

## 사업명 있음

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.917 | 0.958 | 0.975 | 0.896 | 0.906 | 0.908 |
| lexical-c100 | 0.992 | 0.992 | 1.000 | 0.983 | 0.983 | 0.985 |
| hybrid-lexical-c100 | 0.992 | 1.000 | 1.000 | 0.979 | 0.981 | 0.981 |
| hybrid-lexical-c100-cap2 | 0.992 | 1.000 | 1.000 | 0.979 | 0.981 | 0.981 |
| hybrid-ce-c30 | 1.000 | 1.000 | 1.000 | 0.990 | 0.990 | 0.990 |
| hybrid-ce-c50 | 0.992 | 1.000 | 1.000 | 0.988 | 0.990 | 0.990 |
| hybrid-ce-c100 | 0.992 | 1.000 | 1.000 | 0.988 | 0.990 | 0.990 |

## 사업명 일부

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.776 | 0.786 | 0.867 | 0.685 | 0.688 | 0.697 |
| lexical-c100 | 0.959 | 0.969 | 0.969 | 0.937 | 0.940 | 0.940 |
| hybrid-lexical-c100 | 0.974 | 0.980 | 0.990 | 0.964 | 0.964 | 0.966 |
| hybrid-lexical-c100-cap2 | 0.985 | 0.990 | 0.990 | 0.968 | 0.968 | 0.968 |
| hybrid-ce-c30 | 0.990 | 0.990 | 0.990 | 0.980 | 0.980 | 0.980 |
| hybrid-ce-c50 | 0.990 | 0.990 | 0.990 | 0.985 | 0.985 | 0.985 |
| hybrid-ce-c100 | 0.990 | 0.990 | 0.990 | 0.985 | 0.985 | 0.985 |

## 사업명 없음

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.727 | 0.754 | 0.813 | 0.682 | 0.686 | 0.692 |
| lexical-c100 | 0.854 | 0.875 | 0.894 | 0.858 | 0.860 | 0.862 |
| hybrid-lexical-c100 | 0.897 | 0.941 | 0.941 | 0.916 | 0.922 | 0.922 |
| hybrid-lexical-c100-cap2 | 0.910 | 0.944 | 0.944 | 0.916 | 0.922 | 0.922 |
| hybrid-ce-c30 | 0.928 | 0.938 | 0.941 | 0.922 | 0.922 | 0.922 |
| hybrid-ce-c50 | 0.947 | 0.952 | 0.960 | 0.933 | 0.933 | 0.933 |
| hybrid-ce-c100 | 0.933 | 0.952 | 0.960 | 0.925 | 0.927 | 0.927 |

## 전체

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.812 | 0.839 | 0.889 | 0.762 | 0.768 | 0.773 |
| lexical-c100 | 0.936 | 0.947 | 0.956 | 0.928 | 0.930 | 0.931 |
| hybrid-lexical-c100 | 0.955 | 0.974 | 0.977 | 0.954 | 0.956 | 0.957 |
| hybrid-lexical-c100-cap2 | 0.963 | 0.978 | 0.978 | 0.955 | 0.957 | 0.957 |
| hybrid-ce-c30 | 0.973 | 0.976 | 0.977 | 0.965 | 0.965 | 0.965 |
| hybrid-ce-c50 | 0.976 | 0.981 | 0.984 | 0.969 | 0.969 | 0.969 |
| hybrid-ce-c100 | 0.972 | 0.981 | 0.984 | 0.966 | 0.968 | 0.968 |

## 처리 시간과 적용 옵션

| 설정 | 평균(ms) | 상위 5%(ms) | 적용 옵션 |
| --- | ---: | ---: | --- |
| none | 11.5 | 17.9 | `{}` |
| lexical-c100 | 43.8 | 49.9 | `{'rerank': 'lexical', 'candidates': 100}` |
| hybrid-lexical-c100 | 53.5 | 55.7 | `{'rerank': 'lexical', 'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |
| hybrid-lexical-c100-cap2 | 53.7 | 55.8 | `{'rerank': 'lexical', 'candidates': 100, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |
| hybrid-ce-c30 | 486.3 | 553.1 | `{'rerank': 'cross-encoder', 'candidates': 30, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |
| hybrid-ce-c50 | 806.9 | 896.8 | `{'rerank': 'cross-encoder', 'candidates': 50, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |
| hybrid-ce-c100 | 1607.6 | 1753.9 | `{'rerank': 'cross-encoder', 'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |

처리 시간은 retrieve() 한 번의 시간으로 벡터 검색·필터·BM25·RRF·리랭킹·문서당 상한을 포함합니다. 질문 임베딩·재작성은 캐시를 썼으므로 API 지연은 빠져 있고, BM25 색인 생성과 cross-encoder 모델 로드 시간도 제외했습니다.
