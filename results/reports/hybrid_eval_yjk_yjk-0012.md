# 하이브리드 검색·질문 재작성 비교 (검색 단계)

평가셋은 개인이 CSV 메타데이터로 자동 구성한 문서 단위 정답 325문항이며 팀 공통 평가셋이 아닙니다. 답변 생성과 답변 사실성은 측정하지 않습니다.

| 항목 | 값 |
| --- | --- |
| experiment_id | `yjk-0012` |
| measured_at_utc | `20261006T052939Z` |
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
| bm25_build_ms | `3679` |

## 사업명 있음

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.917 | 0.958 | 0.975 | 0.896 | 0.906 | 0.908 |
| lexical-c100 | 0.992 | 0.992 | 1.000 | 0.983 | 0.983 | 0.985 |
| hybrid | 0.992 | 0.992 | 1.000 | 0.975 | 0.975 | 0.976 |
| hybrid-lexical-c100 | 0.992 | 1.000 | 1.000 | 0.979 | 0.981 | 0.981 |
| hybrid-lexical-c100-v50-b50 | 0.992 | 1.000 | 1.000 | 0.975 | 0.977 | 0.977 |
| hybrid-lexical-c200 | 0.992 | 1.000 | 1.000 | 0.988 | 0.989 | 0.989 |

## 사업명 일부

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.776 | 0.786 | 0.867 | 0.685 | 0.688 | 0.697 |
| lexical-c100 | 0.959 | 0.969 | 0.969 | 0.937 | 0.940 | 0.940 |
| hybrid | 0.949 | 0.959 | 0.980 | 0.891 | 0.894 | 0.897 |
| hybrid-lexical-c100 | 0.974 | 0.980 | 0.990 | 0.964 | 0.964 | 0.966 |
| hybrid-lexical-c100-v50-b50 | 0.974 | 0.990 | 0.990 | 0.964 | 0.966 | 0.966 |
| hybrid-lexical-c200 | 0.974 | 0.980 | 0.990 | 0.964 | 0.964 | 0.966 |

## 사업명 없음

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.727 | 0.754 | 0.813 | 0.682 | 0.686 | 0.692 |
| lexical-c100 | 0.854 | 0.875 | 0.894 | 0.858 | 0.860 | 0.862 |
| hybrid | 0.861 | 0.899 | 0.950 | 0.808 | 0.814 | 0.820 |
| hybrid-lexical-c100 | 0.897 | 0.941 | 0.941 | 0.916 | 0.922 | 0.922 |
| hybrid-lexical-c100-v50-b50 | 0.902 | 0.941 | 0.941 | 0.916 | 0.922 | 0.922 |
| hybrid-lexical-c200 | 0.916 | 0.941 | 0.950 | 0.925 | 0.927 | 0.928 |

## 전체

| 설정 | Recall@3 | Recall@5 | Recall@10 | MRR@3 | MRR@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.812 | 0.839 | 0.889 | 0.762 | 0.768 | 0.773 |
| lexical-c100 | 0.936 | 0.947 | 0.956 | 0.928 | 0.930 | 0.931 |
| hybrid | 0.936 | 0.951 | 0.977 | 0.895 | 0.897 | 0.901 |
| hybrid-lexical-c100 | 0.955 | 0.974 | 0.977 | 0.954 | 0.956 | 0.957 |
| hybrid-lexical-c100-v50-b50 | 0.957 | 0.977 | 0.977 | 0.952 | 0.956 | 0.956 |
| hybrid-lexical-c200 | 0.962 | 0.974 | 0.981 | 0.960 | 0.961 | 0.962 |

## 처리 시간과 적용 옵션

| 설정 | 평균(ms) | 상위 5%(ms) | 적용 옵션 |
| --- | ---: | ---: | --- |
| none | 11.4 | 17.5 | `{}` |
| lexical-c100 | 44.0 | 49.9 | `{'rerank': 'lexical', 'candidates': 100}` |
| hybrid | 20.8 | 23.0 | `{'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |
| hybrid-lexical-c100 | 52.7 | 54.7 | `{'rerank': 'lexical', 'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |
| hybrid-lexical-c100-v50-b50 | 49.3 | 53.2 | `{'rerank': 'lexical', 'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 50, 'hybrid_bm25_k': 50, 'rrf_k': 60}` |
| hybrid-lexical-c200 | 79.7 | 87.6 | `{'rerank': 'lexical', 'candidates': 200, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'rrf_k': 60}` |

처리 시간은 질문 임베딩·재작성 캐시를 쓴 검색 시간입니다 (API 지연·BM25 색인 생성 시간 제외).
