# 검색 평가 지표 (Recall@k, MRR@k): eval_sample_v0

| 설정 | 값 |
| --- | --- |
| source | `rerank_grid_20261006T043614Z.json` |
| eval_file | `eval_sample_v0.json` |
| cases | `14` |
| evaluation_sha256 | `62a6d2540e1bf57acda9defaf1f135041b9a7ca40f3ece995cfdf8b40a6a12a7` |
| index_dir | `indexes/parsing-v2-yjk` |
| chunk_count | `9789` |
| embedding_model | `text-embedding-3-small` |
| measured_at_utc | `20261006T043614Z` |

| 리랭킹 | 후보 수 | top-k | Recall@k | MRR@k | 질문당 평균(ms) | 정답 누락 질문 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| none | - | 3 | 0.976 | 0.929 | 11.5 | f09 |
| none | - | 5 | 0.976 | 0.929 | 11.1 | f09 |
| none | - | 10 | 1.000 | 0.929 | 11.1 | - |
| lexical | 30 | 3 | 0.952 | 1.000 | 21.4 | f09 |
| lexical | 30 | 5 | 0.976 | 1.000 | 21.1 | f09 |
| lexical | 30 | 10 | 1.000 | 1.000 | 21.2 | - |
| lexical | 50 | 3 | 0.976 | 1.000 | 28.0 | f09 |
| lexical | 50 | 5 | 0.976 | 1.000 | 27.7 | f09 |
| lexical | 50 | 10 | 1.000 | 1.000 | 27.6 | - |
| lexical | 70 | 3 | 0.952 | 1.000 | 34.5 | f09 |
| lexical | 70 | 5 | 0.976 | 1.000 | 35.1 | f09 |
| lexical | 70 | 10 | 1.000 | 1.000 | 34.4 | - |
| lexical | 100 | 3 | 0.976 | 1.000 | 44.9 | f09 |
| lexical | 100 | 5 | 0.976 | 1.000 | 44.8 | f09 |
| lexical | 100 | 10 | 0.976 | 1.000 | 44.0 | f09 |
| cross-encoder | 30 | 3 | 0.976 | 1.000 | 472.9 | f09 |
| cross-encoder | 30 | 5 | 1.000 | 1.000 | 473.7 | - |
| cross-encoder | 30 | 10 | 1.000 | 1.000 | 478.9 | - |
| cross-encoder | 50 | 3 | 0.976 | 1.000 | 808.7 | f09 |
| cross-encoder | 50 | 5 | 0.976 | 1.000 | 822.7 | f09 |
| cross-encoder | 50 | 10 | 1.000 | 1.000 | 838.6 | - |
| cross-encoder | 70 | 3 | 0.976 | 1.000 | 1183.2 | f09 |
| cross-encoder | 70 | 5 | 0.976 | 1.000 | 1173.3 | f09 |
| cross-encoder | 70 | 10 | 1.000 | 1.000 | 1163.5 | - |
| cross-encoder | 100 | 3 | 0.976 | 1.000 | 1642.7 | f09 |
| cross-encoder | 100 | 5 | 0.976 | 1.000 | 1648.4 | f09 |
| cross-encoder | 100 | 10 | 1.000 | 1.000 | 1661.8 | - |

Recall@k = 상위 k개에 나온 정답 문서 수 / 정답 문서 수. MRR@k = 상위 k개에서 처음 나온 정답 문서 순위의 역수(없으면 0). 문서 단위 정답 기준입니다.
처리 시간은 원래 실험에서 잰 값(질문 임베딩 캐시 사용, 모델 로드 제외)입니다.
