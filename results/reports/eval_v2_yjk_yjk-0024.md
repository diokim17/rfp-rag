# eval_v2 검색 지표: 하이브리드·리랭킹·BM25 메타 접두·필터 정규화

검색 단계만 측정했습니다(답변 생성 없음). BM25 열은 평가셋에 들어 있는 bm25_reference 값이며 어느 인덱스·청킹으로 계산했는지는 평가셋에 적혀 있지 않습니다.

| 항목 | 값 |
| --- | --- |
| experiment_id | `yjk-0024` |
| measured_at_utc | `20261008T055321Z` |
| eval_file | `data/eval_withindoc_yjk.json` |
| evaluation_sha256 | `fa99497bf89eebebf4d699a2203d1f2069c94c0c127e928c438cefcdbaca31f3` |
| cases | `19` |
| index_dir | `indexes/eval-v2-structured-yjk` |
| chunks_sha256 | `7cfb2375ef86dee3a6e07fbc125938634ace27b63586f0243fd338cde1dce413` |
| chunk_count | `11770` |
| chunking_strategy | `structured` |
| embedding_context | `none` |
| embedding_model | `text-embedding-3-small` |
| top_k | `5` |
| cross_encoder_model | `BAAI/bge-reranker-v2-m3` |
| cross_encoder_load_ms | `16348` |
| embedding_api_calls | `1` |
| fresh_query_embeddings | `True` |
| rerank_device | `cuda` |

지표: doc_recall@5 평균 / 정답 문서 첫 순위의 1위 비율·5위 이내 비율·MRR / 정답 청크(정답 문서 + 기대 키워드 포함) 첫 순위의 1위 비율·5위 이내 비율·MRR. 순위는 상위 5개 안에서만 세며, 없으면 MRR 0으로 계산합니다.

## 보통 (19문항)

| 설정 | doc_recall@5 | 문서 1위 | 문서 5위 내 | 문서 MRR | 정답청크 1위 | 정답청크 5위 내 | 정답청크 MRR | 정답청크 없음 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix-hybrid-ce-c50-cap2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.737 | 0.737 | 0.737 | 5 |
| prefix-hybrid-ce-c50-cap2+x5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.947 | 0.947 | 1 |
| prefix-hybrid-ce-c50-cap2+x10 | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.947 | 0.947 | 1 |
| prefix-hybrid-ce-c50-cap2+x20 | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.947 | 0.947 | 1 |

## 본 집계 (19문항)

| 설정 | doc_recall@5 | 문서 1위 | 문서 5위 내 | 문서 MRR | 정답청크 1위 | 정답청크 5위 내 | 정답청크 MRR | 정답청크 없음 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix-hybrid-ce-c50-cap2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.737 | 0.737 | 0.737 | 5 |
| prefix-hybrid-ce-c50-cap2+x5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.947 | 0.947 | 1 |
| prefix-hybrid-ce-c50-cap2+x10 | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.947 | 0.947 | 1 |
| prefix-hybrid-ce-c50-cap2+x20 | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.947 | 0.947 | 1 |

## 문항별 (doc_recall@5 / 정답 문서 첫 순위 / 정답 청크 첫 순위)

| id | 난이도 | 유형 | prefix-hybrid-ce-c50-cap2 | prefix-hybrid-ce-c50-cap2+x5 | prefix-hybrid-ce-c50-cap2+x10 | prefix-hybrid-ce-c50-cap2+x20 |
| --- | --- | --- | --- | --- | --- | --- |
| w01 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w02 | 보통 | 본문-발표 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w03 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w04 | 보통 | 본문-발표 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w05 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w06 | 보통 | 본문-발표 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w07 | 보통 | 본문-발표 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w08 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w09 | 보통 | 본문-발표 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w10 | 보통 | 본문-발표 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w11 | 보통 | 본문-하자 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w12 | 보통 | 본문-하자 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - |
| w13 | 보통 | 본문-하자 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w14 | 보통 | 본문-하자 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w15 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w16 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w17 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w18 | 보통 | 사실-기간 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w19 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |

## 처리 시간과 적용 옵션

| 설정 | 평균(ms) | 상위 5%(ms) | 적용 옵션 |
| --- | ---: | ---: | --- |
| prefix-hybrid-ce-c50-cap2 | 240.7 | 3938.3 | `{'rerank': 'cross-encoder', 'candidates': 50, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60}` |
| prefix-hybrid-ce-c50-cap2+x5 | 334.8 | 3807.0 | `{'rerank': 'cross-encoder', 'candidates': 50, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'expand': 5}` |
| prefix-hybrid-ce-c50-cap2+x10 | 144.8 | 162.0 | `{'rerank': 'cross-encoder', 'candidates': 50, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'expand': 10}` |
| prefix-hybrid-ce-c50-cap2+x20 | 281.9 | 325.1 | `{'rerank': 'cross-encoder', 'candidates': 50, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'expand': 20}` |

처리 시간은 retrieve() 한 번의 시간입니다. 질문 임베딩은 캐시를 썼고 BM25 색인 생성과 cross-encoder 모델 로드 시간은 제외했습니다.
