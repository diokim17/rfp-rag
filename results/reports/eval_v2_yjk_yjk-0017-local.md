# 문서 안 재선택 검증: 새 질문 19개(eval_withindoc_yjk, eval_v2 정답 문서 제외), k=5

로컬 PC 실행 번호입니다. 서버의 yjk-0017(team_eval_yjk_yjk-0017.md, eval_v1)과 다른 실험입니다. 평가셋 sha256 `fa99497b…`(19문항: 발표 10, 하자 4, 금액·기간 5).

검색 단계만 측정했습니다(답변 생성 없음). BM25 열은 평가셋에 들어 있는 bm25_reference 값이며 어느 인덱스·청킹으로 계산했는지는 평가셋에 적혀 있지 않습니다.

| 항목 | 값 |
| --- | --- |
| experiment_id | `yjk-0017` |
| measured_at_utc | `20261008T045931Z` |
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
| cross_encoder_load_ms | `0` |
| embedding_api_calls | `45` |
| fresh_query_embeddings | `True` |
| rerank_device | `None` |

지표: doc_recall@5 평균 / 정답 문서 첫 순위의 1위 비율·5위 이내 비율·MRR / 정답 청크(정답 문서 + 기대 키워드 포함) 첫 순위의 1위 비율·5위 이내 비율·MRR. 순위는 상위 5개 안에서만 세며, 없으면 MRR 0으로 계산합니다.

## 보통 (19문항)

| 설정 | doc_recall@5 | 문서 1위 | 문서 5위 내 | 문서 MRR | 정답청크 1위 | 정답청크 5위 내 | 정답청크 MRR | 정답청크 없음 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix-hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 0.263 | 0.421 | 0.318 | 11 |
| prefix-hybrid+wd | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.842 | 0.546 | 3 |
| prefix-hybrid+wdr | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.789 | 0.539 | 4 |
| prefix-hybrid-cap2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.263 | 0.316 | 0.289 | 13 |
| prefix-hybrid-cap2+wd | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.579 | 0.474 | 8 |
| prefix-hybrid-cap2+wdr | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.579 | 0.474 | 8 |
| prefix-hybrid-lex-c50-cap2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.421 | 0.579 | 0.500 | 8 |
| prefix-hybrid-lex-c50-cap2+wd | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.579 | 0.465 | 8 |

## 본 집계 (19문항)

| 설정 | doc_recall@5 | 문서 1위 | 문서 5위 내 | 문서 MRR | 정답청크 1위 | 정답청크 5위 내 | 정답청크 MRR | 정답청크 없음 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix-hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 0.263 | 0.421 | 0.318 | 11 |
| prefix-hybrid+wd | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.842 | 0.546 | 3 |
| prefix-hybrid+wdr | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.789 | 0.539 | 4 |
| prefix-hybrid-cap2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.263 | 0.316 | 0.289 | 13 |
| prefix-hybrid-cap2+wd | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.579 | 0.474 | 8 |
| prefix-hybrid-cap2+wdr | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.579 | 0.474 | 8 |
| prefix-hybrid-lex-c50-cap2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.421 | 0.579 | 0.500 | 8 |
| prefix-hybrid-lex-c50-cap2+wd | 1.000 | 1.000 | 1.000 | 1.000 | 0.368 | 0.579 | 0.465 | 8 |

## 문항별 (doc_recall@5 / 정답 문서 첫 순위 / 정답 청크 첫 순위)

| id | 난이도 | 유형 | prefix-hybrid | prefix-hybrid+wd | prefix-hybrid+wdr | prefix-hybrid-cap2 | prefix-hybrid-cap2+wd | prefix-hybrid-cap2+wdr | prefix-hybrid-lex-c50-cap2 | prefix-hybrid-lex-c50-cap2+wd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w01 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 4 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / - |
| w02 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / 3 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - |
| w03 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / 2 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / 2 | 1.00 / 1 / - | 1.00 / 1 / 2 |
| w04 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / 4 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 2 |
| w05 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 5 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / - |
| w06 | 보통 | 본문-발표 | 1.00 / 1 / 3 | 1.00 / 1 / 3 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / - |
| w07 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 3 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / - |
| w08 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - |
| w09 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 4 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / - |
| w10 | 보통 | 본문-발표 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 2 | 1.00 / 1 / 1 |
| w11 | 보통 | 본문-하자 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / 2 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / 2 | 1.00 / 1 / - | 1.00 / 1 / 3 |
| w12 | 보통 | 본문-하자 | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - | 1.00 / 1 / - |
| w13 | 보통 | 본문-하자 | 1.00 / 1 / 5 | 1.00 / 1 / 2 | 1.00 / 1 / 2 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / 2 | 1.00 / 1 / 2 | 1.00 / 1 / 2 |
| w14 | 보통 | 본문-하자 | 1.00 / 1 / 2 | 1.00 / 1 / 1 | 1.00 / 1 / 3 | 1.00 / 1 / 2 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / 2 | 1.00 / 1 / 1 |
| w15 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w16 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w17 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 2 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 2 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w18 | 보통 | 사실-기간 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 1 |
| w19 | 보통 | 사실-금액 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / 3 | 1.00 / 1 / 1 | 1.00 / 1 / 1 | 1.00 / 1 / - | 1.00 / 1 / 1 | 1.00 / 1 / 1 |

## 처리 시간과 적용 옵션

| 설정 | 평균(ms) | 상위 5%(ms) | 적용 옵션 |
| --- | ---: | ---: | --- |
| prefix-hybrid | 249.3 | 4281.1 | `{'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60}` |
| prefix-hybrid+wd | 504.1 | 6134.1 | `{'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'within_doc': 'residual+full'}` |
| prefix-hybrid+wdr | 21.6 | 27.2 | `{'candidates': 100, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'within_doc': 'residual'}` |
| prefix-hybrid-cap2 | 19.3 | 23.1 | `{'candidates': 100, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60}` |
| prefix-hybrid-cap2+wd | 177.1 | 316.6 | `{'candidates': 100, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'within_doc': 'residual+full'}` |
| prefix-hybrid-cap2+wdr | 23.2 | 28.0 | `{'candidates': 100, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'within_doc': 'residual'}` |
| prefix-hybrid-lex-c50-cap2 | 28.4 | 34.5 | `{'rerank': 'lexical', 'candidates': 50, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60}` |
| prefix-hybrid-lex-c50-cap2+wd | 104.0 | 213.1 | `{'rerank': 'lexical', 'candidates': 50, 'max_per_doc': 2, 'hybrid': True, 'hybrid_vector_k': 100, 'hybrid_bm25_k': 100, 'bm25_prefix': True, 'rrf_k': 60, 'within_doc': 'residual+full'}` |

처리 시간은 retrieve() 한 번의 시간입니다. 질문 임베딩은 캐시를 썼고 BM25 색인 생성과 cross-encoder 모델 로드 시간은 제외했습니다.
