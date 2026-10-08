# B 검색 설정·성능표 (김연주, 2026-10-08)

B 시나리오 검색은 **하이브리드(벡터 + BM25) + cross-encoder 재정렬 + 문서당 최대 2청크 + BM25 사업명 접두**로 확정합니다. eval_v2(65문항, k=5)에서 비교한 모든 설정 중 문서·정답 청크 지표가 가장 좋았습니다(yjk-0021). 검색 단계만 측정했고 답변 생성은 포함하지 않습니다.

## 확정 설정 (retrieve 기본값, PR #10)

| 항목 | 값 | 바꾸는 방법 |
| --- | --- | --- |
| 반환 청크 수(k) | 5 (팀 기준) | `--top-k`, `retrieve(top_k=)` |
| 1단계 검색 | 벡터 상위 100 + BM25 상위 100을 RRF(상수 60)로 합침 | `RETRIEVAL_HYBRID`, `RETRIEVAL_HYBRID_VECTOR_K`, `RETRIEVAL_HYBRID_BM25_K` |
| BM25 사업명 접두 | 켬. BM25 색인에만 사업명·발주 기관을 붙이고 반환 본문·score는 그대로 | `RETRIEVAL_BM25_PREFIX=off` |
| 재정렬 | cross-encoder `BAAI/bge-reranker-v2-m3`, RRF 상위 50개 후보 | `RETRIEVAL_RERANK`, `RETRIEVAL_CANDIDATES` |
| 문서당 최대 청크 | 2 (재정렬 뒤 적용) | `RETRIEVAL_MAX_PER_DOC` (`none`이면 해제) |
| 필터 | 정확 일치, 일치 값이 없으면 표기 차이(공백·기호, 서울특별시/서울시)를 무시한 유일 값 | 기본 동작 |
| 문서 안 재선택 | 끔 | `RETRIEVAL_WITHIN_DOC=residual+full` |
| 실행 환경 | GPU 필요(L4에서 측정). torch·transformers는 requirements에 없음 | GPU가 없으면 lexical로 대체되고 성능이 크게 떨어짐(아래 표) |

## 성능표: eval_v2 본 집계 62문항, k=5, 후보 50 (yjk-0021, GPU L4)

| 설정 | doc_recall@5 | 문서 MRR | 정답청크 1위 | 정답청크 5위 내 | 5위 안에 정답청크 없음 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **cross-encoder + 상한 2 + 접두 (확정)** | **0.963** | **0.981** | **0.823** | **0.968** | **2** |
| cross-encoder + 접두, 상한 없음 | 0.948 | 0.981 | 0.823 | 0.968 | 2 |
| cross-encoder + 상한 2, 접두 없음 | 0.947 | 0.965 | 0.694 | 0.823 | 11 |
| lexical + 상한 2 + 접두 (GPU 없을 때 대체) | 0.923 | 0.898 | 0.565 | 0.694 | 19 |
| lexical + 상한 2, 접두 없음 | 0.929 | 0.906 | 0.500 | 0.597 | 25 |
| 재정렬 없음 + 상한 2 + 접두 | 0.949 | 0.955 | 0.403 | 0.548 | 28 |
| 재정렬 없음 + 상한 2, 접두 없음 | 0.945 | 0.938 | 0.419 | 0.548 | 28 |

- 기관명 표기가 다른 3문항(n06~n08)은 모든 설정에서 doc_recall@5 1.000(필터 표기 보정).
- 정답 청크 = 정답 문서의 청크 중 기대 키워드를 포함한 청크. 지표는 상위 5개 안에서만 셈.
- 데이터: documents_sha256 `0e93af3a…`(LF 기준, shn-0035와 동일), structured 11,770청크(chunks `7cfb2375…`), eval_v2 `3f1f37b5…`, text-embedding-3-small. 질문 임베딩은 API로 새로 받음.

## 근거 요약

1. **cross-encoder가 lexical보다 확실히 좋음**: 같은 상한 2 + 접두에서 정답청크 5위 내 0.694 → 0.968, 문서 MRR 0.898 → 0.981.
2. **상한 2는 cross-encoder와 함께일 때 손해 없이 이득**: 정답 청크 지표는 상한 유무와 같고(0.968), doc_recall@5는 0.948 → 0.963. 재정렬이 없을 때는 상한 2가 정답 청크를 줄이므로(yjk-0014, shn-0023~0026) 상한은 cross-encoder와 묶어서 씀.
3. **BM25 접두는 cross-encoder와 함께 효과가 가장 큼**: 정답청크 1위 0.694 → 0.823, 5위 내 0.823 → 0.968.
4. **eval_v1에서도 같은 방향**: 46문항(yjk-0017, 접두·필터 보정 반영 전)에서 하이브리드 + cross-encoder + 상한 2가 doc_recall@5 0.953, 정답청크 1위 0.674로 가장 좋았음(평가셋 BM25 기준값 0.915 / 0.442).
5. **문서 안 재선택은 끔**: 재정렬 없는 하이브리드에서는 정답청크 5위 내 0.629 → 0.806(yjk-0016)이지만, cross-encoder가 이미 0.968이라 기본으로 켤 이유가 없음. cross-encoder와 함께 쓴 경우는 미측정.

## 응답 시간

- cross-encoder 모델 로드 26초(서버 시작 시 한 번).
- 질문당 retrieve() 시간: 재정렬 없음·lexical 26~85ms, cross-encoder 실제 계산 약 0.2~0.7초(L4). yjk-0021의 일부 cross-encoder 설정 시간(27~182ms)은 같은 후보 점수를 캐시로 재사용한 값이라 실제 속도가 아님.
- 질문 임베딩 API 시간은 포함하지 않음.

## 남은 일

- 답변 생성까지 포함한 end-to-end 평가(키워드 포함률, 응답 시간)는 미측정.
- 해시 수정(parsing.py·run.py) 담당자 확인: 나상훈, 김도영.
- 문서 안 재선택은 새 질문으로 검증 후 cross-encoder와 함께 재측정.

| 근거 보고서 | 내용 |
| --- | --- |
| `results/reports/eval_v2_yjk_yjk-0021.md` | cross-encoder 대 lexical, k=5, GPU (서버) |
| `results/reports/eval_v2_yjk_yjk-0014.md`, `-0016.md` | lexical·문서 안 재선택, k=5 (로컬 CPU) |
| `results/reports/team_eval_yjk_yjk-0017.md` | eval_v1 46문항 비교 (서버) |
