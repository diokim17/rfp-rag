# PR #14 structured 기본값 호환성 비교

작성일: 2026-10-08

## 비교 조건과 결과

| 항목 | 현재 dev 기준 | PR #14 수정 코드 |
| --- | --- | --- |
| 코드 커밋 | `65cb96ff97f3ce9aa1ce65ebc87ee988b2513fed` | 옵션 기본값 수정 `027629a27850072d84ac0f8b62ca42bdbb890a54` |
| 데이터 | `data/processed/dev-shared/documents.json`, SHA-256 `0e93af3a97c4f261ea7304aa8db76b7638ed02eadf021eb6babc279ef7a5842c` | 같은 데이터·해시 |
| 청킹 / 임베딩 문맥 | structured v2 / none | structured 기본값 v2 / none |
| 청크 설정 / 임베딩 모델 | 1000자 / overlap 150 / `text-embedding-3-small` | 동일 설정 |
| 청크 수 / 청크 파일 해시 | 11,770 / `7cfb2375ef86dee3a6e07fbc125938634ace27b63586f0243fd338cde1dce413` | 11,770 / 동일. PR 코드에서 재생성한 청크와 기존 청크가 정확히 일치함을 확인 |
| 평가셋 | `data/eval_v2.json`, 65문항, SHA-256 `3f1f37b5eb5a06ed0827d4368dad73c99a6bd48ab4048334804bd6cd6d2f2b00` | 같은 평가셋 |

PR에서 바뀐 동작은 structured 전략의 암묵적 기본 버전을 v3에서 v2로 되돌린 점입니다. v3를 선택하려면 `RFP_STRUCTURED_CHUNKING_VERSION=3`을 명시해야 합니다. 기존 인덱스는 환경변수 변경만으로 재작성되지 않습니다. 재사용 시 기존 `--index-dir`를 그대로 쓰고, 새 옵션 검증을 위해 build할 때는 별도 인덱스 경로를 사용합니다.

## 검색 지표 참고

공유 인덱스와 같은 청크 해시를 사용한 기존 GPU 검색 측정(`yjk-0025`)의 본 집계 62문항, k=5, hybrid + BM25 prefix + `BAAI/bge-reranker-v2-m3` + 후보 50 + 확장 5 + 문서당 최대 2 결과는 다음과 같습니다.

| 지표 | 기존 측정값 |
| --- | ---: |
| doc_recall@5 | 0.963 |
| 문서 MRR | 0.981 |
| 정답 청크 hit@5 | 0.984 |
| 평균 retrieve 시간 | 194.2 ms |
| 답변 키워드 포함률 / 생성 응답 시간 | 측정 안 함 (검색 전용 평가) |
| 당시 임베딩 API 호출 | 2 (질문 임베딩), 문서 재임베딩 없음 |

이 값은 PR 수정 코드에서 새로 실행한 평가 결과가 아닙니다. PR의 기본 structured 청크가 기존 dev 인덱스와 정확히 같고 검색 구현은 변경하지 않았으므로, 같은 인덱스·질문·검색 설정에서는 같은 검색 결과가 예상됩니다. 이번 호환성 확인은 청크 생성 일치만 검증했으며 cross-encoder가 필요한 전체 검색 재측정은 수행하지 않았습니다. 따라서 새로운 품질 개선을 주장하지 않습니다.

- 평가셋 원본과 결과 상세: [`eval_v2_yjk_yjk-0025.md`](eval_v2_yjk_yjk-0025.md)
- 팀 검색 설정의 요약: [`b_retrieval_settings_yjk.md`](b_retrieval_settings_yjk.md)
- 이번 호환성 검증의 추가 embedding/generation API 호출: 0
- 이번 검증에서 처리 시간은 재측정하지 않음
