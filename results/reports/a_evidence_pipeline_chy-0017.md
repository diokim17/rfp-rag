# A 기본 검색: 청킹·project blend 근거 검증

실험 ID: chy-0017
실행: 2026-10-19T05:52:58.972532+00:00; commit 28b069d36084c0febd841e16b29bd23610d2275e (미커밋 변경 포함)

100문서 / eval_v4 50문항 / fixed 또는 structured-v3 1000/150 / BGE-M3 / 동일 최신 검색 설정.
기존 인덱스 읽기 전용 재사용. 재파싱·재임베딩·생성·판정 API 호출 없음.

| 조건 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p95 ms | 답변불가 빈 결과율 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-none | 0.976190 | 0.976190 | 0.90 | 1123.26 | 1111.23 | 0.25 |
| fixed-blend20 | 0.976190 | 0.976190 | 0.90 | 1162.29 | 1178.54 | 0.125 |
| structured-v3-none | 1.000000 | 0.988095 | 0.85 | 1044.61 | 1013.08 | 0.125 |

Recall/MRR 분모 42문항, 근거 지표 20문항, 답변불가 8문항. 근거 지표는 값/키워드 대리지표.
반복은 독립 표본을 늘리지 않음. 실행 순서를 반복별 회전함. 시간에는 단계 캡처의 작은 오버헤드 포함.

## 단계별 근거 누락(첫 반복)

### fixed-none

분류: `{"hit": 18, "candidate_miss": 1, "document_cap_loss": 1, "not_scored": 30}`

| 문항 ID | 분류 | RRF 후보 순위 | 확장 후보 순위 | 재정렬 순위 |
| --- | --- | ---: | ---: | ---: |
| yB058 | candidate_miss | None | None | None |
| yC080 | document_cap_loss | 1 | 1 | 3 |

질문 토큰/실패 통계: `{"calls": 150, "texts": 150, "input_tokens": 4224, "truncated_texts": 0, "oom_retries": 0, "failures": 0}`
리랭커 토큰 재계산: `{"pairs": 8640, "input_tokens": 5292258}`

### fixed-blend20

분류: `{"hit": 18, "candidate_miss": 1, "document_cap_loss": 1, "not_scored": 30}`

| 문항 ID | 분류 | RRF 후보 순위 | 확장 후보 순위 | 재정렬 순위 |
| --- | --- | ---: | ---: | ---: |
| yB058 | candidate_miss | None | None | None |
| yC080 | document_cap_loss | 1 | 1 | 3 |

질문 토큰/실패 통계: `{"calls": 150, "texts": 150, "input_tokens": 4224, "truncated_texts": 0, "oom_retries": 0, "failures": 0}`
리랭커 토큰 재계산: `{"pairs": 8805, "input_tokens": 5398509}`

### structured-v3-none

분류: `{"hit": 17, "document_cap_loss": 3, "not_scored": 30}`

| 문항 ID | 분류 | RRF 후보 순위 | 확장 후보 순위 | 재정렬 순위 |
| --- | --- | ---: | ---: | ---: |
| yC002 | document_cap_loss | 3 | 3 | 3 |
| yC080 | document_cap_loss | 1 | 1 | 3 |
| x04 | document_cap_loss | 20 | 20 | 3 |

질문 토큰/실패 통계: `{"calls": 150, "texts": 150, "input_tokens": 4224, "truncated_texts": 0, "oom_retries": 0, "failures": 0}`
리랭커 토큰 재계산: `{"pairs": 8805, "input_tokens": 4384662}`

## 고정 옵션·재현 정보

```json
{
  "options": {
    "rerank": "cross-encoder",
    "candidates": 50,
    "max_per_doc": 2,
    "hybrid": true,
    "hybrid_vector_k": 100,
    "hybrid_bm25_k": 100,
    "bm25_prefix": true,
    "rrf_k": 60,
    "rewrite": "off",
    "rewrite_model": null,
    "within_doc": "off",
    "expand": 5,
    "rerank_section": true,
    "dedup": false,
    "min_score": 0.53,
    "uncap_named": false
  },
  "experiment": {
    "documents_sha256": "97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a",
    "evaluation_sha256": "ded1e0b3be82f4567f2a5f9ad91c8e2329f56cc79cc3715bf2f866a4f2fdbf4a",
    "model_manifest_sha256": "c08690d7741b680b8a382f643588ba1909e6ae6e7cffee7c030afe15f4879611",
    "experiment_id": "chy-0017",
    "owner": "chy",
    "commit": "28b069d36084c0febd841e16b29bd23610d2275e",
    "dirty": true
  },
  "code_hashes": {
    "a_client.py": "cfbe717d833350b49af63bc5b4a3de0f12e86b41f160f565279522657142bc23",
    "a_embedding.py": "af65fd5e65bf25adf9f49eeee9989e65e4d33b04fc960588ef43910beff50694",
    "a_evaluation.py": "32045ef786d0dd49f49d5c78058e48b65a66da06cf11c6eaafa216c40688ff85",
    "a_experiment_ids.py": "291a21e4991babe98389c66226a29338b876c886478f1666c5a5fec0a0c3934a",
    "a_experiment_reports.py": "47ad3b7440e8e0dbfabb64ac02a459b668078d7d3d67ee58c8968e583667f570",
    "a_generation.py": "cfc093478d87e9f7c6d93bd739fc946dc5aa8dce1ef309e0ab7080c5021a3d27",
    "a_local_embedding.py": "60c2e8c04628850e37943d8cb0197f910af2e5c82a7f0dd056ab4415a8d0801c",
    "a_local_generation.py": "e8dfd22d7ba236c0f3a241039af1e1e7b1472fc157682b76695ee99b73876b20",
    "a_observability.py": "7b2408f92face6a98ba797fd10b4f1e80d1bcaa4b86e8815404403f6f6d35c24",
    "a_parsing.py": "746eedb11d5d15d8b11f93e780cb5c9f108eeb65a6bbd024143f97952cd13f8a",
    "a_query_rewrite.py": "8476037b592dda2ce23bd42e68f3287c0b0c8a1a1724d19cc87da736e705f697",
    "a_retrieval.py": "b46f6cb8e08809fbb7e1762f41e0a4161872012f5c7f42c57bd26a5b80448638",
    "a_run.py": "c5e5cbfcbce1cd2e7b75fa3bc594cf72cb9ae6a0fa9d1a4e31a4a16f5ecf8695",
    "experiments/a_chunking_chy.py": "19efd6caaf0b52efdf0d036fc2f7973b189593b6ea0961a5d47907c02e51116f",
    "experiments/a_embedding_chy.py": "10e372551c1248f4f725788ba6a05c5f4ed7e444b402505fede3b5079ab8c3cd",
    "experiments/a_retrieval_metrics.py": "8c83849a097b08ed9fd505370717fbb1bac92139c369c65c65416a7a47c85354",
    "experiments/a_evidence_pipeline_chy.py": "5b01f1b5a0287fc7885c12a94854dd9e6cdb6dfcf456f6413fd8d3db5d1425c9"
  },
  "source_results": [
    "/home/spai1304/rfp-rag/results/a-chunking/chy-0011/evaluation.json",
    "/home/spai1304/rfp-rag/results/a-chunking/chy-0013/evaluation.json",
    "/home/spai1304/rfp-rag/results/a-chunking/chy-0005/evaluation.json"
  ]
}
```

```bash
.venv-a/bin/python experiments/a_evidence_pipeline_chy.py --repeat 3
```

리랭커 토큰은 동일 tokenizer 설정(only_second, 최대 1024)으로 측정 구간 밖에서 다시 계산한 padding 제외 입력 토큰이며 API 과금량이 아님.
기존 추적·ID 생성·공개 계약을 유지. 다른 담당 모듈은 수정하지 않음. 독립 평가셋·답변 생성은 미검증.

## 결론

이번 A 기본 검색에서는 fixed/project_blend 20%가 fixed/none 대비 Recall·MRR·근거 적중률을 개선하지 않았다. 두 조건의 근거 실패 문항도 동일하다. 신규 build에서 두 종류의 문서 임베딩이 필요한 blend를 기본으로 채택할 근거는 얻지 못했다. 기본 설정은 유지한다.

structured-v3는 문서 Recall 1.0으로 높지만 근거 적중률이 0.85로 낮다. 이는 이번 실패 문항에서 청킹이 정답을 잘라서가 아니라, 재정렬 이후 문서당 2개 제한이 필요한 세 번째 청크를 제거한 결과다. structured 전반이 더 좋거나 나쁘다고 단정하지 않는다.

이전 chy-0006/0007의 a-default 프로필은 min_score를 끄고 당시 재정렬 구현을 사용했다. 이번에는 최신 코드의 section 접두 포함, min_score=0.53을 명시했다. 과거 점수와 이번 점수의 차이를 한 옵션의 효과로 귀속하지 않는다. 비교 결론은 이번 실행의 세 조건 사이에 한정한다.

## 원문 근거 수동 확인

원문 본문·질문·사업명은 이 보고서에 싣지 않는다. 로컬 `manual_evidence_review.json`에 문서 ID, 문자 좌표, 해당 근거를 완전히 포함하는 청크 ID를 저장했다. 평가 정답·파싱 원문·기존 인덱스는 수정하지 않았다.

| 문항 | 원문에서 확인한 근거 | 확인 결과 |
| --- | --- | --- |
| yB058 | 사업 개요의 사업기간 항목 | 세 조건 모두 근거 보존. fixed 두 조건에서는 RRF 50개와 확장 후 후보에 해당 청크가 없음. structured에서는 최종 근거 확보 |
| yC080 | 사업 개요의 소요예산 항목 | 세 조건 모두 근거 보존. 후보 1위 → 재정렬 3위, 같은 문서의 세 번째 결과여서 상한 2에 의해 탈락 |
| yC002 | 사업기간 항목 및 다른 절의 동일 기간 안내 | 세 조건 모두 근거 보존. structured에서는 후보/재정렬 3위의 근거가 상한으로 탈락. 기간과 용지 규격의 숫자 일치를 구분해 원문 확인 |
| x04 | 계약조건의 지체상금 항목 | 세 조건 모두 근거 보존. structured 후보 20위 → 재정렬 3위, 상한으로 탈락 |

이 네 문항은 단순 키워드 대리지표의 오탐으로만 설명되지 않으며, 원문에서 확인한 실제 근거 청크도 동일 단계에서 탈락했다. 다른 모든 문항의 정답 정확성을 전수 수동 검증한 것은 아니다.

문서 Recall 차이는 yC067 한 문항에서 발생했다. 정답 문서는 fixed/none과 blend20 모두 후보 4위에 있었으나 재정렬 후 첫 정답 문서가 각각 7위·6위로 밀렸다. structured는 후보 10위 → 재정렬 2위로 최종 결과에 남았다. 이 문항은 여러 사업에서 유사하게 쓰는 목적 표현을 포함하며 근거 대리지표 미적용(LLM 채점 대상)이다. 문서 Recall 차이를 답변 정확도 차이로 해석하지 않는다.

## 문서당 상한의 추가 분석 — 저장된 순위 재계산

첫 반복에서 이미 측정한 동일 후보·재정렬 순위를 그대로 사용해 최종 선택만 바꿨다. 새 모델 추론·독립 실험·실행 시간 측정이 아니다. 상한 2 재계산 결과는 실제 검색 결과와 완전히 일치했다.

| 조건 | 상한 2 Recall / 근거 | 상한 3 Recall / 근거 | 상한 없음 Recall / 근거 |
| --- | --- | --- | --- |
| fixed-none | 0.976190 / 0.90 | 0.960317 / 0.90 | 0.960317 / 0.90 |
| fixed-blend20 | 0.976190 / 0.90 | 0.960317 / 0.90 | 0.960317 / 0.90 |
| structured-v3-none | 1.000000 / 0.85 | 0.992063 / 1.00 | 0.992063 / 1.00 |

- fixed의 상한 완화는 yC080 근거를 회복하지만 h10의 근거를 잃어 전체 근거 적중률이 그대로다. yD008·h10의 문서 Recall도 각각 1→2/3으로 하락한다.
- structured의 상한 완화는 근거 실패 3문항을 회복하지만 yD008의 문서 Recall이 1→2/3으로 하락한다.
- 따라서 상한을 일괄 해제하지 않는다. 단일 사업 질문과 여러 사업을 찾는 질문을 구분한 최종 청크 선택 검증이 후속 후보다. 검색 담당 모듈 변경은 담당자와 협의해야 한다.
- fixed의 yB058은 후보 자체에 근거가 없어 상한 완화로 해결되지 않는다. 후보 확장 단계가 이 청크를 포함하지 못한 원인 확인이 별도로 필요하다.

## 시간·토큰·실패 및 한계

- 3조건 × 50문항 × 3반복 = 450건. 워밍업 3건 별도. 세 반복의 문항별 Hit ID·점수·지표가 모두 일치했다.
- 질문 임베딩: 조건당 150건/4,224토큰, 워밍업 포함 전체 453건/12,780토큰. 신규 문서 임베딩 0, API 호출 0. 임베딩 실패·잘림·OOM 재시도 0, 검색 실패 0.
- 리랭커 입력 토큰(워밍업 제외): fixed-none 5,292,258 / blend20 5,398,509 / structured 4,384,662. 동일 tokenizer의 1024토큰 제한 이후 padding 제외 길이를 재계산했다. 리랭커 입력 잘림 건수는 별도로 측정하지 않았으므로 '리랭커 잘림 0'을 주장하지 않는다.
- 평균/중앙값/p95(ms): fixed 1123.26/994.68/1111.23, blend20 1162.29/1012.61/1178.54, structured 1044.61/885.28/1013.08.
- yA052에서 매 반복 약 8.85~9.50초의 지연이 있어 평균이 p95보다 높다. 해당 지연의 내부 원인은 이번 계측으로 분리하지 않았다. 공유 GPU·추적·계측 영향을 포함하므로 인과적인 속도 개선을 주장하지 않는다.
- 답변불가 8문항 중 빈 검색 결과: fixed 2/8(25%), blend20 및 structured 각각 1/8(12.5%). 생성·판정을 하지 않았으므로 거절 정확도나 환각률이 아니다. 모델/청킹 변경 시 고정 임계값 0.53의 적합성은 별도 검증 대상이다.
- 평가셋은 기존 탐색에 사용한 eval_v4다. 독립 평가셋, 실제 생성 답변, 모든 문항의 근거 좌표를 이용한 채점은 미검증이다.

## 변경 범위·검증·재실행

- 이번 추가 파일: `experiments/a_evidence_pipeline_chy.py`, `tests/test_a_evidence_pipeline.py`, 본 보고서. 이전 작업의 a_embedding.py 비율 옵션과 개인 파일은 보존했다.
- 단계 캡처는 실험 프로세스 안에서 원래 `_expand_candidates`/`_rerank`를 그대로 호출하여 입력/출력 순위를 복사한다. 검색 점수·순서·옵션을 변경하지 않으며 종료/예외 시 원래 함수로 복원한다.
- 공통 모듈·run.py 기본값·공개 함수·필수 키·Document → Chunk → Hit → Answer·JSON/FAISS 저장 계약·Langfuse 본문 제외 정책 유지. 새 패키지 설치 및 커밋·푸시·병합 없음.
- 새 테스트 7개 포함 전체 544개 통과. pip check, git diff --check 통과.
- 저장 결과 450건 및 단계 진단 150건을 원본 청크로 재계산하여 일치 확인. 후보 수(최대 50+10), 재정렬 전후 집합, 문서당 상한, 최종 결과 선택도 검사했다.
- 실행 전후 기존 인덱스·모델·리랭커 파일 및 실행 코드 해시 확인. 원본 입력과 별도 processed 사본 해시 일치.
- 재파싱·인덱스 재생성 불필요. 기존 세 인덱스와 manifest, 결과 JSON, 로컬 모델, 이전 개인 실험 도구 a_chunking_chy.py가 필요하다.

```bash
# 동일 평가 재실행: 새 자동 ID 및 개인 경로 생성, 로컬 GPU 사용
.venv-a/bin/python experiments/a_evidence_pipeline_chy.py --repeat 3

# 저장된 이번 결과 재검증: 모델 추론 없음
PYTHONPATH=. .venv-a/bin/python results/a-evidence-pipeline/chy-0017/verify_saved.py
PYTHONPATH=. .venv-a/bin/python results/a-evidence-pipeline/chy-0017/audit_cap_saved.py

.venv-a/bin/python -m unittest discover -s tests -v
```
