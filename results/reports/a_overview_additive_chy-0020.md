# fixed 청크 유지 및 개요 추가 실험

실험 ID: chy-0020; 2026-10-19T07:52:36.080511+00:00

변경: overview-additive v1. fixed/overview-stable 대비 임베딩 문맥 none, 최신 A 검색 옵션 동일.
규칙: 다음 절까지 1000자 이내, 사업명/기간/예산/계약 방식 중 2종류 이상. 개요 인식 규칙은 v1과 동일. 모든 fixed 청크를 유지하고 개요 청크만 추가. 동일 좌표만 중복 제거. 원본 fixed 벡터 재사용.
질문·정답·문서 ID를 청킹 규칙에 사용하지 않음. 기존 기본값·공개 함수·원문 좌표·3파일 인덱스 형식 유지.

| 평가 | 조건 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| existing | fixed-none | 0.976190 | 0.976190 | 0.900 | 1095.52 | 1056.80 |
| existing | overview-stable-none | 0.976190 | 0.976190 | 0.950 | 1140.81 | 1103.30 |
| existing | overview-additive-none | 0.976190 | 0.976190 | 0.950 | 1121.90 | 1103.56 |
| new | fixed-none | 1.000000 | 0.916667 | 1.000 | 1063.56 | 1131.00 |
| new | overview-stable-none | 1.000000 | 0.916667 | 1.000 | 1035.89 | 1119.09 |
| new | overview-additive-none | 1.000000 | 0.916667 | 1.000 | 1052.48 | 1141.02 |

기존 eval_v4 50문항(답변 가능 42, 근거 대리지표 20)과 새 질문 6문항은 별도 집계.
추가 6문항은 CHY-0018에서 작성한 개요 기간/예산 질문을 그대로 재사용했다. 이미 사용한 목표 검증셋이며 독립 평가셋이 아니다.
새 질문은 원문 근거 좌표로 전체 항목이 최종 청크에 포함되는지도 검사. 답변 생성·판정 미실행.

선택 문서 46, 구간 48, 후보 청크 10266.
실제 후보 build 5.61초, 로컬 임베딩 통계 `{"calls": 2, "texts": 48, "input_tokens": 7504, "truncated_texts": 0, "oom_retries": 0, "failures": 0}`.

## 재현

```bash
.venv-a/bin/python experiments/a_overview_additive_chy.py --new-cases data/processed/a-overview-additive/chy-0020/new_cases.json --repeat 1
```

```json
{
  "experiment": {
    "experiment_id": "chy-0020",
    "owner": "chy",
    "commit": "28b069d36084c0febd841e16b29bd23610d2275e",
    "dirty": true,
    "documents_sha256": "97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a",
    "evaluation_sha256": "ded1e0b3be82f4567f2a5f9ad91c8e2329f56cc79cc3715bf2f866a4f2fdbf4a",
    "new_cases_sha256": "a6f90a789b1e34935d39fc6779a611f186777ca01e90a797c2fdef0f47ef2981"
  },
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
  "code_hashes": {
    "a_client.py": "cfbe717d833350b49af63bc5b4a3de0f12e86b41f160f565279522657142bc23",
    "a_embedding.py": "8fc8fc682fbb00881a14d8586311474bfde84d9a1cd3cbdd64a8f22cb123a702",
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
    "experiments/a_overview_additive_chy.py": "07c23e40b3fe0e07ee8421b65d6f2f3712ac1e3276c67b7789a0cfcfbb3ea7eb",
    "experiments/a_evidence_pipeline_chy.py": "5b01f1b5a0287fc7885c12a94854dd9e6cdb6dfcf456f6413fd8d3db5d1425c9"
  }
}
```

## 판단 및 주요 사례

현재 평가셋과 검색 설정에서는 **overview_additive를 최종 청킹 권장 후보**로 선택한다. fixed 전체 원문 청크와 벡터를 유지하면서 근거 적중이 18/20(90%)에서 19/20(95%)로 개선됐다. overview-stable과 Recall/MRR/근거 적중은 문항별로 모두 같다. 실제 기본값과 환경 파일은 변경하지 않았으며 최종 설정 제안은 별도 문서에 정리했다.

| 사례 | fixed | stable | additive |
| --- | --- | --- | --- |
| yB058 | 근거 후보 누락 | 근거 최종 1위 | 근거 최종 1위 |
| h10 | 근거 최종 5위 | 근거 최종 5위 | 근거 최종 5위 |
| yC080 | 문서당 제한으로 제외 | 문서당 제한으로 제외 | 문서당 제한으로 제외 |
| yC067 | 정답 문서 누락 | 정답 문서 누락 | 정답 문서 누락 |
| u09(답변불가) | 결과 0개 | 결과 5개 | 결과 0개 |

fixed와 비교해 문항별 Recall/MRR 하락은 없고, 근거 적중 변화는 yB058의 개선 한 건이다. CHY-0018 overview의 yC067 문서 포함 효과는 재현하지 못했다. 답변불가 8문항의 빈 결과 비율은 fixed/additive 0.25, stable 0.125지만 이를 거절 정확도나 환각률로 해석하지 않는다.

이번 실험은 각 조건 1회로 총 168회 검색이다. 반복 안정성·속도 우열·독립 평가 일반화·실제 답변 정확도는 검증하지 않았다. 근거 지표는 값·키워드 대리지표이며 동일 평가셋을 반복 사용했다. 추가 6문항은 모든 조건에서 Recall 1.0, MRR 0.916667, 근거 적중과 전체 근거 좌표 포함률 1.0으로 차이가 없다.

## 재사용·계약·검증

- 기존 fixed 10,218개 청크의 본문·좌표·메타데이터와 벡터를 모두 보존했다. 새 개요 48개를 추가해 총 10,266개(+0.47%). 청크 ID 번호 대응은 달라질 수 있다.
- 실제 신규 문서 임베딩은 48텍스트, 2배치, 7,504입력 토큰. 잘림·OOM 재시도·실패 0. 5.61초는 기존 벡터 캐시를 재사용한 구축 시간이며 전체 재임베딩 시간과 직접 비교하지 않는다.
- 모델·문서 해시 및 모델명, fixed 1000/150, embedding_context none 일치를 확인한 뒤 재사용했다. 저장된 10,218개 재사용 벡터는 원본과 원소 단위 완전 일치했다.
- additive 개요는 fixed 청크와 겹치거나 그 안에 포함될 수 있다. overlap 150은 기반 fixed 청크에 적용되며 추가 개요와의 전체 중복량 상한이 아니다. 동일 문서·동일 원문 좌표만 중복 제거한다. 전체 원문은 fixed 청크로 보존한다.
- 전체 unittest 570개 통과. pip check 및 git diff --check 통과. 저장 검색 기록 168개와 단계 진단 168개를 독립 재계산 검증했다. 인덱스·코드 해시, 벡터 정규화, 원문 범위, 메타데이터, 검색 제한, 근거 좌표를 확인했다.
- 기존 인덱스는 읽기 전용 재사용. 공통 검색·생성 모듈과 환경 파일 변경 없음. 기본 전략 fixed 유지. 커밋·푸시·병합 없음.
- 결과 JSON과 검증 스크립트는 `results/a-overview-additive/chy-0020/`에 저장했다. 원문 포함 결과 대신 이 요약을 공유한다.

| 조건 | 질의 임베딩 호출 | 질의 토큰 | 재정렬 쌍 | 재정렬 토큰 |
| --- | ---: | ---: | ---: | ---: |
| fixed-none | 56 | 1566 | 3240 | 1988911 |
| overview-stable-none | 56 | 1566 | 3300 | 1911575 |
| overview-additive-none | 56 | 1566 | 3240 | 1912871 |
