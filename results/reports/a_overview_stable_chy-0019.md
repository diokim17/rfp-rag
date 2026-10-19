# 사업 개요 외부 청크 경계 보존 실험

실험 ID: chy-0019; 2026-10-19T07:25:37.688236+00:00

변경: overview-stable v1. fixed/overview-v1 대비 임베딩 문맥 none, 최신 A 검색 옵션 동일.
규칙: 다음 절까지 1000자 이내, 사업명/기간/예산/계약 방식 중 2종류 이상. 개요 인식 규칙은 v1과 동일. 원래 fixed 청크에서 개요 교집합만 차감하고 개요를 추가. 비접촉 청크 좌표 보존.
질문·정답·문서 ID를 청킹 규칙에 사용하지 않음. 기존 기본값·공개 함수·원문 좌표·3파일 인덱스 형식 유지.

| 평가 | 조건 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| existing | fixed-none | 0.976190 | 0.976190 | 0.900 | 1130.18 | 1002.91 | 1115.57 |
| existing | overview-none | 1.000000 | 0.980952 | 0.900 | 1143.66 | 992.95 | 1138.76 |
| existing | overview-stable-none | 0.976190 | 0.976190 | 0.950 | 1137.69 | 985.09 | 1113.61 |
| new | fixed-none | 1.000000 | 0.916667 | 1.000 | 1064.12 | 1061.29 | 1130.41 |
| new | overview-none | 1.000000 | 0.916667 | 1.000 | 1044.45 | 1039.26 | 1132.38 |
| new | overview-stable-none | 1.000000 | 0.916667 | 1.000 | 1049.17 | 1019.75 | 1133.60 |

기존 eval_v4 50문항(답변 가능 42, 근거 대리지표 20)과 새 질문 6문항은 별도 집계.
새 6문항은 기존 평가 정답 문서와 겹치지 않는 6문서의 개요 기간/예산 질문. 규칙 고정 후 작성했으나 개요가 인식되는 문서만 선택한 목표 검증이므로 대표성 있는 독립 평가셋이 아님.
새 질문은 원문 근거 좌표로 전체 항목이 최종 청크에 포함되는지도 검사. 답변 생성·판정 미실행.

선택 문서 46, 구간 48, 후보 청크 10294.
실제 후보 build 141.45초, 로컬 임베딩 통계 `{"calls": 322, "texts": 10294, "input_tokens": 5733558, "truncated_texts": 0, "oom_retries": 0, "failures": 0}`.

## 재현

```bash
.venv-a/bin/python experiments/a_overview_stable_chy.py --new-cases data/processed/a-overview-stable/chy-0019/new_cases.json --repeat 3
```

```json
{
  "experiment": {
    "experiment_id": "chy-0019",
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
    "a_embedding.py": "2e653ef1f4d010ad8742d073997f60f433de6eec60ef31ff872eea259ded7cda",
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
    "experiments/a_overview_stable_chy.py": "5d285e84b0c0c0493306a5f1b011d1b8c9574742b340a56e44bc9e1aba9c2726",
    "experiments/a_evidence_pipeline_chy.py": "5b01f1b5a0287fc7885c12a94854dd9e6cdb6dfcf456f6413fd8d3db5d1425c9"
  }
}
```

## 판단

목표 가설은 관측 범위에서 지지됐다. overview-stable은 yB058의 기간 근거 최종 1위를 유지하면서 h10을 최종 5위로 회복했다. 기존 근거 평가 20문항에서 적중률은 fixed/overview의 18/20(90%)에서 19/20(95%)로 상승했다. 다만 문서 Recall@5는 41/42(97.619%)로 fixed와 같고 overview의 42/42(100%)보다 낮다. overview가 성공했던 yC067은 다시 누락됐다. yC067 정답 문서는 후보 단계에서 fixed 4위, overview 8위, stable 5위로 모두 확보됐으나 재정렬 후 각각 7위/5위/8위여서 overview에서만 최종 결과에 포함됐다. 따라서 **기본값 채택은 보류**하고 선택형 후보로 유지한다.

| 사례 | fixed-none | overview-none | overview-stable-none |
| --- | --- | --- | --- |
| yB058 | 후보 누락 | 근거 최종 1위 | 근거 최종 1위 유지 |
| h10 | 근거 최종 5위 | 근거 제외 | 근거 최종 5위 회복 |
| yC080 | 문서당 제한으로 근거 제외 | 동일 | 동일 |
| yC067 | 정답 문서 누락 | 정답 문서 최종 5위 | 정답 문서 누락 |

20문항의 근거 적중은 기존 값·키워드 대리지표이며 실제 답변 정확도가 아니다. 3회 반복은 독립 평가 문항 수를 늘리지 않는다. 추가 6문항은 CHY-0018에서 이미 사용한 목표 검증셋이다. 세 조건 모두 Recall 1.0, MRR 0.916667, 근거 적중 및 전체 근거 좌표 포함률 1.0으로 추가 우위는 없었다. 답변불가 8문항의 빈 결과 비율은 fixed 0.25, 두 overview 조건 0.125이며 거절 정확도·환각률로 해석하지 않는다.

## 원문·경계 보존과 비용

개요 인식 규칙과 대상 46문서/48구간은 CHY-0018과 동일하다. fixed 10,218개 → stable 10,294개(+76), 기존 overview 10,273개 대비 +21개다. 개요와 겹치지 않는 fixed 청크 10,151개는 본문·원문 좌표·메타데이터가 모두 보존된다. 겹치는 fixed 청크에서 개요를 차감하고, 동일하거나 다른 조각에 완전히 포함된 조각을 제거한다. 모든 비공백 원문을 보존하고 최대 길이 1,000자·overlap 150자 제한을 지킨다. 청크 ID의 번호 대응은 달라질 수 있다. 표 보존 전반의 개선을 주장하지 않는다.

공유 GPU 메모리 확보 후 실행했다. 각 조건 워밍업 후 실행 순서를 회전했고, p50/p95는 질의 임베딩·검색·재정렬·단계 캡처 시간이다. 사후 토큰 집계와 답변 생성은 제외한다. 공유 장비 측정이므로 속도 우열을 주장하지 않는다.

| 조건 | 질의 임베딩 호출 | 질의 입력 토큰 | 재정렬 쌍 | 재정렬 입력 토큰 | 임베딩 실패 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-none | 168 | 4698 | 9720 | 5966733 | 0 |
| overview-none | 168 | 4698 | 9900 | 5763156 | 0 |
| overview-stable-none | 168 | 4698 | 9900 | 5734725 | 0 |

## 검증 및 후속 범위

- 전체 unittest 568개, pip check, git diff --check 통과.
- 저장 검색 결과 504개와 단계 진단 168개를 독립 재계산했다. 인덱스 해시·벡터 정규화·문서당 제한·지표·새 질문 근거 좌표·원문 보존을 검증했다. 조건별 3회 반복의 검색 결과와 점수는 동일했다.
- 기존 fixed/overview 인덱스는 읽기 전용 재사용. 문서·평가셋·모델·reranker 해시와 검색 옵션을 확인했다. 실험 종료 시 코드와 인덱스 해시 불변을 확인했다.
- 기본 전략 fixed 및 공개 함수 계약·3파일 인덱스 형식 유지. 공통 검색·생성 모듈 및 환경 설정 파일 변경 없음. 커밋·푸시·병합 없음.
- 다음 검증은 미사용 문서·질문에서 근거 적중을 재평가하고 yC067의 손실을 원문과 단계별 후보로 진단하는 것이다. yC080 문서당 제한은 검색 담당 범위다. 이번 결과에 맞춰 같은 평가셋으로 청킹 규칙을 추가 조정하지 않는다.
- 검증 스크립트: `results/a-overview-stable/chy-0019/verify_saved.py`; 원문이 포함된 결과 JSON은 로컬 결과 폴더에 보관한다.
