# 사업 개요 핵심 사실 보존 청킹 실험

실험 ID: chy-0018; 2026-10-19T06:33:51.898533+00:00

변경: 짧은 사업 개요를 원문 그대로 격리하는 overview v1. fixed/structured-v3 대비 임베딩 문맥 none, 최신 A 검색 옵션 동일.
규칙: 다음 절까지 1000자 이내, 사업명/기간/예산/계약 방식 중 2종류 이상. 긴 절·미인식 절은 fixed. 개요 양끝에서 overlap 중단.
질문·정답·문서 ID를 청킹 규칙에 사용하지 않음. 기존 기본값·공개 함수·원문 좌표·3파일 인덱스 형식 유지.

| 평가 | 조건 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| existing | fixed-none | 0.976190 | 0.976190 | 0.900 | 1140.25 | 1005.94 | 1120.69 |
| existing | structured-v3-none | 1.000000 | 0.988095 | 0.850 | 1038.61 | 881.75 | 1012.09 |
| existing | overview-none | 1.000000 | 0.980952 | 0.900 | 1140.84 | 990.69 | 1123.43 |
| new | fixed-none | 1.000000 | 0.916667 | 1.000 | 1062.06 | 1058.45 | 1126.37 |
| new | structured-v3-none | 1.000000 | 0.916667 | 1.000 | 960.14 | 977.47 | 1040.40 |
| new | overview-none | 1.000000 | 0.916667 | 1.000 | 1044.43 | 1038.82 | 1128.38 |

기존 eval_v4 50문항(답변 가능 42, 근거 대리지표 20)과 새 질문 6문항은 별도 집계.
새 6문항은 기존 평가 정답 문서와 겹치지 않는 6문서의 개요 기간/예산 질문. 규칙 고정 후 작성했으나 개요가 인식되는 문서만 선택한 목표 검증이므로 대표성 있는 독립 평가셋이 아님.
새 질문은 원문 근거 좌표로 전체 항목이 최종 청크에 포함되는지도 검사. 답변 생성·판정 미실행.

선택 문서 46, 구간 48, 후보 청크 10273.
실제 후보 build 141.73초, 로컬 임베딩 통계 `{"calls": 322, "texts": 10273, "input_tokens": 5732055, "truncated_texts": 0, "oom_retries": 0, "failures": 0}`.

## 재현

```bash
.venv-a/bin/python experiments/a_overview_chy.py --new-cases data/processed/a-overview/chy-0018/new_cases.json --repeat 3
```

```json
{
  "experiment": {
    "experiment_id": "chy-0018",
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
    "a_embedding.py": "03c12d61519ff8f2032e555a114b351a9b0aed6244cb6fc3add3190941fecf36",
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
    "experiments/a_overview_chy.py": "b430445cf593555f81ceb01828337eea6cccd51040c76b822d699454b0005b6b",
    "experiments/a_evidence_pipeline_chy.py": "5b01f1b5a0287fc7885c12a94854dd9e6cdb6dfcf456f6413fd8d3db5d1425c9"
  }
}
```

## 판단 및 실패 사례

목표 문제는 짧은 사업 기간·예산 사실이 1000자 청크의 다른 내용에 섞여 검색 후보에서 누락되는 현상이다. project_blend 비율을 다시 조정한 실험이 아니라 청킹 단계의 근거 보존 가설을 검증했다. 최종 판단은 **기본값 채택 보류**다. 특정 후보 누락은 해결했지만 전체 근거 적중률은 개선되지 않았고, 답변 생성 성능은 측정하지 않았다.

| 문항 | fixed-none | overview-none | 해석 |
| --- | --- | --- | --- |
| yB058 | 벡터 top100 밖, BM25 49위, RRF 96위로 후보 누락 | 벡터 1위, BM25 6위, RRF·재정렬·최종 모두 1위 | 기간 근거 청크 1000→153자. 목표 후보 누락 해결 |
| h10 | 근거 본문 재정렬 6위, 문서별 제한 적용 후 최종 5위 | 근거 본문 재정렬 8위, 최종 제외 | 근거 대리지표 1→0. 세 정답 문서는 최종 결과에 모두 포함되어 문서 Recall은 유지 |
| yC080 | 근거 재정렬 3위, 문서당 2개 제한으로 제외 | 근거 재정렬 5위, 같은 제한으로 제외 | 검색 후보 확보 이후의 문제는 청킹만으로 해결되지 않음 |
| yC067 | 정답 문서 최종 결과 누락 | 정답 문서 최종 5위 | 문서 Recall 상승에 기여. 근거 대리지표 평가 대상은 아님 |

기존 근거 평가 20문항에서 yB058의 개선과 h10의 하락이 상쇄됐다. h10은 본문 키워드 기반 대리지표이며 실제 답변의 퇴행으로 단정할 수 없다. 개요 격리 후 나머지 구간의 fixed 시작점도 이동하므로, 효과 전체를 개요 길이 축소 하나의 인과효과로 단정하지 않는다.

신규 6문항은 세 조건 모두 Recall 1.0, MRR 0.916667, 근거 적중률 1.0이며, 별도 원문 좌표로 검사한 전체 근거 포함률도 모두 1.0이다. 기존 방식도 성공해 추가 우위는 입증하지 못했다. 같은 100문서 안의 목표 사례이며 대표성 있는 신규 코퍼스 평가가 아니다.

답변불가 8문항의 빈 검색 결과율은 fixed 0.25, structured/overview 0.125다. 이는 거절 정확도나 환각률이 아니며 답변 생성 없이 품질 우열을 판단할 수 없다.

## 시간·토큰·실패

표의 p50은 저장된 median_seconds다. p50은 전형적인 요청, p95는 느린 쪽 요청의 지연을 보여주므로 둘 다 제시했다. 시간은 질의 임베딩·검색·재정렬 및 단계 캡처를 포함하며 답변 생성과 사후 토큰 재계산은 제외한다. 각 조건 워밍업 1회 후 반복별 실행 순서를 회전했다. 반복은 독립 평가 문항 수를 늘리지 않는다. 평균이 p95보다 큰 경우는 상위 5% 밖의 긴 요청 영향이며 모순이 아니다. 공유 장비에서의 3회 측정으로 속도 개선을 주장하지 않는다.

| 조건 | 측정 질의 임베딩 호출 | 질의 입력 토큰 | 재정렬 쌍 | 재정렬 입력 토큰 | 실패 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-none | 168 | 4698 | 9720 | 5966733 | 0 |
| structured-v3-none | 168 | 4698 | 9885 | 4906935 | 0 |
| overview-none | 168 | 4698 | 9900 | 5763156 | 0 |

후보 인덱스 생성은 141.73초, 로컬 임베딩 322배치·10,273텍스트·5,732,055입력 토큰이다. 임베딩 잘림·OOM 재시도·실패는 0이다. 재정렬 토큰은 실제 후보를 토크나이저로 재계산한 길이이며 패딩은 제외한다. 재정렬 토큰은 위 표에 별도로 집계하며 trace의 임베딩 토큰 합계와 혼동하지 않는다. 원격 생성·판정 API 호출은 0, 답변 모델은 사용하지 않았다.

## 고정 조건·계약·검증

- 담당자: chy; 상태: 실행 및 검증 완료; 평가일: 2026-10-19 UTC.
- 문서 100개, 기존 eval_v4 50문항과 신규 목표 검증 6문항. 파일 SHA256은 위 실험 메타데이터에 기록했다.
- fixed-none 인덱스는 chy-0011, structured-v3-none은 chy-0005에서 읽기 전용 재사용하고 현재 검색 코드로 다시 평가했다. 과거 보고서 수치를 직접 비교한 것이 아니다.
- 청킹 1000/150, 임베딩 BGE-M3(default_model), 문맥 none. 변경 변수는 청킹 전략 fixed / structured-v3 / overview-v1이다.
- 검색은 vector100 + BM25(prefix)100 → RRF(k=60) 후보50 → 확장5(최대 후보60) → BGE reranker v2 m3 → 최종5, 문서당 최대2. min_score 0.53, section 사용, rewrite/within_doc off, dedup false를 유지했다.
- overview가 선택한 문서46개·구간48개, 나머지54개 문서 청크는 fixed와 동일하다. 전체 청크는 10,218→10,273(+55)이다. 모든 원문 범위가 유지된다. 표 행 경계 분할이 일부 늘어 표 보존 전반의 개선을 주장하지 않는다.
- 공개 함수 이름·인자·기본값·반환형·필수 키·원문 좌표·문서 ID 규칙·3파일 인덱스 저장 형식 유지. 청킹 변경으로 청크 경계와 문서 내 청크 번호의 대응은 바뀐다. 기본 전략 fixed는 유지한다.
- run.py와 타 담당 모듈 및 공통 의존성 변경 없음. Langfuse 추적 유지, 원문·질문·답변·인증 정보 로그 추가 없음. 실험 ID는 자동 발급했고 .env의 담당자를 사용했다.
- 신규 경계·원문 보존·잘못된 설정·저장/검색/평가 연결 테스트 포함 전체 553개 통과. pip check 및 git diff --check 통과.
- 저장 결과 504개와 단계 진단 168개를 독립 재계산 검증했다. 인덱스 해시·벡터 정규화·문서당 제한·원문 범위·지표·신규 근거 좌표를 확인했다. 각 조건의 3회 반복 검색 결과와 점수는 동일했다.
- 문서의 시나리오 A 미구현 설명은 현재 코드와 다르다. 실제 연결은 run.py → a_run.py → a_embedding의 build/load → a_retrieval.retrieve → a_generation/a_evaluation이다. 이번 실험은 같은 build/load/retrieve를 사용하고 답변 생성은 생략했다.

## 재실행 범위와 후속 검증

위 재현 명령은 새 실험 ID와 별도 processed/index/results 경로를 자동 생성한다. overview 도입 시 청크 경계와 벡터가 바뀌므로 인덱스 재생성이 필요하다. 기존 문서 스냅샷을 재사용하므로 파싱은 불필요하다. 이번에 생성한 인덱스로 검색 설정만 비교한다면 재임베딩은 불필요하다.

다음 검증은 개요 외부의 기존 청크 경계를 최대한 유지하는 방식과 현재 방식을 비교하여 h10의 상쇄를 줄일 수 있는지 확인하는 것이다. 이후 더 넓은 미사용 문서·질문으로 평가하고 실제 답변 근거 충족 여부를 확인해야 한다. yC080의 문서별 제한 문제는 타 담당 검색 모듈과 협의할 별도 사안이다. 이번 결과만으로 기본값을 변경하지 않는다.

원문 포함 JSON은 로컬 results/a-overview/chy-0018에 보관하고 공유용으로 이 요약을 사용한다. 커밋·푸시·병합은 수행하지 않았다. p50은 완료 후 원래 결과의 median_seconds에서 이 보고서에 추가했으며 측정 코드는 변경하지 않았다.
