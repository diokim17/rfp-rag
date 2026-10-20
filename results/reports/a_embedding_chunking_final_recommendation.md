# 시나리오 A 임베딩·청킹 최종 권장안

2026-10-19. CHY-0020까지의 측정에 따른 설정 제안이며 기본값·환경 파일에 자동 적용하지 않았다.

| 항목 | 권장 설정 |
| --- | --- |
| 임베딩 | BGE-M3, 로컬 `default_model` |
| 임베딩 문맥 | `none` |
| 청킹 전략 | `overview_additive` v1: fixed 전체 유지 + 짧은 개요 추가 |
| 기반 청크 크기 / overlap | 1000자 / 150자 |
| 개요 규칙 | 기존 overview와 동일: 1000자 이내, 사업명·기간·예산·계약 방식 중 2종 이상, 표 내부에서 시작·종료하는 구간 제외 |
| 벡터 | 기존 방식 유지: 1024차원, L2 정규화, FAISS IndexFlatIP |
| 현재 평가용 인덱스 | `indexes/a-overview-additive/chy-0020/` |

## 선택 이유

- CHY-0020에서 fixed/stable/additive를 같은 100문서·eval_v4 50문항과 기존 목표 검증 6문항으로 재평가했다. additive는 fixed 대비 문항별 Recall/MRR 하락 없이 근거 적중을 18/20→19/20으로 개선했다. stable과 품질 지표가 같고, 모든 fixed 청크·벡터를 보존하며 추가 청크가 48개뿐이다. [CHY-0020](a_overview_additive_chy-0020.md)
- BGE-M3는 과거 eval_v2 dense 조건에서 KURE-v1보다 Recall이 높았다(0.9351 vs 0.9262). KURE는 MRR이 높았다. 현재 hybrid 조건에서 모델 우열을 새로 검증한 것은 아니므로 BGE-M3 유지의 보조 근거로만 사용한다. [임베딩 모델 비교](a_embedding_comparison_chy.md)
- project_blend는 dense 조건에서 이점이 있었으나 CHY-0017의 현재 hybrid·reranker 조건에서는 none과 Recall/MRR/근거 적중이 같았다. 혼합 벡터 생성 비용을 추가할 근거가 부족해 none을 유지한다. [현재 검색 조건 비교](a_evidence_pipeline_chy-0017.md)
- 청크 크기 500/1500 비교는 dense 조건에서 상충하는 결과였다. 현재 실험에서 검증한 1000/150을 유지한다. [크기 비교](a_chunk_size_comparison_chy-0008.md)

## 적용 범위와 남은 한계

CHY-0020의 권장안은 현재 평가셋과 검색 설정에 대한 실용적 선택이다. 동일 평가셋 재사용, 단일 반복, 근거 대리지표 20문항의 한계가 있다. 답변 생성과 독립 일반화는 검증하지 않았다. yC067 문서 누락, yC080 문서당 제한에 의한 근거 누락은 남아 있다. Recall 100%를 최우선으로 선택한다면 과거 overview 조건이 더 높았으나 근거 적중은 90%였다. 본 권장안은 fixed의 Recall 97.62%를 유지하면서 근거 보존 95%를 얻는 판단이다.

기존 청크 보존은 최종 검색 결과의 불변을 보장하지 않는다. 추가 개요가 후보·문서별 제한 슬롯을 차지할 수 있다. 추가 개요와 fixed 사이에는 150자를 넘는 중복도 가능하다.

검색 담당 설정은 이번 실험에서 고정했다: Vector100+BM25(prefix)100, RRF k60 후보50, expand5(최대60), BGE reranker v2 m3, 최종5, 문서당2, min_score0.53, section 사용, rewrite/within_doc off, dedup false. 검색 설정이 달라지면 이 권장안의 성능도 재확인이 필요하다.

새 문서의 일반 build 경로에서도 `RFP_CHUNKING_STRATEGY=overview_additive`, `RFP_EMBEDDING_CONTEXT=none`, `A_EMBEDDING_MODEL=default_model`, chunk_size1000/chunk_overlap150을 사용할 수 있다. 일반 build는 전체 청크를 임베딩한다. CHY-0020 전용 실행기의 벡터 재사용은 동일 모델·문서·fixed 설정을 확인한 인덱스에 한정한다.
