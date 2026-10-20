# 시나리오 A additive 기본값 및 개인 경로 적용

## 변경과 계약

- A 전략 미설정 기본값을 overview_additive로 변경. 명시적 fixed 및 기존 인덱스 로드 유지.
- 공개 함수 시그니처·필수 데이터 키·원문 좌표·3파일 저장 형식 유지. run.py/a_run.py·검색·생성·추적 제품 코드 변경 없음.
- 기존 미커밋 청킹·project_blend 실험 변경은 보존. 이번 작업은 기본값·설정 예제·테스트·문서·개인 배포 연결에 한정.
- 실제 B 기본값은 structured이며 수정하지 않음. 문서의 과거 fixed 설명을 정정.
- additive는 fixed의 본문·좌표·메타데이터를 보존하지만 개요 삽입 때문에 chunk_id 순번은 바뀔 수 있음.

## 검증

- 변경 전 570개, 변경 후 577개 unittest 통과. pip check 및 git diff --check 통과.
- 기본값, 명시적 fixed, 개요 조건·표 경계·중복·빈 입력·작은 청크/최대 overlap/CRLF, JSON/입력 불변성, 기존 fixed 로드·검색·생성 연결 검증.
- 현재 기본 청킹 결과가 PR #37 전체 재임베딩 additive의 10,266개 청크와 완전 일치. 문서·평가·임베딩·리랭커 해시와 인덱스 파일 검증 후 사본 사용.
- 재파싱·재임베딩 없음. 기존 검증된 fixed 10,218개/개요 48개 추가 산출물 재사용.
- 자동 실험 ID: chy-0021. 개인 EXPERIMENT_OWNER 및 기존 번호 생성기 사용. Langfuse 설정 유지.

| 조건 | Recall@5 | MRR@5 | 근거 적중률 | 평균 검색 초 | 질문 임베딩 토큰(56문항) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 0.976190 | 0.976190 | 0.90 | 0.917 | 1566 |
| overview_additive | 0.976190 | 0.976190 | 0.95 | 0.967 | 1566 |

동일 문서 100건, 기존 평가 50문항(답변 가능 42·근거 지표 20), 기존 목표 검증 6문항 별도 집계. BGE-M3/none/1000/150, cross-encoder 후보50, hybrid100/100, max_per_doc2, expand5, threshold0.53 등 기존 A 옵션 동일.

- 두 조건 모두 이전 전체 재임베딩 검증의 56문항 점수와 일치. yB058 근거 적중 0→1 외 검색 점수 변화 없음. 목표 검증 6문항의 Recall 1, MRR 0.916667, 근거 적중 1 유지.
- 질문 임베딩 실패·OOM 재시도·잘림은 모두 0.
- 저장 문서의 오프라인 토큰 재계산: fixed 5,735,042, additive 5,742,546 (+7,504). 이번에 실제 build를 수행한 사용량이 아님.
- 평균 검색은 이번 단일 순차 실행에서 additive가 약 0.050초 느림. 이전 실행과 속도 차이 방향이 달라 속도 우열은 단정하지 않음.
- 이전 전체 build 시간은 fixed 159.43초, additive 158.09초. 이번 전환은 해당 산출물을 복사함.

## 개인 운영 경로와 재실행

- `data/processed/a` → `a-additive-deploy/chy-0021`
- `indexes/a` → `a-additive-deploy/chy-0021`
- 원본 실험 인덱스·문서는 그대로 보존. 배포 데이터·인덱스·원문 포함 결과 JSON은 Git 제외 확인.

```bash
# 기본 A 경로 검색 (로컬 모델; 자동 실험 ID)
.venv-a/bin/python a_run.py ask --question "사업 기간은 얼마인가요?"
# run.py 사용 시 반드시 A를 명시
RFP_SCENARIO=A .venv-a/bin/python run.py ask --question "사업 기간은 얼마인가요?"
# 문서/알고리즘/모델 변경 시 새 경로에만 build
RFP_CHUNKING_STRATEGY=overview_additive .venv-a/bin/python a_run.py build \
  --processed-dir data/processed/a --index-dir indexes/a-additive-next
```

새 build 경로가 없는지 확인하고 실행합니다. 현재 활성 링크를 대상으로 build하지 않습니다. 같은 청크·모델·문서인 이번 전환에는 추가 build가 필요 없습니다. 실행기는 `experiments/a_additive_activation.py`이며 기본 경로가 이미 존재하면 덮어쓰지 않고 중단합니다.

복귀가 필요하면 질의·빌드를 중단하고 `indexes/a`가 위 개인 링크인지 확인한 뒤 링크만 다음 보존 fixed로 교체합니다. 문서 링크는 그대로 유지합니다.

`/home/spai1304/rfp-rag/results/pr37-full-rebuild-prep/runs/20261020T002355Z/fixed`

## 한계

- yC067 정답 문서 누락 및 yC080 근거 미적중 유지. 평가셋 재사용·단일 반복의 한계가 있음.
- 생성 정확도 0.798→0.821은 이전 담당자 공유 결과. 이번 검색 재평가에서 새로 측정한 생성 정확도가 아님.
- 유료 판정 API 호출 없음. yB058·h10 로컬 생성 smoke 모두 완료. 두 결과 status=ok, 출처 5개, 인용 번호 유효, 실제 additive 인덱스 해시 일치. 실험 ID는 자동 생성된 chy-0022·chy-0023이며 두 결과에 기존 trace ID가 저장됨. 서버의 Langfuse 수신 성공을 별도 확인한 것은 아님.
- 생성 모델 자체의 토큰 계측은 기존 구현 범위 그대로이며, 이번 보고서의 실측 토큰은 질문 임베딩에 한정. 두 문항 smoke는 전체 생성 정확도 재평가가 아님.
- 이 보고서의 구현·평가 당시에는 커밋·푸시·병합·PR 수정 없이 검증했습니다. 후속 게시 작업은 사용자 요청에 따라 별도 수행합니다.

최종 정리에서는 주석·README 설명과 실행기 미사용 import만 정리했습니다. 성능 평가 이후 알고리즘 변경은 없습니다.

## 게시 전 파일 집합 검증

무관한 project_blend 비율 실험을 제외한 PR 파일 집합에서 unittest 563개 통과. 개인 전체 작업 공간의 577개와 구분합니다. 게시할 코드가 현재 적용된 additive 10,266개 청크와 완전히 같은 결과를 만드는지도 확인했습니다.
