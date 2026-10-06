# 사업명 문맥 임베딩·구조 청킹 구현 및 오프라인 검증

후속 상태: 승인된 후보 build와 smoke 유료 평가를 완료했으나 답변 품질 퇴행으로 sample은 실행하지 않았다. 최신 결과는 [유료 검증 보고서](chunking-project-context-paid-validation.md)를 참고한다. 아래 내용은 유료 실행 전 오프라인 검증 기록이다.

2026-10-06 UTC. 작업 브랜치: `feature/chunking-project-context`, 기준 커밋: `da4db56`.
코드는 커밋하지 않은 상태이며 코드 해시를 아래에 기록했다. 커밋·푸시·병합과 유료 API 실험은 수행하지 않았다.

## 변경과 호출 관계

`run.py → parse_documents → build_index → chunk_documents/embed_texts → load_index → retrieve → generate_answer → evaluate`의 연결을 유지했다.
검색도 `embed_texts`를 호출하므로 문맥 추가는 `build_index`의 문서 입력에만 적용한다.

- `RFP_EMBEDDING_CONTEXT=none|project`: 기본값 `none`. `project`는 사업명·발주 기관을 임베딩 요청 앞에만 붙인다. 빈 값·비문자열 값은 생략하며 둘 다 없으면 원래 본문만 사용한다. 요청 문맥에만 NFC 정규화·공백 정리를 적용하고 원본 메타데이터는 바꾸지 않는다. 사업명 추론이나 파일명 대체는 하지 않는다.
- `RFP_CHUNKING_STRATEGY=fixed|boundary|structured`: 기본값 `fixed`. 기존 두 전략의 실제 100문서 출력이 기준 커밋과 완전히 같음을 검증했다.
- `structured`: 섹션 시작 경계를 우선하고, 크기 제한 이내의 표는 통째로 보존한다. 큰 표는 행 경계를 우선하며 행 자체가 제한을 넘으면 문자 분할한다. 제목과 뒤따르는 본문 첫 줄을 함께 두도록 시도하되 최대 크기는 지킨다. 표 앞의 짧은 청크를 허용한다. 구형 무번호 표 표시도 지원하며, 짝이 안 맞거나 중첩된 표 표시는 일반 본문으로 처리한다.
- 구조 경계 보호를 위해 `structured`의 실제 중복량은 `chunk_overlap` 이하일 수 있다. 짧은 표와 행 중간에서 시작하지 않도록 시작점을 앞으로 조정하고, 같은 끝점이 반복되면 중복을 줄여 전진한다. 비공백 원문 누락 없이 종료한다.
- `config.json`에 선택 키 `embedding_context`, `embedding_context_version=1`을 추가했다. `chunking_strategy`로 신규 전략을 구분하며 `chunking_version=1`이다. 구형 인덱스는 신규 키 없이도 로드한다.

공개 함수 이름·인자 순서·기본값·반환 형식, `Document → Chunk → Hit → Answer` 필수 키, `doc_id`와 `chunk_id` 규칙을 유지했다. 청크는 계속 원문 `text[start_char:end_char]`이며 본문·표 헤더·표 표시를 합성하거나 복제하지 않는다. FAISS 벡터와 청크 순서, L2 정규화, 32개 배치, 저장 파일 3종을 유지한다. 원본·기존 인덱스의 실행 전후 해시가 일치한다.

기존 Langfuse 관측과 토큰 집계를 그대로 사용한다. 문맥 값은 임베딩 요청에만 전달되며 config·Langfuse에 추가하지 않는다. 사업명·기관명·본문·파일명이 기록에 없는지 가짜 추적 클라이언트로 검증했다. 실험 ID 생성 코드 및 `.env`는 수정하지 않았다.

## 문서와 코드의 차이

- 공개 함수와 필수 키는 팀 가이드와 일치한다. 실제 코드는 이미 `Document.sections`, `section_of`, `fixed/boundary`, `chunking_strategy/version`을 제공하지만 가이드에 이 확장이 설명되어 있지 않다.
- 이번 구현은 파싱이 제공하는 선택형 `sections`와 번호가 붙은 표 표시를 소비한다. 없는 정보나 잘못된 섹션 위치는 제외한다. 파싱 코드를 변경하지 않았다.
- 자동 평가 Markdown은 청킹 전략·문맥 설정을 출력하지 않는다. 로컬 config/평가 JSON에는 설정이 남으며 공유 비교에는 이 별도 보고서를 사용한다. 공통 보고서 모듈은 수정하지 않았다.
- build는 로컬 평가 JSON/공유 보고서를 만들지 않는다. 향후 build의 실제 임베딩 토큰은 기존 Langfuse 기록에서 확인해야 한다. 이번 오프라인 실행에는 외부 추적도 API 호출도 없다.

## 같은 100문서의 오프라인 비교

공통 조건: 100문서, 원문 8,658,300자, `chunk_size=1000`, `chunk_overlap=150`.
모든 전략의 문자 위치·크기·전체 비공백 커버리지·ID 고유성·메타데이터·JSON 왕복·원본 불변성을 검사했다.
청킹 시간은 준비 실행 후 전략 순서를 번갈아 5회 실행한 중앙값이며 API·디스크 저장 시간은 제외한다. Python 3.12.3, Linux x86_64에서 측정했다.

| 지표 | fixed | boundary | structured |
| --- | ---: | ---: | ---: |
| 청크 수 | 10,218 | 12,408 | 11,844 |
| 인덱스 생성 예상 요청 수(32개/배치) | 320 | 388 | 371 |
| 길이 최소 / 중앙값 / 95백분위 / 최대 | 161 / 1000 / 1000 / 1000 | 160 / 906 / 1000 / 1000 | 16 / 868 / 997 / 1000 |
| 150자 미만 청크 | 0 | 0 | 127 |
| 중복 문자 수 | 1,517,700 | 1,846,200 | 820,198 |
| 짧은 표 중 하나의 청크에 온전히 포함되지 못한 표 | 2,853 | 856 | 0 |
| 짧은 표 내부의 분할·중복 시작 경계 | 8,314 | 6,487 | 0 |
| 전체 표 내부 경계 | 14,243 | 12,391 | 4,950 |
| 표 행 중간 경계 | 14,011 | 9,457 | 303 |
| 1,000자 이하 행의 중간 경계 | 13,135 | 8,575 | 0 |
| 섹션 시작을 가로지르는 청크 | 2,350 | 2,726 | 2,579 |
| 청킹 중앙 시간(ms) | 36.54 | 459.10 | 488.88 |
| project 문맥 사용 시 추가 입력 문자 수 | 547,340 | 664,254 | 633,849 |

짧은 표는 여는 표시부터 닫는 표시까지 1,000자 이하인 표이며 총 7,770개다. 표 내부 경계는 문서별 중복 제거된 청크 시작·끝 위치를 센다. 긴 표의 행 사이 분할도 전체 표 내부 경계에 포함되므로 전체 경계 수를 모두 오류로 해석하면 안 된다.

`structured`는 짧은 표를 모두 보존했고, 남은 행 중간 경계 303개는 모두 행 자체가 1,000자를 넘는 경우였다. 반면 fixed보다 청크 수·청킹 시간이 증가했고 짧은 조각 127개가 생겼다. 섹션 경계 침범도 fixed보다 많아 섹션 분리의 일괄적인 향상은 확인하지 못했다. 섹션 경계는 크기·표 보존·중복 조건과 함께 적용하는 우선순위이며 모든 섹션을 독립 청크로 만드는 기능은 아니다.

문맥의 추가 문자 수는 토큰 수가 아니다. 신규 API 호출은 **0회**이며 실제 임베딩·답변 토큰 사용량이나 검색 품질 개선을 측정한 결과가 아니다.

## 테스트 및 기존 평가 기준

- `python -m unittest discover -s tests -v`: 기존 43개 + 신규 14개, 총 **57개 통과**.
- `python -m pip check`: 의존성 오류 없음. 새 패키지 없음.
- 신규 검증: 정확히 크기 제한인 표, 연속 표, 큰 표·긴 행·작은 크기·큰 중복, Unicode/CRLF, 섹션 누락/오류, 제목, 잘못된/중첩/구형 표 표시, 고정 시드 혼합 문서 80개.
- 문맥 정규화·누락 처리·기본 none·잘못된 설정의 API 호출 전 거부, 65문서 다중 배치·역순 응답 대응, 원문 저장, 질문 입력 유지, 구형 config 및 검색→생성→평가 연결, 실패 시 기존 인덱스 보존, 추적 본문 제외를 검증했다.
- 테스트는 가짜 API 클라이언트를 사용한다. 아래 실제 baseline 평가는 이번 변경 전 실행 결과이며 재실행하지 않았다.

| 기존 평가 | 자동 실험 ID | 문항 | Recall@5 | 키워드 포함률 | 평균 응답(s) | 질문 임베딩 토큰 | 생성 입력/출력 토큰 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | chy-0004 | 6 | 1.000000 | 1.000000 | 3.1874 | 251 | 31,950 / 1,567 |
| sample | chy-0005 | 14 | 0.940476 | 0.940476 | 3.3182 | 613 | 76,456 / 3,571 |

기존 sample의 마지막 두 문항은 기관별 여러 사업 나열이며 Recall이 각각 0.5, 2/3이었다. 동일 문서의 청크가 상위 5개를 차지하는 검색 다양성 문제가 보인다. 검색 모듈은 이번 범위 밖이며 이 실패가 문맥 임베딩만으로 해소된다고 주장하지 않는다. 키워드 포함률은 사실성 지표가 아니다.

## 재현 및 후속 유료 평가

오프라인 비교 재실행(유료 호출 없음, 실행마다 새 산출물 폴더 생성):

```bash
cd /home/spai1304/rfp-rag
.venv/bin/python tests/benchmark_chunking.py --baseline-ref da4db56
```

최종 측정 산출물 라벨: `chy-project-context-20261006T043827593755Z`.
이는 로컬 폴더 식별자이며 자동 실험 ID를 대체하지 않는다.

- 파싱 복사본: `data/processed/chy-project-context-20261006T043827593755Z/`
- 전략별 청크·원문 없는 집계: `results/chy-project-context-20261006T043827593755Z/`
- 후보 인덱스 예정 경로: `indexes/chy-project-context-20261006T043827593755Z/structured-project` (아직 생성하지 않음)

문맥 또는 청킹 전략 변경은 **인덱스 재생성이 필요**하다. 동일 파싱 결과를 재사용하므로 parse는 다시 실행할 필요가 없다. 아래 유료 명령은 범위 협의 후에만 실행한다. `.env`의 `EXPERIMENT_OWNER`와 기존 자동 번호 생성을 그대로 사용한다.

```bash
CHUNKING_RUN_LABEL=chy-project-context-20261006T043827593755Z
RFP_CHUNKING_STRATEGY=structured RFP_EMBEDDING_CONTEXT=project \
OPENAI_EMBEDDING_MODEL=text-embedding-3-small OPENAI_GENERATION_MODEL=gpt-5-mini \
.venv/bin/python run.py build \
  --processed-dir "data/processed/$CHUNKING_RUN_LABEL" \
  --index-dir "indexes/$CHUNKING_RUN_LABEL/structured-project" \
  --chunk-size 1000 --chunk-overlap 150
```

build 성공 후 같은 문서·평가셋·모델·top-k로 평가한다:

```bash
for CHUNKING_EVAL_NAME in eval_smoke_v0 eval_sample_v0; do
  OPENAI_GENERATION_MODEL=gpt-5-mini .venv/bin/python run.py evaluate \
    --processed-dir "data/processed/$CHUNKING_RUN_LABEL" \
    --index-dir "indexes/$CHUNKING_RUN_LABEL/structured-project" \
    --results-dir "results/$CHUNKING_RUN_LABEL/structured-project" \
    --eval-file "data/$CHUNKING_EVAL_NAME.json" --top-k 5
done
```

후보 1개는 빌드 371회 임베딩 요청 + 20문항 질문 임베딩 최대 20회 + 생성 최대 20회, 합계 최대 **411회 기본 요청**이다. SDK 재시도는 제외하며 토큰/금액 상한을 뜻하지 않는다. 이번에는 실행하지 않았다.

변경 설정은 `fixed/none → structured/project`이며 나머지 모델·크기·중복·top-k·평가셋·100문서 범위를 고정한다. 실제 평가 후 기존 성공 문항의 퇴행, 위 두 실패, 응답 시간·토큰을 비교해야 한다. 두 변경의 효과를 따로 판단하려면 문맥만 또는 구조만 바꾼 추가 후보가 필요하며 비용을 별도 협의한다. 불리하면 기본 fixed/none과 기존 인덱스를 계속 사용할 수 있다.

## 식별 정보와 보관

| 항목 | SHA-256 |
| --- | --- |
| documents.json | `97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a` |
| embedding.py | `6aacd400cb0864ca5089c93dd7a2802478af3cb0ff5c23116f4c5715c43ff5ef` |
| benchmark_chunking.py | `70ceeb0cb8bd370cef71877986a7998489fbbbcecfb6a4b17988e352b8b54090` |
| 기존 index.faiss | `e8a0573dec81ab8ecc1fd1a940063c39bc989495487202c109426c3af13dfa0b` |
| structured chunks.json | `d9059c5d9c35ea07862aa9757564b0eebc23a82b55afca11483d54100f24fe4d` |
| eval_smoke_v0.json | `a0cc5ac0d961d2398a1eb2919922b104b286caf96de3c4e03cb70b9b2750b6b0` |
| eval_sample_v0.json | `62a6d2540e1bf57acda9defaf1f135041b9a7ca40f3ece995cfdf8b40a6a12a7` |

상세 표본 시간과 나머지 해시는 같은 라벨의 `summary.json`에 저장했다. 기존 산출물과 다른 사람의 작업 경로는 변경하지 않았다. 원문이 포함된 documents/chunks/평가 JSON은 Git에서 제외하며 공유 대상은 본 요약이다. 원문·질문·답변·기관명·사업명·인증 정보는 이 보고서에 포함하지 않는다.
