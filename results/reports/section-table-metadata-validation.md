# 섹션·표 메타데이터 검증

2026-10-06 UTC. 사용자가 본문을 유지하고 metadata에 표 ID·헤더·좌표를 저장하는 방식을 선택했다.

## 변경과 실제 호출 관계

- `embedding.py`에서 모든 청크에 시작 위치 기준 `metadata.section_path`를 추가했다. 기존 parsing.section_of를 사용하며 sections가 없는 구형 문서는 빈 문자열이다. 잘못된 섹션 항목은 원본 변경 없이 제외한다.
- 청크와 겹치는 유효한 표마다 `metadata.tables`에 문서 내 ID, 표 범위, 원문 헤더와 헤더 범위를 저장한다. 구간은 문서 기준 문자 위치의 `[start_char, end_char)`다. 헤더는 첫 Markdown 행과 구분선이며 본문에 합성하지 않는다. ID 없는 구형 표는 null, 헤더를 확인할 수 없으면 빈 문자열/null 좌표다.
- 실제 코드는 요청 당시에도 structured 모드에서 sections·표 표시를 경계 보호에 사용하고 있었다. 빠진 기능은 섹션 경로 저장과 긴 표의 후속 청크에 헤더·ID를 제공하는 것이었다. fixed 기본 설정은 그대로다.
- structured는 짧은 표를 함께 유지하고 긴 표의 행 경계를 보호한다. 섹션은 가능한 경계를 우선하지만 표·크기·overlap 조건 때문에 여러 섹션에 걸칠 수 있어 시작 위치로 표시한다. 행 자체가 chunk_size보다 길면 문자 분할한다.
- 호출 관계: `run.py build → build_index → chunk_documents → section_of/표 정보 → chunks.json`; `load_index → retrieve → generate_answer`가 metadata를 그대로 전달한다. 다른 담당 모듈을 수정하지 않았다.
- 공개 함수·인자·기본값, 원문 슬라이스, 기존 필수 키, 사업명 매핑과 documents_sha256 저장은 유지한다. config에 선택 필드 `chunk_metadata_version=1`을 추가하고 구형 로드를 유지한다. 원문 헤더·섹션 경로는 Langfuse에 추가하지 않는다.

## 검증 결과

전체 unittest **90개 통과**(기존 79개 + 섹션·표 메타데이터 11개), 0.695초. API 요청/토큰 **0**.
LF/CRLF·Unicode 좌표, 섹션 없음/잘못된 항목, 여러 섹션·표, 겹침 없는 인접 구간, 구형/잘못된 표 표시,
초과 길이 행, 원본 불변성, JSON 저장·로드, 검색·생성 전달, 임베딩 입력 보존과 추적 비노출을 확인했다.
`git diff --check` 통과. 패키지 추가 없음.

기존 100문서와 같은 chunk_size=1000, overlap=150으로 오프라인 비교했다.
structured는 직전 v2 인덱스의 청크와, fixed/boundary는 보관된 이전 구현과 비교했다.
새 두 메타데이터 필드만 제외한 **기존 전체 필드·본문·순서·ID·좌표가 일치**하고, 새 필드는 원문·section_of 결과와 별도로 대조했다.
none/project 임베딩 입력도 전부 같았다. 사업명·기관명 매핑과 기본 none, 모델·검색·생성 설정은 바꾸지 않았다.

| 모드 | 청크 수 | 청킹 시간(s) | 표 정보가 있는 청크 | 짧은 표 중 온전히 포함되지 않은 표 | 1,000자 이하 표 행 중간의 청크 경계 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 10,218 | 0.164354 | 8,529 | 2,853 | 4,975 |
| boundary | 12,408 | 0.595243 | 10,157 | 856 | 3,007 |
| structured | 11,844 | 0.549799 | 9,224 | **0** | **0** |

표 총 9,158개에서 ID·헤더·좌표를 확인했다. structured에서 짧은 표 **7,770개 모두 보존**했다.
위 분할 차이는 기존 전략의 차이이며 이번 메타데이터 추가로 새로 개선된 검색 성능을 뜻하지 않는다.
시간은 단일 로컬 청킹 측정이며 API·답변 지연이나 안정적인 속도 비교가 아니다.

structured 전체 청크의 JSON은 이전보다 3,014,198문자 증가했다(ensure_ascii=False 직렬화 기준).
이 중 반복 헤더 원문은 합계 661,637문자다. **문자 수는 토큰 수가 아니다.** 답변은 검색된 청크만 받지만 헤더·좌표로 입력 토큰이 늘 수 있다.
실제 유료 답변 평가·토큰 측정은 실행하지 않았으므로 기존 v2의 답변 점수를 이번 생성 입력의 성능으로 주장하지 않는다.

## 재실행과 적용

오프라인 재검증:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python results/section-table-metadata/validate.py
```

100문서 비교 스크립트·집계는 로컬 results/section-table-metadata/에 Git 제외 상태로 보관한다.
문서 SHA-256: `97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a`.
검증한 embedding.py SHA-256: `f0304fd98cd326a45fc3b463e5937389b1b2c6012904c7388a8774b050c5a6e9`.

현재 인덱스는 수정하지 않았다. 새 메타데이터를 실제 질의에 적용하려면 별도 인덱스 저장 세트를 생성해야 한다.
parse 재실행은 필요 없다. run.py의 documents_sha256 검사는 코드/메타데이터 버전 변경까지 감지하지 않는다.
일반 CLI 적용 명령(이번에는 실행하지 않음):

```bash
RFP_CHUNKING_STRATEGY=structured RFP_EMBEDDING_CONTEXT=none \
  .venv/bin/python run.py build \
  --processed-dir data/processed/chunking-followup-20261006T070744037625Z \
  --index-dir indexes/section-table-metadata \
  --results-dir results/section-table-metadata
```

기본 build는 같은 11,844청크를 다시 임베딩하므로 371회 기본 배치 요청을 사용한다. 이 유료 재빌드는 실행하지 않았다.
이번에는 임베딩 입력이 모두 동일하므로 기존 모델·문맥·본문·벡터 순서를 검증한 캐시 재사용으로 API 없이 별도 인덱스를 구성할 수 있다.
해시만 바꾸거나 기존 인덱스의 chunks.json을 단독 교체하지 않는다.

## 한계

- 표 경계 보장은 structured에서만 제공한다. 기본 fixed/boundary는 메타데이터를 받지만 기존 분할을 유지한다.
- 헤더는 파서의 첫 Markdown 행만 식별한다. 다단 헤더·중간에 바뀐 헤더 전체를 추론하지 않는다. 잘못된/중첩 표 표시는 일반 본문으로 처리한다.
- 메타데이터는 모델 입력으로 전달되지만 답변이 이를 올바르게 활용하는지는 후속 유료 평가가 필요하다. 임베딩에 헤더를 추가한 검색 개선은 이번 범위에 포함하지 않는다.
- 검증 당시 변경 코드는 로컬 미커밋 상태였으며 이 검증 단계에서 커밋·푸시·병합을 수행하지 않았다.
