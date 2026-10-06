# documents_sha256 저장 및 CLI 연동 검증

2026-10-06 UTC, 작업 브랜치 `feature/chunking-project-context`.

## 작업 순서와 변경

사용자 요청에 따라 기존 청킹·문맥 임베딩 작업을 `704986a`로 커밋하고,
`git fetch origin` 후 `origin/feature/index-check`를 병합했다.
병합 커밋은 `c8c8ea1`이며 충돌은 없었다. 아래 검증은 해시 저장 구현의 커밋·푸시 전에 수행했다.

- `build_index`의 기존 인자 뒤에 `documents_sha256=None`을 추가했다.
- 전달값을 재계산·검증·정규화하지 않고 반환 config와 저장 `config.json`에 동일하게 기록한다.
- 인자를 생략한 기존 호출은 유지되며 값은 Python `None` / JSON `null`이다.
- Document → Chunk → Hit → Answer, 청크 본문·좌표·ID, 벡터 순서, 기존 config 필수 키와 저장 파일 3종은 유지한다.
- 청킹·문맥 모드 기본값, 모델·배치·정규화·검색 설정, 의존성, 실험 ID 생성, 추적 구현은 변경하지 않았다.
- 이번 구현 단계에서 `run.py`는 추가 수정하지 않았다. 최신성 검사는 병합된 코드가 담당한다.

## 호출 관계와 문서 차이

build/all은 `run.py`가 `documents.json`을 읽은 동일 바이트로 JSON을 해석하고 SHA-256을 계산한다.
그 값을 `build_index`에 전달하고 청킹 → 임베딩 → 저장 이후 저장 config를 검사한다.
all은 파싱 이후 새 파일을 읽는다. ask/evaluate는 API 클라이언트 생성 전에 해시를 검사하며,
통과하면 load_index → retrieve → generate_answer(→ evaluate)로 진행한다.

팀 가이드의 이전 시그니처와 “embedding 담당 구현 대기”를 완료된 계약으로 갱신했다.
Langfuse 가이드의 별도 인덱스 명령 예시에 짝이 맞는 `--processed-dir`를 추가했다.
기존 청킹 실험의 선택 모드·파싱 sections 확장은 이전 청킹 보고서에 설명되어 있으며 이번 작업의 변경 대상은 아니다.

## 검증 결과

- 병합 전 기존 테스트 62개 통과.
- 병합된 최신성 테스트 7개와 이번 신규 테스트 6개를 포함해 **75개 통과**. unittest 실행 시간 0.359초(전체 프로세스 시간 아님).
- `.venv/bin/python -m unittest discover -s tests -v`, `.venv/bin/python -m pip check`, `git diff --check` 통과.
- 받은 값 그대로 저장·반환·로드, 생략 시 null, 기존/추가 위치 인자 호출, 해시 인자로 청크·벡터 파일이 바뀌지 않음을 검증했다.
- 실제 build → load → ask 연결과 parse mock → 실제 build → all 연결을 가짜 모델 클라이언트로 실행했다.
- CLI evaluate fixture의 수동 config 보정을 제거하고 실제 build_index 저장 결과로 기존 평가·보고서 테스트를 통과했다.
- 구형 config의 load_index 호환, 해시 누락·잘못된 형식·불일치·파일 누락 차단 및 추적 본문 제외를 확인했다.
- JSON 내용이 같아도 파일 바이트가 달라지면 재빌드가 요구됨을 확인했다.

테스트 파일은 임시 폴더에만 저장했다. 실제 데이터·기존 인덱스·개인 환경 설정은 수정하지 않았다.
외부 API 호출 **0회**, 유료 모델 토큰 **0개**다. 실제 문서 검색 품질·토큰·응답 시간의 성능 비교는 수행하지 않았다.

## 재실행과 한계

해시 없는 기존 인덱스도 load_index로 읽을 수 있지만 CLI 질의는 중단된다.
해시가 없거나 현재 파일과 다르면 명시적으로 build해야 하며 자동 재빌드는 없다.
기존 config에 현재 해시만 붙이지 않는다. 최신 documents.json이 이미 있으면 parse 재실행은 불필요하다.

아래 경로를 실제 개인 문서 폴더와 새 인덱스 폴더로 바꾸고, 비교 대상의 청킹·문맥·모델·크기·중복 설정을 유지한다.
build와 ask는 유료 호출을 발생시키며 이번 검증에서는 실행하지 않았다.

```bash
.venv/bin/python run.py build --processed-dir '<문서폴더>' --index-dir '<새인덱스폴더>'
.venv/bin/python run.py ask --processed-dir '<문서폴더>' --index-dir '<새인덱스폴더>' --results-dir '<별도결과폴더>' --question '주요 요구사항은 무엇인가요?'
```

해시는 문서 파일 바이트의 일치만 확인한다. 청킹 구현·모델 설정 변경 여부나 여러 파일의 원자적 저장을 보장하지 않는다.
동일 인덱스에 빌드와 질의를 동시에 실행하지 않는 기존 운영 규칙은 유지된다.
