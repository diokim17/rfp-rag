# 팀 개발 가이드

기준: 2026-10-01, `dev` 커밋 `9b6ae32`. 이후 공통 인터페이스나 실행 방법을 변경하는 PR은 이 문서도 함께 갱신합니다.

목표는 각자 담당 모듈 안에서 성능 개선을 실험하면서도 `run.py`로 연결된 파이프라인이 계속 동작하게 하는 것입니다. 아래 협업 절차는 팀 운영 제안이며, 현재 코드의 동작과 앞으로 추가할 기능은 구분해서 적었습니다.

## 1. 서버에서 바로 시작하기

개인 Langfuse 연결과 자동 평가 요약은 [Langfuse 사용 가이드](LANGFUSE_GUIDE.md)를 참고하세요.

### 처음 한 번: 본인 계정에 프로젝트와 가상환경 만들기

팀원마다 서버 로그인 계정이 다르므로 **각자 계정의 홈 디렉터리에 프로젝트와 `.venv`를 만듭니다.** `~`는 현재 로그인한 계정의 홈 경로입니다. 예를 들어 계정이 `team01`이면 `~/rfp-rag`는 `/home/team01/rfp-rag`입니다.

아직 프로젝트를 내려받지 않았다면 다음을 실행합니다. 비공개 저장소라면 본인 GitHub 계정의 저장소 접근 권한과 서버의 Git 인증이 필요합니다.

```bash
cd ~
git clone --branch dev https://github.com/diokim17/rfp-rag.git rfp-rag
cd ~/rfp-rag
```

이미 본인 계정에 프로젝트가 있다면 clone을 반복하지 않고 해당 폴더로 이동합니다. 다른 경로에 내려받았다면 이후의 `~/rfp-rag`를 실제 프로젝트 경로로 바꿉니다. `requirements-server.lock.txt`가 있는 팀 공유 버전의 코드를 준비한 뒤 다음을 실행합니다.

```bash
cd ~/rfp-rag
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-server.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

기준 환경은 Python 3.12.3 / Linux입니다. `requirements-server.lock.txt`에는 기준 서버의 하위 의존성까지 고정되어 있어 패키지를 개별 선택할 필요 없이 한 번에 설치할 수 있습니다. `requirements.txt`는 직접 의존성 목록이므로 이것만 설치하면 하위 패키지 버전은 달라질 수 있습니다. 버전 고정 파일이 없다면 팀에 해당 파일이 포함된 코드 버전을 요청합니다.

`.venv` 생성과 패키지 설치는 본인의 작업 폴더에서 처음 한 번 필요합니다. 이후에는 아래 활성화만 하면 됩니다. 의존성 목록이 변경되었을 때는 새 목록으로 다시 설치합니다. 기존 `.venv`가 있다면 재생성하기 전에 Python 버전과 설치 상태부터 확인합니다.

`python3.12`가 없거나 `ensurepip`/`venv` 오류가 나면 서버 관리자에게 Python 3.12 및 해당 배포판의 venv 패키지 설치를 요청합니다. 시스템 Python에 `sudo pip install`하지 않습니다. 버전 고정 파일은 Python 자체, 시스템 라이브러리, GPU 환경까지 설치하는 파일은 아닙니다.

새 환경에 `.env`가 없는 경우에만 `.env.example`을 `.env`로 복사하고 승인된 키를 설정합니다. 원본 데이터와 인덱스는 Git에 없으므로 팀에서 정한 비공개 경로로 별도 제공받아 본인의 프로젝트에 준비해야 합니다. 기존 `.env`는 덮어쓰지 않습니다. 가상환경 설치와 API 키·데이터 준비는 별도 작업입니다.

### 이후 접속할 때: 본인의 가상환경 활성화

본인 계정에서 최초 설치를 마쳤다면 새 터미널이나 SSH 접속을 열 때 다음을 실행합니다.

```bash
cd ~/rfp-rag
source .venv/bin/activate
python --version
python -c "import sys; print(sys.executable)"
python -m pip check
python -m unittest discover -s tests -v
```

Python 버전은 기준 환경인 `3.12.3`, 실행 경로는 **본인 프로젝트의 `.venv/bin/python`**이어야 합니다. 예를 들어 `team01` 계정이라면 `/home/team01/rfp-rag/.venv/bin/python`입니다. `source`는 현재 터미널에서 사용할 Python을 선택하는 명령입니다. 종료는 `deactivate`입니다. Conda는 필요 없습니다.

활성화 명령조차 생략하려면 가상환경의 Python을 직접 실행할 수 있습니다.

```bash
cd ~/rfp-rag
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py ask --question "주요 요구사항은 무엇인가요?"
```

두 번째 명령은 본인 프로젝트에 준비된 인덱스와 API 키가 필요하며 API 비용이 발생합니다.

VS Code에서 본인 계정으로 서버에 접속하고 본인의 프로젝트 폴더를 엽니다. **Python: Select Interpreter**에서 해당 프로젝트의 `.venv/bin/python`을 선택합니다. 목록에 없으면 **Enter interpreter path**로 앞에서 확인한 절대 경로를 입력합니다. 인터프리터 선택은 터미널의 `source`와 별개이므로 실행 버튼을 사용할 때도 확인합니다.

### 같은 서버에서 여러 명이 작업할 때

하나의 checkout에서 여러 사람이 동시에 파일을 고치거나 브랜치를 바꾸면 서로의 작업이 바뀝니다. 각자 별도 clone 또는 worktree에서 작업하고, 개인 실험용 패키지는 개인 `.venv`에 설치합니다. 공용 `.venv`는 실험용 `pip install`이나 업그레이드로 바꾸지 않습니다. 공용 환경을 변경할 때는 통합 담당자와 조율하고 실행 중인 작업을 확인합니다.

## 2. 담당 파일과 수정 범위

흐름: `parsing → embedding → retrieval → generation → evaluation`, 연결과 명령 실행은 `run.py`가 담당합니다.

| 담당 | 주 파일 | 내부에서 실험할 수 있는 내용 | 유지할 연결 지점 |
| --- | --- | --- | --- |
| 김도영 / 통합 | `run.py` | CLI, 설정 전달, 파이프라인 연결 | 기존 명령과 모듈 호출, 결과 저장 |
| 나상훈 / 파싱 | `parsing.py` | 추출 품질, 정제, OCR·표 처리 확장 | Document, CSV 매핑, 오류 기록 |
| 유찬혁 / 청킹·임베딩 | `embedding.py` | 청킹 전략, 배치 처리, 임베딩 실험 | Chunk, 벡터 순서, 인덱스 저장·로드 |
| 김연주 / 검색 | `retrieval.py` | 검색 전략, 메타데이터 필터, 리랭킹 | Hit 목록, top_k, 필터 동작 |
| 박단비 / 생성 | `generation.py` | 프롬프트, 근거 사용, 답변 품질 | Answer, sources와 인용 번호 |
| 김시현 / 평가 | `evaluation.py` | 정답셋, 평가 지표, 비교 보고서 | evaluate 반환 구조, 정답 문서 ID |

OCR, 리랭킹, 대화 기억, 시나리오 A는 현재 구현되지 않았습니다. 추가할 때도 기존 baseline 실행 경로를 유지합니다. 담당 파일 외 수정이 필요하면 PR에 이유와 영향을 적고 관련 담당자와 함께 검토합니다. 함수 내부 보조 함수나 선택 옵션을 추가할 수 있지만 기존 호출이 그대로 동작해야 합니다.

## 3. 반드시 유지할 입출력 계약

기본 자료형은 JSON으로 저장 가능한 `dict` / `list`입니다. 기존 필수 키를 삭제·변경하거나 dataclass, DataFrame 등으로 반환 형식을 바꾸지 않습니다. 필드 추가도 JSON 직렬화와 기존 소비 코드에 영향이 없는지 확인합니다.

```python
Document = {
    "doc_id": "문서 ID 문자열",
    "text": "추출 본문 문자열",
    "metadata": {"filename": "문서.hwp", "source": "files/문서.hwp", "발주 기관": "기관명"},
}
Chunk = {
    "chunk_id": "문서ID:0", "doc_id": "문서 ID 문자열", "text": "청크 본문",
    "metadata": {"filename": "문서.hwp", "source": "files/문서.hwp", "발주 기관": "기관명",
                 "start_char": 0, "end_char": 1000},
}
Hit = {**Chunk, "score": 0.85}
Answer = {
    "question": "질문", "answer": "근거를 포함한 답변 [1]",
    "sources": [{"citation": 1, **Hit}], "model": "모델명",
}
```

위 값은 형식을 보여주는 예시입니다. 실제 청크의 `text`는 원문 `text[start_char:end_char]`에 대응해야 합니다.

| 함수 | 현재 호출 형식 | 반환 |
| --- | --- | --- |
| `parse_documents` | `(raw_dir, output_dir, limit=None)` | Document 목록 및 documents/errors JSON 저장 |
| `chunk_documents` | `(documents, chunk_size=1000, chunk_overlap=150)` | Chunk 목록 |
| `embed_texts` | `(texts, client, model)` | 입력 순서와 같은 정규화된 float32 벡터 |
| `build_index` | `(documents, client, index_dir="indexes", model="text-embedding-3-small", chunk_size=1000, chunk_overlap=150)` | config dict 및 인덱스 파일 저장 |
| `load_index` | `(index_dir="indexes")` | `(index, chunks, config)` 튜플 |
| `retrieve` | `(question, client, index, chunks, config, top_k=5, filters=None)` | Hit 목록 |
| `generate_answer` | `(question, hits, client, model="gpt-5-mini")` | Answer dict |
| `evaluate` | `(cases, answer_fn)` | `{"summary": {...}, "records": [...]}` |

`parse_documents`의 기본 경로는 `data/raw`, `data/processed`입니다. `run.py`는 여러 함수를 위치 인자로 호출하므로 이름뿐 아니라 인자 순서도 유지합니다. `read_json(path)`와 `write_json(path, value)`도 여러 모듈에서 사용합니다. `answer_fn(question, filters)`는 Answer를 반환해야 합니다.

### 파싱·청킹 규칙

- `doc_id`는 NFC 정규화 파일명의 SHA-256 앞 16자리입니다. 같은 파일명은 같은 ID를 유지하며 내용 변경을 ID로 감지하지 못합니다. 파일명이나 ID 규칙을 바꾸면 평가 정답도 영향을 받습니다.
- `metadata`는 CSV 열 이름과 문자열 값을 유지합니다. `텍스트` 열은 제외하고 실제 원본에서 본문을 추출합니다. `filename`, `source`를 유지해야 출처 출력이 동작합니다.
- `chunk_id`는 현재 `doc_id:순번`입니다. 청크마다 고유해야 하며 원문 문서의 `doc_id`를 유지합니다. 문자 위치를 토큰 위치로 바꾸거나 가짜 위치를 넣지 않습니다. 원문에 대응하지 않는 청킹 전략은 위치 규약부터 협의합니다.
- 청킹 실험은 `RFP_CHUNKING_STRATEGY=fixed|boundary|structured`로 고릅니다. structured의 기본 버전은 기존 v2이며 v3는 `RFP_STRUCTURED_CHUNKING_VERSION=3`으로 명시할 때 사용합니다. fixed와 boundary는 버전 1을 유지합니다.
- 임베딩 입력 문맥은 `RFP_EMBEDDING_CONTEXT=none|project|project_blend|table|section`으로 선택하며 기본은 `none`입니다. `project`는 사업 문맥을 붙이고 `project_blend`는 본문/문맥 벡터를 80:20으로 결합합니다. `table`과 `section`은 표 헤더·절 경로를 문서 임베딩 입력에만 추가합니다.
- 선택 설정마다 별도 인덱스를 빌드하고 같은 `--index-dir`를 질의·평가에 사용합니다. `config.json`의 청킹 전략·버전과 임베딩 문맥이 조건 재현 기준입니다. 옵션은 `run.py` 기본값과 Document → Chunk → Hit → Answer 계약을 변경하지 않습니다.
- 환경변수 변경은 이미 빌드된 인덱스를 바꾸지 않습니다. 기존 인덱스를 재사용하면 build하지 말고, 다른 옵션을 적용하려면 별도 경로로 새로 build합니다.
- `limit`은 성공 문서 수가 아니라 CSV 앞쪽 N행입니다. 개별 실패는 `parsing_errors.json`에 남기며, 성공 문서가 하나도 없으면 실패합니다.

### 인덱스·검색 규칙

- 현재 저장 세트는 `index.faiss`, `chunks.json`, `config.json`입니다. FAISS 벡터 i번과 chunks i번이 같은 청크여야 합니다.
- config 필수 키는 `embedding_model`, `dimension`, `chunk_count`, `chunk_size`, `chunk_overlap`입니다. 로드할 때 벡터 수·청크 수·차원을 검증합니다.
- 현재는 L2 정규화 벡터 + `IndexFlatIP`로 코사인 유사도를 구합니다. 질문은 `.env`의 새 모델이 아니라 **저장된 config의 임베딩 모델**로 임베딩합니다. 같은 차원이라도 서로 다른 모델의 벡터를 섞으면 안 됩니다.
- `retrieve`는 관련도가 높은 순서로 최대 `top_k`개를 반환합니다. `score`는 Python float이며 현재는 코사인 유사도입니다. 리랭킹 점수로 의미를 바꾸려면 소비 코드·평가와 협의하고 새 필드 또는 점수 종류 기록을 고려합니다.
- 필터는 metadata의 NFC 정규화 문자열 정확 일치입니다. 정확히 일치하는 값이 없으면 공백·기호와 '서울특별시'/'서울시' 차이를 무시하고 그 값을 포함하는 metadata 값이 하나뿐일 때 그 값으로 거릅니다(예: `한국철도공사` → `한국철도공사 (용역)`). 조건을 만족하는 결과가 없으면 `[]`입니다. 리랭킹을 넣어도 필터 밖의 문서를 다시 포함하지 않습니다.

### 생성·평가 규칙

- `sources`는 전달된 검색 결과 전체에 1부터 붙인 `citation` 목록입니다. 모델이 실제 인용한 출처만의 목록이 아닙니다. 평가가 이를 검색 결과로 사용하므로 임의로 줄이지 않습니다.
- 답변의 `[1]` 등은 sources의 번호와 대응해야 합니다. 검색 결과가 비어 있으면 API를 호출하지 않고 근거가 없다는 답변을 반환합니다.
- 문서 속 명령을 따르지 않고 근거 없는 내용을 단정하지 않는 프롬프트 원칙을 유지합니다. import만 했을 때 API 호출, 키 로드, 파일 변경이 일어나지 않도록 합니다.
- 평가 사례는 `question`, 비어 있지 않은 `expected_doc_ids`, 선택 `expected_keywords`, 선택 `filters`를 갖습니다. 실제 문서를 확인해 정답을 작성합니다.
- 현재 지표는 문서 기준 Recall@k, 키워드 포함률, 평균 응답 시간입니다. 키워드 미지정 점수는 `null`이며, 키워드 포함률만으로 사실성을 주장하지 않습니다. 기존 summary/record 키는 유지하고 새 지표를 추가합니다.

## 4. 공통 부분을 바꾸는 절차

다음은 개인 실험 때문에 공용 상태에서 바로 바꾸지 않습니다: `run.py`의 기본 실행 설정, 공개 함수의 시그니처·필수 필드, 문서 ID 규칙, 인덱스 파일 형식, 공용 `.env`와 `.venv`, 공용 데이터·인덱스, 팀 공통 의존성 버전.

계약 변경이 꼭 필요하면 ① 변경 이유와 영향 모듈을 PR에 적고 ② 관련 담당자와 새 규약을 정하고 ③ 호출 코드·테스트·문서와 기존 산출물의 재생성 방법을 함께 변경합니다. 공통 통합은 김도영 담당자 검토 후 `dev`에 반영하는 방식으로 운영할 것을 권장합니다.

## 5. 실험은 저장 경로를 분리해서 실행

기본 `data/processed`, `indexes`를 개인 실험으로 덮어쓰지 않습니다. 아래 `retrieval-yj-001`을 본인의 고유 실험 ID로 바꾸세요. 모든 후속 명령에 같은 경로를 전달합니다.

```bash
python run.py parse --limit 3 --processed-dir data/processed/retrieval-yj-001
python run.py build --processed-dir data/processed/retrieval-yj-001 --index-dir indexes/retrieval-yj-001 --chunk-size 1000 --chunk-overlap 150
python run.py ask --index-dir indexes/retrieval-yj-001 --results-dir results/retrieval-yj-001 --question "주요 요구사항은 무엇인가요?" --top-k 5
python run.py evaluate --index-dir indexes/retrieval-yj-001 --results-dir results/retrieval-yj-001 --eval-file data/eval.json --top-k 5
```

`data/eval.json`은 원문을 보고 팀이 작성해야 하는 파일이며 현재 자동 제공되지 않습니다. build/ask/evaluate는 API 비용이 발생합니다. 먼저 소수 문서로 연결을 확인하고 규모를 늘립니다. `build`는 해당 processed 폴더의 documents.json 전체를 사용합니다.

검색·생성만 바꾸는 실험은 baseline 인덱스를 **읽기 전용으로 재사용**할 수 있습니다. 같은 인덱스 폴더에 build와 ask를 동시에 실행하지 않습니다. 세 파일 저장은 원자적 교체가 아니므로 빌드 중 읽으면 불일치가 생길 수 있습니다.

| 변경 | 필요한 재실행 |
| --- | --- |
| 원본·CSV·파싱·정제 | 실험 폴더에 parse → build → ask/evaluate |
| 청킹·임베딩 모델·차원·정규화·인덱스 구조 | build → ask/evaluate; 새 인덱스 폴더 사용 |
| 검색 top_k·필터·기존 인덱스를 쓰는 리랭킹 | ask/evaluate; 검색 구조까지 변경하면 build도 필요 |
| 생성 프롬프트·답변 모델 | ask/evaluate |
| 정답셋·평가 지표 | evaluate |

`python run.py`처럼 인자 없이 실행하면 기본 indexes의 세 파일 존재 여부만으로 ask/all을 선택합니다. 데이터 최신 여부나 모델 일치까지 판단하지 않습니다. 파일이 없으면 all이 실행되어 기본 processed/indexes를 생성하고 API 비용이 듭니다. 개인 실험에서는 명령과 경로를 명시하고 `DEFAULT_COMMAND`, `DEFAULT_LIMIT`, `DEFAULT_QUESTION` 변경을 실험 PR에 섞지 않습니다.

## 6. 브랜치·PR·검증

개인 checkout에서 변경 사항이 없는지 먼저 확인하고 최신 dev에서 역할별 브랜치를 만듭니다. 아래 이름은 예시입니다.

```bash
git status --short
git switch dev
git pull --ff-only origin dev
git switch -c feature/retrieval-rerank-yj
```

작업 중인 변경이 있으면 무리하게 브랜치를 전환하거나 reset하지 않습니다. 다른 사람이 작업하는 공용 checkout에서도 위 명령을 실행하지 않습니다. PR 대상은 `dev`로 하고 담당 파일과 필요한 연동 변경만 포함합니다.

병합 전 최소 확인:

```bash
python -m pip check
python -m unittest discover -s tests -v
git diff --check
git diff --stat
git status --short
```

현재 오프라인 테스트는 청킹 위치, 인덱스→검색→생성→평가 연결, 빈 검색 결과, 평가 정답 검증과 선택적 추적·본문 제외·공유 요약을 확인합니다. 실제 HWP/PDF 추출 품질이나 실제 API 호출 성공까지 보장하지 않습니다. 수정한 기능의 경계 조건 테스트를 추가하고, 필요하면 별도 실험 폴더에서 소규모 실제 실행 결과도 확인합니다. 기존 테스트를 삭제하거나 검증을 약하게 만들어 통과시키지 않습니다.

PR에는 다음을 기록합니다.

- 변경 이유·담당 파일·공통 계약 변경 여부.
- baseline과 변경 후 결과: 같은 원본 범위, 정답셋, top_k 등 비교 조건 및 차이.
- 실행 명령, 코드 커밋, 청킹·모델·검색·생성 설정, 정답셋 버전, 실행 시각.
- 품질·응답 시간·비용 및 실패 사례, 실행한 테스트 결과.
- 새 패키지와 버전, 인덱스 재생성 여부, 기존 방식으로 돌아가는 방법.

원본, 전처리 본문, chunks, 원문을 포함한 결과 JSON, `.env`, API/SSH 키, `.venv`는 커밋하지 않습니다. `git add .`로 전부 넣기 전에 내용을 확인하고 파일을 명시해 추가합니다. Git 제외 규칙만 믿지 말고 새 경로의 추적 여부도 확인합니다. 공유용 요약은 원문·민감정보를 제거한 `results/reports/*.md` 또는 `*.csv`에 저장할 수 있습니다.

새 의존성이 필요하면 개인 환경에서 검증하고 `requirements.txt`를 갱신합니다. 통합 담당자는 깨끗한 검증 환경에서 설치·테스트 후 서버 스냅샷도 갱신합니다. 개인 환경의 불필요한 패키지가 섞인 `pip freeze`를 그대로 공통 목록으로 덮어쓰지 않습니다.

## 7. 팀에서 추가로 확정할 사항

| 항목 | 필요한 합의 / 보완 |
| --- | --- |
| 공통 평가셋 | 담당자, 저장 위치·배포 방법, 버전 고정, 실험용 질문과 최종 평가 질문 분리 |
| baseline 기록 | 고정된 코드 커밋·데이터 범위·설정으로 최초 지표 측정. 현재 README 결과표는 미작성 |
| 병합 기준 | 리뷰 담당자, 최소 테스트, 허용할 품질 저하와 지연·비용 기준; CI·브랜치 보호는 별도 설정 필요 |
| 서버 작업 구역 | 개인 checkout/환경 경로, 공용 인덱스 갱신 담당자와 시간, 읽기 권한·데이터 배포 경로 |
| API 예산 | 대규모 build/evaluate 사전 공유 기준과 사용량 확인 담당자. 현재 코드는 누적 비용 제한을 자동 집행하지 않음 |
| 실패 분석 | 파싱 실패율·표/스캔 문서 누락, 필터 검색 누락, 답변 근거 적합성을 공통 항목으로 기록 |
| 실험 추적 | 실행 결과와 Langfuse에 데이터/인덱스/평가셋 해시·코드 커밋이 기록됨. 상세 변경 설명과 비교 조건은 보고서에 보완. JSON 파일(documents·chunks·config·평가셋) 해시는 줄바꿈을 LF로 맞춰 계산해 Windows·macOS·Linux에서 같고, index.faiss는 바이트 그대로 계산 |
| 복구·호환성 | 검증된 baseline 인덱스 보존, 새 형식의 schema version과 마이그레이션, 원자적 저장/잠금은 향후 구현 |

위 보완 기능은 이 문서 작성으로 구현된 기능이 아닙니다. 우선 공통 평가셋, baseline 측정, 개인 작업 폴더, 병합 검토 기준을 확정하면 팀원이 같은 기준으로 실험을 비교할 수 있습니다.
