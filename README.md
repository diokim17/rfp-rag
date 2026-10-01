# RFP-RAG

> 코드잇 AI 중급 프로젝트 · 3팀
>
> RFP(제안요청서) 문서 기반 정보 검색·요약 및 질의응답 시스템

## 일정

| 구분 | 일정 |
| --- | --- |
| 프로젝트 진행 기간 | 26.09.29 ~ 26.10.22 |

## 프로젝트 개요

공공입찰 컨설팅 서비스 ‘입찰메이트’의 사내 RAG 시스템 구축을 주제로 진행하는 프로젝트입니다. 컨설턴트가 수십 페이지의 제안요청서를 일일이 읽는 부담을 줄이고, 고객사에 적합한 입찰 기회를 검토하는 데 필요한 정보를 제공하는 것을 목표로 합니다.

제공된 RFP 문서 100개와 메타데이터를 활용하여 문서 처리, 청킹, 임베딩, 검색, 답변 생성 및 성능 평가를 진행합니다.

| 항목 | 내용 |
| --- | --- |
| 입력 데이터 | RFP 문서(HWP, PDF), `data_list.csv` 메타데이터 |
| 주요 기능 | 문서 기반 정보 검색, 핵심 내용 요약, 질의응답, 대화 맥락 유지 |
| 비교 실험 | GCP에서 직접 실행하는 모델과 LLM API 기반 방식 |
| 최종 산출물 | 재현 가능한 코드, 분석 보고서, 개인별 협업일지 |

## 팀원 및 역할

| 담당 역할 | 이름 | GitHub | 협업일지 |
| --- | --- | --- | --- |
| PM / 통합 | 김도영 | [@diokim17](https://github.com/diokim17) | [협업일지](https://docs.google.com/document/d/1UcFhq4vEb1SL2uDldYpeGibJ07wi1n34tP9XeckNyWs/edit?tab=t.ka3ijil7t5i2) |
| 파싱 / 전처리 | 나상훈 | [@Na-SangHun](https://github.com/Na-SangHun) | [협업일지](https://docs.google.com/document/d/1UcFhq4vEb1SL2uDldYpeGibJ07wi1n34tP9XeckNyWs/edit?tab=t.4bl1rvpraiza#heading=h.qv2phv602y3p) |
| 청킹 / 임베딩 | 유찬혁 | [@jins091125-gif](https://github.com/jins091125-gif) | [협업일지](https://docs.google.com/document/d/1UcFhq4vEb1SL2uDldYpeGibJ07wi1n34tP9XeckNyWs/edit?tab=t.g14lnaobk5xt#heading=h.xrtl9g9gkphg) |
| 검색 고도화 | 김연주 | [@AIengineeung](https://github.com/AIengineeung) | [협업일지](https://docs.google.com/document/d/1UcFhq4vEb1SL2uDldYpeGibJ07wi1n34tP9XeckNyWs/edit?tab=t.sj1bxo9tvmqa#heading=h.qo3mm8oq6yi) |
| 생성 / LLM | 박단비 | [@DBDBDEEP02](https://github.com/DBDBDEEP02) | [협업일지](https://docs.google.com/document/d/1UcFhq4vEb1SL2uDldYpeGibJ07wi1n34tP9XeckNyWs/edit?tab=t.u2cla8e84xyr#heading=h.tg7gi3kx1lvh) |
| 평가 | 김시현 | [@dypower1559-cell](https://github.com/dypower1559-cell) | [협업일지](https://docs.google.com/document/d/1UcFhq4vEb1SL2uDldYpeGibJ07wi1n34tP9XeckNyWs/edit?tab=t.lijv4uvh1zh6#heading=h.qi25fkhl73oj) |

### 데일리 스크럼

[데일리 스크럼 일지](https://docs.google.com/document/d/1UcFhq4vEb1SL2uDldYpeGibJ07wi1n34tP9XeckNyWs/edit?tab=t.ka3ijil7t5i2)

매일 10~15분 동안 어제 한 일, 오늘 할 일, 진행을 막는 이슈(Blocker)를 공유합니다.

## 개발 환경 및 기술 스택

시나리오 B의 최소 baseline을 먼저 구현했습니다. 팀 목표 환경은 Python 3.12 / GCP Linux이며, 로컬에서도 실행할 수 있습니다. 시나리오 A는 이후 확장 대상입니다.

| 구분 | 시나리오 A: GCP 모델 직접 실행 | 시나리오 B: OpenAI API 기반 |
| --- | --- | --- |
| 개발 언어 및 버전 | Python 3.12.3 | Python 3.12.3 |
| 실행 환경 | GCP 서버 / Linux | GCP 서버 / Linux |
| 문서 파싱 | 작성 예정 | pyhwp(HWP 5.x), pypdf(PDF) |
| 임베딩 모델 | 미정 | text-embedding-3-small |
| Vector DB | 미정 | FAISS IndexFlatIP (코사인 유사도) |
| LLM | 미정 (GCP에서 직접 실행) | gpt-5-mini (Responses API) |
| 주요 라이브러리 | 프로젝트 종료 후 작성 | requirements.txt 참조 |

## 프로젝트

### 프로젝트 구조

역할별로 하나의 Python 파일을 담당하고, `run.py`에서 전체 흐름을 연결합니다. 아래는 현재 구성입니다. 테스트 코드는 `tests/test_pipeline.py`에 있습니다.

```text
rfp-rag/
├── data/                        # 데이터 저장 (데이터 파일은 Git 제외)
│   ├── raw/                     # 원본 RFP 문서(HWP·PDF) 및 메타데이터
│   │   ├── .gitkeep             # 빈 폴더 구조 유지용
│   │   ├── data_list.csv        # 파일명 및 사업 메타데이터 (Git 제외)
│   │   └── files/               # 원본 HWP·PDF 파일 (Git 제외)
│   └── processed/               # 텍스트 추출·정제 및 메타데이터 결합 결과
│       └── .gitkeep             # 빈 폴더 구조 유지용
├── indexes/                     # 청킹·임베딩으로 생성한 검색 인덱스 (Git 제외)
├── results/                     # 모델별 평가 결과 및 A/B 비교 실험 기록
├── run.py                       # 김도영: 전체 파이프라인 연결 및 질의응답 실행
├── parsing.py                   # 나상훈: HWP·PDF 추출, 정제, 메타데이터 결합
├── embedding.py                 # 유찬혁: 청킹, 임베딩 생성, 벡터 DB 구축
├── retrieval.py                 # 김연주: 검색, 메타데이터 필터링, 리랭킹
├── generation.py                # 박단비: 프롬프트 구성, OpenAI·GCP 모델 호출
├── evaluation.py                # 김시현: 평가 데이터 구성 및 성능 평가
├── .env                         # API 키 등 환경 변수 (Git 제외)
├── .env.example                 # 실제 키가 없는 환경 변수 설정 예시
├── .gitignore                   # 데이터·환경 변수 등 Git 제외 규칙
├── requirements.txt             # 팀 공통 패키지 및 버전 목록
└── README.md                    # 프로젝트 소개 및 실행 안내
```

각 파일의 공통 입력·출력 형식을 먼저 정한 뒤, 해당 형식을 유지하면서 내부 구현을 실험합니다. 파일 간 연결과 통합은 `run.py`에서 관리합니다.

현재 `generation.py`는 시나리오 B의 OpenAI 호출만 구현하며, 이후 시나리오 A를 추가할 수 있습니다. 청킹·임베딩 비교 실험은 `embedding.py`에서 진행합니다. 청킹 방식이나 임베딩 모델을 변경하면 인덱스를 다시 생성하며, 문서와 검색 질문에는 동일한 임베딩 모델을 사용합니다.

### 성능 평가

| 실험 | 주요 설정 | 평가 지표 및 결과 | 응답 시간 | 비용 |
| --- | --- | --- | --- | --- |
| 베이스라인 | 작성 예정 | 작성 예정 | 작성 예정 | 작성 예정 |
| 개선 모델 | 작성 예정 | 작성 예정 | 작성 예정 | 작성 예정 |

### 결론 및 개선 사항

- 주요 성과: 작성 예정
- 핵심 의사결정과 근거: 작성 예정
- 한계 및 개선 방향: 작성 예정
- 회고 및 멘토링 피드백: 작성 예정


## 실행 방법

프로젝트 루트에서 실행합니다. Python 3.12 이상을 사용하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

기존 `.env`를 유지하세요. 새 환경에서만 `.env.example`을 복사한 뒤 `OPENAI_API_KEY`를 설정합니다. 선택 변수는 `OPENAI_EMBEDDING_MODEL`과 `OPENAI_GENERATION_MODEL`이며, 생략하면 위 표의 모델을 사용합니다. 실행 시 프로그램이 키를 로드하고 출력하지 않습니다.

CSV는 UTF-8(또는 UTF-8 BOM), 원본 파일은 `data/raw/files/`에 둡니다. `파일명` 열로 연결하며 한글 파일명은 NFC 정규화 후 비교합니다. CSV의 `텍스트` 열은 사용하지 않고 원본에서 추출합니다.

### VS Code 실행 버튼으로 실행

프로젝트의 `.venv` Python 인터프리터를 선택하고 `run.py`를 연 뒤 오른쪽 위 **Run Python File(▶)** 버튼을 누르세요. 현재 기본 실행은 `ask`이며 저장된 인덱스로 검색 → 답변 생성을 실행합니다. 처음 실행하여 인덱스가 없다면 `DEFAULT_COMMAND = "all"`로 설정하여 문서 3개로 전체 과정을 실행하세요.

질문과 문서 수는 `run.py` 상단의 `DEFAULT_QUESTION`, `DEFAULT_LIMIT`에서 변경합니다. `DEFAULT_LIMIT`은 `parse`/`all`에서 적용하며 `None`이면 전체 문서를 처리합니다. `DEFAULT_COMMAND = "all"`은 실행할 때마다 인덱스를 다시 생성하며 OpenAI API 비용이 발생합니다. 인덱스 생성 이후에는 `DEFAULT_COMMAND = "ask"`로 변경하세요.

답변 생성 중 429 오류에 `RPM: Limit 0`이 표시되면 해당 프로젝트의 모델 요청 한도가 0인 상태입니다. OpenAI 프로젝트의 Limits를 확인하거나, 사용 가능한 모델명을 `.env`의 `OPENAI_GENERATION_MODEL`에 설정하세요. 답변 모델만 변경할 때는 인덱스를 다시 만들 필요가 없습니다.

### 터미널에서 실행하는 경우

```bash
python run.py all --limit 3 --question "한영대학교 교육환경 구축 사업의 주요 요구사항은 무엇인가요?"
```

`all`은 전처리·인덱스를 다시 생성하고 질의합니다. `--limit`은 CSV 앞쪽 N행을 의미하며, 생략하면 전체를 처리합니다. OpenAI 임베딩 및 답변 API 호출에는 비용이 발생합니다.

### 단계별 실행 및 기존 인덱스 재사용

```bash
python run.py parse --limit 3
python run.py build
python run.py ask --question "주요 요구사항을 알려주세요." --top-k 5
python run.py ask --question "사업 범위는 무엇인가요?" --filter "발주 기관=한영대학"
```

`parse`는 API나 `.env` 없이 실행됩니다. `build`는 저장된 모든 전처리 문서를 사용합니다. `ask`는 원문을 재임베딩하지 않고 저장된 인덱스와 동일한 모델로 질문만 임베딩합니다. 필터는 CSV 메타데이터의 정확 일치이며 여러 번 지정할 수 있습니다.

- `data/processed/documents.json`: 추출 본문과 메타데이터
- `data/processed/parsing_errors.json`: 파일별 누락·추출 오류 (성공한 문서로 계속 진행)
- `indexes/index.faiss`, `chunks.json`, `config.json`: 벡터, 청크, 임베딩 모델·청킹 설정
- `results/ask_*.json`, `all_*.json`, `evaluate_*.json`: 답변·검색 근거·설정·평가 기록

데이터·청킹·임베딩 모델을 바꿨다면 `parse`/`build`를 다시 실행하세요. 인덱스 생성 중에는 같은 폴더로 질의하지 마세요. 실험별 보관은 `--processed-dir`, `--index-dir`, `--results-dir`로 경로를 분리할 수 있습니다.

### 모듈 간 입출력 약속

모든 데이터는 기본 `dict`/`list`이며 저장 형식은 JSON입니다. 팀원들은 아래 키를 유지하면서 내부 구현을 개선합니다.

| 모듈 / 함수 | 입력 | 출력 |
| --- | --- | --- |
| parsing / parse_documents | 원본 폴더, 출력 폴더, limit | Document 목록 |
| embedding / chunk_documents | Document 목록, 청크 크기·중복 길이 | Chunk 목록 |
| embedding / build_index | Document 목록, OpenAI client, 설정 | 저장된 FAISS 파일 + 설정 dict |
| embedding / load_index | 인덱스 폴더 | `(index, chunks, config)` |
| retrieval / retrieve | 질문, client, index, chunks, config, top_k, filters | Hit 목록 |
| generation / generate_answer | 질문, Hit 목록, client, 모델명 | Answer |
| evaluation / evaluate | 평가 사례 목록, answer_fn | summary, records |

```text
Document = {doc_id: str, text: str, metadata: dict}
Chunk    = {chunk_id: str, doc_id: str, text: str, metadata: dict}
Hit      = Chunk + {score: float}
Answer   = {question: str, answer: str, sources: list, model: str}
```

`doc_id`는 NFC 파일명 기반 해시이고 `chunk_id`는 `doc_id:순번`입니다. `metadata`는 CSV 열 이름과 문자열 값을 유지하고 `filename`, `source`를 추가합니다. 청크에는 `start_char`, `end_char`를 추가합니다. `score`는 코사인 유사도이며 확률이 아닙니다. `sources`는 검색 근거에 1부터 시작하는 `citation`을 추가한 목록이며, 답변의 `[1]`과 연결됩니다. 모델이 실제 인용한 근거만 추린 목록은 아닙니다.

### 기본 평가

EDA 및 정답 작성 전이므로 실제 성능을 주장하는 평가 데이터는 만들지 않았습니다. `documents.json`의 `doc_id`와 원문을 확인한 뒤 `data/eval.json`을 수동 작성하세요.

```json
[
  {
    "question": "해당 사업의 예산은 얼마인가요?",
    "expected_doc_ids": ["실제 정답 문서의 doc_id로 교체"],
    "expected_keywords": ["원문에서 확인한 예산 표현"]
  }
]
```

```bash
python run.py evaluate --eval-file data/eval.json --top-k 5
python -m unittest discover -s tests -v
```

평가는 문서 기준 Recall@k(정답 문서 중 검색된 비율), 답변 키워드 포함률, 평균 응답 시간을 저장합니다. `expected_keywords`를 생략하면 해당 점수는 null이며, 키워드 포함률은 답변의 사실성 평가가 아닙니다. 사례별 `filters`도 선택적으로 지정할 수 있습니다. 테스트는 외부 API와 `.env`를 사용하지 않습니다.

### 현재 범위

프로젝트 제공 API는 답변 모델 `gpt-5-mini`, `gpt-5-nano`와 임베딩 모델 `text-embedding-3-small`을 허용합니다. 기본 답변 모델은 `gpt-5-mini`이며 추론 강도는 `low`, 추론을 포함한 출력 한도는 2,000토큰입니다. 팀별 총 사용 한도는 $20이며, 팀 채널의 `!usage`로 사용량을 확인합니다(1분에 1회). 코드가 팀의 누적 사용액을 추적하거나 $20에서 자동 중단하는 것은 아닙니다.

문자 기준 1,000자 청킹/150자 중복, 벡터 검색, 단일 질문 답변으로 연결을 확인하는 baseline입니다. OCR, HWPX, 표 구조 보존, 리랭킹, 대화 기억, 자동 정답 생성, 시나리오 A는 아직 구현하지 않았습니다. 암호화·손상 파일과 스캔 PDF 등은 오류 기록을 확인하세요.

API 호출 형식은 [OpenAI 임베딩 문서](https://developers.openai.com/api/docs/guides/embeddings)와 [OpenAI API Quickstart](https://developers.openai.com/api/docs/quickstart)를 따릅니다.

## 결과물

| 구분 | 링크 |
| --- | --- |
| GitHub Repository | [rfp-rag](https://github.com/diokim17/rfp-rag) |
| 최종 보고서 PDF | 파일 및 다운로드 링크 추가 예정 |
| 발표 자료 | 링크 추가 예정 |
| 개인별 협업일지 | [팀원 및 역할 바로가기](#팀원-및-역할) |

## 데이터 및 보안

- 제공받은 원본 RFP 문서는 외부 공유가 금지되어 있으므로 저장소에 업로드하지 않습니다.
- API Key, SSH Key 등 인증 정보는 저장소에 포함하지 않습니다.
