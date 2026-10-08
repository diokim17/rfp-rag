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

두 시나리오 모두 GCP Linux 서버에서 개발·실행합니다. A는 서버에 저장한 모델을 직접 실행하고, B는 OpenAI API를 호출합니다. 공통 검색 인덱스는 FAISS를 사용합니다. A 모델과 추론 패키지 버전은 담당자 선정 후 업데이트합니다.

| 구분 | 시나리오 A: GCP 모델 직접 실행 | 시나리오 B: OpenAI API 기반 |
| --- | --- | --- |
| 개발 언어 및 버전 | Python 3.12.3 | Python 3.12.3 |
| 실행 환경 | GCP 서버 / Linux | GCP 서버 / Linux |
| 문서 파싱 | 작성 예정 | 작성 예정 |
| 임베딩 모델 | 담당자 선정·구현 예정 | `text-embedding-3-small` (기본, 환경 변수로 변경) |
| Vector DB | FAISS | FAISS |
| LLM | 담당자 선정·구현 예정 (GCP 직접 실행) | `gpt-5-mini` (기본) / `gpt-5-nano` |
| 주요 라이브러리 | 프로젝트 종료 후 작성 | 프로젝트 종료 후 작성 |

## 프로젝트

### 프로젝트 구조

`dev`는 시나리오 A/B를 함께 개발하는 통합 브랜치입니다. 기존 파이프라인과 입출력 형식을 유지하며, `.env`의 `RFP_SCENARIO`에 따라 사용할 모델 구현을 연결합니다. B의 OpenAI 호출과 A/B 연결 코드는 구현되어 있고, 실제 A 생성·임베딩 모듈은 각 담당자가 구현해야 합니다.

```text
rfp-rag/
├── data/
│   ├── raw/                     # 원본 RFP 문서(HWP·PDF) 및 메타데이터
│   └── processed/               # 파싱·전처리 결과 (A/B 공유 가능)
├── indexes/                     # B 기본 인덱스
│   └── scenario-a/              # A 기본 인덱스
├── results/                     # B 기본 실행 결과
│   ├── reports/                 # B 평가 요약
│   └── scenario-a/              # A 기본 실행 결과
│       └── reports/             # A 평가 요약
├── run.py                       # 김도영: 설정·A/B 선택·경로·전체 실행 흐름
├── scenario_a.py                # 김도영: A 담당 모듈 로딩·호출 연결·모델 객체 재사용
├── parsing.py                   # 나상훈: HWP·PDF 추출, 정제, 메타데이터 결합
├── embedding.py                 # 유찬혁: 공통 청킹·벡터 검증/정규화·FAISS 구축, B 임베딩 호출
├── scenario_a_embedding.py      # 유찬혁: A 임베딩 모델 로딩·추론 (구현 예정)
├── retrieval.py                 # 김연주: 벡터·하이브리드 검색, 필터링, 리랭킹
├── query_rewrite.py             # 김연주: 질문 재작성 (현재 OpenAI 의존)
├── generation.py                # 박단비: B 프롬프트·생성, A 생성 호출 전달
├── scenario_a_generation.py     # 박단비: A 생성 모델 로딩·프롬프트·추론 (구현 예정)
├── evaluation.py                # 김시현: 평가 데이터 검증 및 검색·답변·지연 평가
├── tests/                       # 오프라인 설정·연결·오류 처리·B 회귀 테스트
├── .env                         # 시나리오·모델 폴더 이름·API 키 등 (Git 제외)
├── .env.example                 # 환경 변수 설정 예시
├── requirements.txt             # 팀 공통 패키지 목록
└── README.md
```

`scenario_a_embedding.py`와 `scenario_a_generation.py`는 연결 코드에서 사용하는 모듈 이름이며, 아직 저장소에 없는 구현 예정 파일입니다. 모델 파일은 저장소 밖의 공용 경로에 저장합니다.

```text
/home/spai1313/models/
├── generate/<생성 모델 폴더>/
└── embedding/<임베딩 모델 폴더>/
```

### 브랜치 운영

A/B 모두 최신 `dev`에서 개인 기능 브랜치를 만들고, PR 대상도 `dev`로 지정합니다. 각자의 기능 브랜치에서 작업·테스트한 뒤 리뷰를 거쳐 통합하며, 최종 검증된 `dev`를 `main`에 반영합니다. 기존 B 기능 브랜치는 그대로 사용하되 최신 `dev`를 반영하고 PR 대상을 확인하세요.

```text
feature/a-generation ─┐
feature/a-embedding  ──┤
기타 A/B 기능 브랜치 ───┴─→ dev ─→ main
```

기존 `dev-a`의 통합 준비 코드는 `dev`에 반영했습니다. 앞으로 시나리오별 통합 브랜치를 추가하지 않고 `dev`를 기준으로 개발합니다.

### A/B 담당 파일과 연결 방식

| 담당자 | 주요 파일 | 맡을 작업 |
| --- | --- | --- |
| 김도영 | `run.py`, `scenario_a.py`, `.env.example` | 설정·경로·A/B 연결, 실행 환경 조율, 통합 검증 |
| 박단비 | `scenario_a_generation.py`, `generation.py` | A 생성 모델 선정·다운로드·로딩·프롬프트·답변 생성, 기존 답변 형식 유지 |
| 유찬혁 | `scenario_a_embedding.py`, `embedding.py` | A 임베딩 모델 선정·다운로드·로딩·텍스트 벡터화, 청킹·인덱스 구축 |
| 김연주 | `retrieval.py`, `query_rewrite.py` | 검색·하이브리드·리랭킹 연결, A 질문 재작성 의존성 정리 |
| 김시현 | `evaluation.py`, 평가 데이터·실험 코드 | 동일 평가셋을 이용한 A/B 품질·지연 비교 |
| 나상훈 | `parsing.py` | A/B에서 공유할 문서 추출·정제·메타데이터 유지 |

각 A 담당 모듈은 아래 접점으로 연결합니다. 모델 다운로드는 별도로 준비하고, `load_model`은 전달된 로컬 모델 경로를 로딩해 실행 중 재사용할 객체를 반환합니다.

| 담당 모듈 | 로딩 접점 | 반환 객체의 메서드 | 추론 반환값 |
| --- | --- | --- | --- |
| `scenario_a_embedding.py` | `load_model(model_path: Path)` | `embed_texts(texts)` | 입력 순서의 2차원 숫자 벡터. 공통 `embedding.py`에서 float32 변환·검증·L2 정규화 |
| `scenario_a_generation.py` | `load_model(model_path: Path)` | `generate_answer(question, hits, model)` | 기존 답변 dict: `question`, `answer`, `sources`, `model`, `status` 등 |

생성 결과의 `sources`는 기존처럼 검색 근거와 `citation` 번호를 포함해야 합니다. 모델별 입력 접두어·토크나이저·길이 제한·장치 배치는 담당 모듈에서 처리합니다. 문서와 검색 질문에는 동일한 임베딩 모델을 사용합니다.

```dotenv
RFP_SCENARIO=A
A_GENERATION_MODEL=생성-모델-폴더명
A_EMBEDDING_MODEL=임베딩-모델-폴더명
RETRIEVAL_REWRITE=off
```

모델 설정값은 Hugging Face 저장소 ID가 아니라 공용 경로 아래의 단일 하위 폴더 이름입니다. `run.py`가 고정 기본 경로와 합쳐 담당 모듈에 전달합니다. `python run.py check-config`로 설정만 확인할 수 있으며, 기존 프로세스 환경 변수가 `.env`보다 우선합니다. B는 `RFP_SCENARIO=B`로 선택하고, 미설정 시에도 B를 사용합니다.

A의 기본 인덱스·결과 경로는 위 구조처럼 B와 분리됩니다. 명시한 `--index-dir`, `--results-dir`, `--reports-dir`는 우선 적용됩니다. 임베딩 모델이나 청킹을 바꾸면 인덱스를 다시 구축하고, 생성 모델만 바꾸면 기존 인덱스를 재사용할 수 있습니다.

현재 A에서는 OpenAI 기반 질문 재작성을 꺼야 하며, 켜져 있으면 안내 후 종료합니다. 모델 폴더·담당 모듈이 없거나 로딩이 미구현이면 B로 대체하지 않고 오류를 안내합니다. 오프라인 연결 검증은 완료했으며, 실제 A 모델과 GPU 메모리 통합 검증은 담당 모듈 구현 후 진행합니다.

### 팀 실험 옵션

`RFP_CHUNKING_STRATEGY`는 `fixed`(기본), `boundary`, `structured`를 선택합니다. structured의 기본 버전은 기존과 같은 v2이며, v3는 `RFP_STRUCTURED_CHUNKING_VERSION=3`으로 명시할 때 선택합니다. `RFP_EMBEDDING_CONTEXT`는 `none`(기본), `project`, `project_blend`, `table`, `section`을 선택합니다. 조건마다 별도 `--index-dir`로 빌드하고 질의·평가에도 같은 경로를 사용하세요. 선택한 청킹 전략과 버전은 인덱스 `config.json`에서 확인할 수 있습니다.

```bash
RFP_CHUNKING_STRATEGY=structured RFP_STRUCTURED_CHUNKING_VERSION=2 \
RFP_EMBEDDING_CONTEXT=section python run.py build \
  --processed-dir data/processed --index-dir indexes/structured-v2-section
```

청킹·임베딩 설정을 바꾸면 전체 청크의 인덱스를 다시 만들어야 합니다. B에서는 임베딩 API 비용이 발생하며, A에서는 서버 추론 자원을 사용합니다. `run.py`의 기본 실행 설정은 변경하지 않습니다.

이미 생성된 인덱스는 설정 환경변수를 바꿔도 변하지 않습니다. 그 인덱스를 그대로 쓸 때는 동일한 `--index-dir`로 `ask`/`evaluate`를 실행하고, 새 옵션을 적용할 때만 별도 경로로 `build`한 뒤 해당 경로를 질의·평가에 사용하세요.

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

프로젝트 폴더에서 가상환경을 활성화하고 필요한 패키지를 준비한 뒤 실행합니다. 자세한 서버 준비·공통 입출력 규칙은 [팀 개발 가이드](docs/TEAM_DEVELOPMENT_GUIDE.md)를 참고하세요. 실제 A 추론 패키지는 담당 모델에 맞춰 별도로 준비해야 합니다.

```bash
source .venv/bin/activate
```

`.env.example`을 참고해 자신의 `.env`를 설정하세요. 이미 작성한 `.env`는 덮어쓰지 않습니다. `EXPERIMENT_OWNER`에는 담당자 고유 영문 이니셜을 지정합니다.

### 시나리오 B: OpenAI API

```dotenv
RFP_SCENARIO=B
OPENAI_API_KEY=본인_API_키
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_GENERATION_MODEL=gpt-5-mini
EXPERIMENT_OWNER=본인_이니셜
```

```bash
python run.py check-config
python run.py parse
python run.py build
python run.py ask --question "질문 내용"
python run.py evaluate --eval-file data/eval.json
```

`data/eval.json`은 팀에서 준비한 평가셋이 있어야 합니다. 다른 파일을 사용할 때는 해당 경로를 지정하세요. `all --question "질문 내용"`은 파싱·인덱스 구축·질의를 연속 실행하며, B에서는 임베딩·생성 API 비용이 발생합니다.

### 시나리오 A: 서버 모델 직접 실행

공용 모델 폴더, 생성·임베딩 담당 모듈, 추론 패키지를 먼저 준비하고 위의 A 환경 변수 예시를 적용합니다. 이후 B와 같은 명령을 사용하며, 기본 인덱스·결과는 A 경로로 저장됩니다. 실제 모델·모듈 준비 전에는 안내 후 종료합니다.

인자 없이 `python run.py`를 실행하면 선택한 시나리오의 기본 인덱스 존재 여부에 따라 `ask` 또는 `all`을 선택합니다. 질문은 `run.py`의 `DEFAULT_QUESTION`, 기본 처리 문서 수는 `DEFAULT_LIMIT`을 사용합니다. 실행당 질문은 `--question`, 평가 질문 목록은 `--eval-file`로 지정합니다.

### 오프라인 검증

```bash
python -m unittest discover -s tests -v
```

가짜 모델·API로 설정, 인덱스 구축·검색·생성·평가 연결과 오류 처리를 검증합니다. 실제 모델 품질·GPU 메모리·OpenAI API 접속 성공은 별도 실행으로 확인해야 합니다.

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
