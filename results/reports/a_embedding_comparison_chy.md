# 시나리오 A 임베딩 구현 및 모델 비교

2026-10-12 · 브랜치 `feature/a-embedding-chy` · 기준 dev `2c3136996f477eb66d8f7fed800cec1094fb667c` · 평가 당시 미커밋 작업

## 결론

이번 100문서·eval_v2 65문항의 dense 검색 조건에서는 Recall@5를 우선하여 **BAAI/bge-m3**를 선택 후보로 권장한다. KURE-v1은 첫 정답의 순위(MRR)가 더 좋지만 여러 정답 문서를 모두 찾는 비율(Recall)이 낮았다. 통계적 유의성이나 다른 데이터·검색 설정으로의 일반화는 검증하지 않았다.

| 모델 | 실험 ID | Recall@5 | MRR@5 | 평균 검색(ms) | 중앙값(ms) | p95(ms) | 구축(초) | 로딩(초) | 실패 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BAAI/bge-m3 | chy-0029 | 0.9351 | 0.9513 | 35.04 | 33.53 | 45.35 | 136.18 | 10.24 | 0 |
| nlpai-lab/KURE-v1 | chy-0030 | 0.9262 | 0.9628 | 35.36 | 34.04 | 45.99 | 137.85 | 18.05 | 0 |

KURE-v1의 Recall은 BGE-M3보다 0.90%p 낮고 MRR은 1.15%p 높다. 지연 차이는 약 0.32ms로, 이번 단일 서버 실행만으로 속도 우열을 일반화하지 않는다.

## 변경 내용과 계약

- `scenario_a_embedding.py`: 공개 `load_model(model_path: Path)`와 객체의 `embed_texts(texts)` 구현. Sentence Transformers의 로컬 토크나이저·pooling 사용, 입력 순서의 CPU float32 2차원 배열 반환.
- GPU FP16 / CPU FP32, 최대 8192토큰(모델 상한 이내), 내부 배치 8. CUDA OOM 시 배치를 절반씩 줄이며 단일 입력 실패는 전파한다. import 시 모델 다운로드·로딩·환경 파일 접근은 없다.
- 모델 다운로드와 추론 분리. 고정 revision과 파일 해시 manifest를 준비하고 실험 전 검증한다. 기존 후보 폴더는 prepare로 덮어쓰지 않는다.
- 기존 공통 벡터 검증·L2 정규화·FAISS와 Document → Chunk → Hit → Answer, JSON 필수 키·공개 인자·기본값을 유지한다. `run.py`, 파싱·검색·생성·평가·추적·ID 생성기·공통 requirements에는 변경이 없다.
- `experiments/a_embedding_chy.py`: 기존 ScenarioAClient, build_index, retrieve, 자동 실험 ID, Trace를 재사용한 검색 전용 도구. 본문 없는 요약과 입력 사본을 별도 경로에 저장한다.
- 기존 계약 테스트 6개 파일에 이미 존재하는 `retrieval_baseline.setUpModule`을 연결했다. PyTorch 설치로 실제 cross-encoder가 자동 활성화되는 영향만 격리하며 기존 assertion은 유지한다. 기본 검색 동작은 기존 전용 테스트에서 검증한다.
- 개인 `.venv`에만 선택 의존성을 설치했다. 직접 의존성 목록 및 검증된 추론 의존성 48개의 별도 lock 파일을 제공한다.

## 동일 비교 조건

- 현재 개인 processed의 100문서를 복사하여 사용. 두 모델의 문서·평가셋·chunks 해시 일치, 평가 정답 문서 누락 0개.
- fixed 1000자 / overlap 150자 / embedding_context=none / 10,218청크 / 1024차원 / L2 + IndexFlatIP.
- top-k=5, 기존 필터 처리 유지. hybrid·rerank·rewrite·within_doc·expand 비활성, 문서당 결과 상한 없음. 모델 가중치·고유 구성만 변경.
- NVIDIA L4, FP16, 배치 8, 최대 길이 8192. 실행 시 OMP_NUM_THREADS=4 지정, 런타임 torch CPU threads 기록은 두 모델 모두 2.
- 모델별 워밍업 1문항 후 65문항을 순서대로 3회(195회) 측정. 질문 임베딩 캐시 없음. 3회 모두 문항별 Recall·MRR 일치. 195개의 서로 다른 평가 사례가 아니라 65개 사례의 반복이다.
- 검색 시간은 질문 임베딩·필터·검색을 포함하며, 로딩·구축·생성은 제외한다. 구축·모델 로딩 시간은 별도 측정했다. 공유 서버 부하와 파일 캐시 영향이 있을 수 있다.
- 평가 함수의 문서 Recall 정의와 동일. MRR은 상위 5개 청크에서 첫 정답 문서가 나타나는 청크 순위의 역수. 문서 중복 제거 후의 순위가 아니다.

## 토큰·자원·실패

- 두 모델 모두 구축 5,718,975토큰, 워밍업 25토큰, 3회 평가 4,869토큰. 합계 5,723,869토큰/모델. padding 제외·특수 토큰 포함·잘림 이후 실제 입력 길이 기준이며 API 과금 토큰이 아니다.
- 모델별 구축 공통 임베딩 호출 320회(내부 배치로 분할), 워밍업 1회, 질문 임베딩 195회. OpenAI 임베딩/생성 호출 0회.
- 두 모델 모두 입력 잘림 0, OOM 재시도 0, 추론·검색 실패 0. 최대 torch GPU allocated memory 약 1263.76MiB/모델이며 전체 GPU 예약 메모리나 시스템 사용량과는 다르다.
- 기존 Langfuse 추적을 유지하고 내부 usage 어댑터로 로컬 토큰 합계를 기록했다. 외부 기록은 수치·모델 식별자·해시·문서/청크 ID·오류 종류로 제한한다. 서버 저장 성공을 별도 API 조회로 검증하지는 않았다.

## 문항별 차이

본문 없이 평가 ID와 지표만 공유한다. 아래는 어느 한 지표라도 달라진 모든 문항이다.

| ID | BGE Recall | KURE Recall | BGE MRR | KURE MRR |
| --- | ---: | ---: | ---: | ---: |
| e03 | 1.0000 | 1.0000 | 0.2500 | 1.0000 |
| h05 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| h10 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| h14 | 0.6667 | 0.3333 | 1.0000 | 1.0000 |
| g02 | 0.7500 | 0.5000 | 1.0000 | 1.0000 |
| g03 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| g04 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| g09 | 0.5000 | 0.0000 | 1.0000 | 0.0000 |
| g10 | 0.5000 | 1.0000 | 0.2500 | 0.2500 |

- KURE는 약칭 문항 h05의 정답을 찾았고, g10의 다중 정답 포함률도 높였다. e03은 첫 정답 순위가 좋아졌다.
- h10·h14·g02·g03·g04에서는 다중 정답 포함률이 낮아졌고, g09는 정답이 상위 5개에서 빠졌다. 첫 정답 순위만으로 다중 문서 검색 품질을 판단하면 안 된다.
- 두 모델 모두 완전 Recall에 도달하지 못한 문항: h14, g02, g06, n06, g07, g09. 정답 문서 수가 많은 문항에서는 중복 청크가 top-k 공간을 차지할 수 있으나, 이를 원인으로 확정하려면 문서 다양성·리랭킹의 별도 통제 실험이 필요하다.

## 검증

- 변경 전: 기존 220개 테스트 통과, pip check 통과.
- 변경 후: 기존+신규 235개 테스트 통과. 새 테스트는 로컬 로딩·모델 재사용·입력 순서·경계 입력·OOM 재시도·토큰 집계·추적 개인정보 제외·추적 장애 시 중복 추론 방지·실제 어댑터를 통한 FAISS→검색→대역 생성→평가·잘못된 벡터의 기존 인덱스 보호·모델 파일 해시 검증을 포함한다.
- 두 실제 모델 모두 합성 입력 3개(빈 문자열 포함)의 GPU 추론 성공, 출력 (3,1024), float32, 유효 수치 확인.
- 실제 `run.py build`도 BGE-M3와 연결해 통과했다(`chy-0031`). 임시 경로의 합성 문서 1개로 1024차원·1청크와 기존 인덱스 세 파일을 확인했다. 생성 모듈은 로딩하지 않았으며 이 합성 smoke에서만 Langfuse를 비활성화했다.
- 최종 `pip check`, `git diff --check` 통과.
- 반복 결과의 입력·청크·구현 코드 해시 일치와 문항별 점수 일치를 별도로 검증했다.

## 재실행

저장소 루트에서 본인의 가상환경을 사용한다. .env의 EXPERIMENT_OWNER는 읽기만 하며 실험 ID를 수동 지정하지 않는다. 아래 run은 실행마다 새 ID와 별도 processed/index/results 경로를 만든다.

```bash
.venv/bin/python -m pip install -r experiments/requirements-a-embedding.lock.txt
.venv/bin/python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -q
```

다른 환경에서 모델이 아직 없을 때만 다음 준비 명령을 실행한다. 이 서버에는 이미 존재하므로 재실행할 필요가 없다.

```bash
.venv/bin/python experiments/a_embedding_chy.py prepare --model bge-m3 --revision 5617a9f61b028005a4858fdac845db406aefb181
.venv/bin/python experiments/a_embedding_chy.py prepare --model kure-v1 --revision 8b418a58414668e75532ed045c22d9ca018ae2b2
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python experiments/a_embedding_chy.py run --model-folder chy-bge-m3-5617a9f61b02
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python experiments/a_embedding_chy.py run --model-folder kure-v1
```

위 실행 명령은 현재 서버 경로 기준이다. KURE-v1 폴더는 평가 후 `kure-v1`로 이름을 변경했다. 다른 환경에서 prepare로 처음 다운로드하면 출력된 `chy-kure-v1-8b418a584146` 이름을 사용한다.

다른 입력 사본을 지정하려면 `--documents <documents.json> --eval-file <eval.json>`을 두 모델에 동일하게 전달한다. 기존 저장 인덱스를 재평가하려면 아래 보존 경로를 사용한다. 현재 도구의 run 명령은 새 인덱스를 구축하는 비교 실행이다.

## 산출물과 재현 정보

- 기존 개인 데이터·인덱스와 .env는 수정하지 않았다. 비교 후 사용자 요청으로 BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`을 `default_model`에 복사하고 파일별 해시 일치를 확인했다. BGE-M3 원본 폴더도 유지하며, KURE-v1 폴더는 `kure-v1`로 이름을 변경했다. 모두 공용 embedding 경로 아래에 있다.
- `data/processed/a-embedding/chy-0029/`, `chy-0030/`: 각 실행의 documents.json·eval.json 사본.
- `indexes/a-embedding/chy-0029/`, `chy-0030/`: 후보별 FAISS 인덱스 세 파일.
- `results/a-embedding/chy-0029/evaluation.json`, `chy-0030/evaluation.json`: 원시 측정·검색 ID 결과. Git 제외 확인.
- 상세 수치·모델 파일 해시: [BGE-M3 보고서](a_embedding_chy-0029.md), [KURE-v1 보고서](a_embedding_chy-0030.md).
- KURE-v1 이름 변경 후 새 인덱스 `indexes/a-embedding/chy-0032/`를 구축했다. 기존 FAISS 파일과 바이트가 일치하고 검색 3건을 통과했다. [재구축 보고서](kure_v1_rebuild_chy-0032.md).

- documents_sha256: `97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a`
- evaluation_sha256: `3f1f37b5eb5a06ed0827d4368dad73c99a6bd48ab4048334804bd6cd6d2f2b00`
- chunks_sha256: `874aee869ed0fc4b2242f37ae7ede49517c1307ec421ef503794ee36e0496a0c`
- module_sha256: `67744ea6c44236a31af1bb43160ed8df6cf8705ed92e3e97f438dbbcdd91b296`
- runner_sha256: `01c19563fa841033ecdfb8943baf01fb750702ad6c5c55137c7a11fd158b59f6`

위 코드 해시는 실제 모델 비교 시점의 값이다. 이후 임베딩 모듈에 한국어 주석·설명문을 추가했으며, 이를 제외한 AST 비교로 실행 로직이 동일함을 확인했다.

## 문서 차이와 한계

- 팀 가이드의 A·리랭킹 미구현 설명은 현재 코드보다 오래됐다. A 연결 코드는 이미 있고 이번에 임베딩 담당 모듈을 구현했다. 생성 담당 모듈은 아직 없다.
- 가이드의 build_index 표에는 실제 코드의 documents_sha256 인자가 빠져 있고 retrieve 표에는 추가 keyword-only 옵션이 생략돼 있다. 이번 작업은 현재 공개 함수를 변경하지 않았다.
- 로컬 baseline은 BGE-M3다. 기존 B 보고서는 문서·검색 설정이 달라 직접 비교하지 않았다. B 대비 성능 향상을 주장하지 않는다.
- eval_v3, 하이브리드·리랭킹 결합, 실제 생성 답변 품질, 생성 모델과 동시 GPU 사용, 실제 CPU 추론, 더 긴 문서의 실제 잘림·OOM은 후속 검증 대상이다. 긴 입력·OOM 경계는 테스트 대역으로 검증했다.
- 같은 폴더명에 다른 모델을 덮어쓰면 기존 공통 config는 이를 모델명만으로 구분하지 못한다. 후보별 revision 폴더와 manifest를 유지해야 한다.
- 임베딩 모델·청킹 변경에는 **새 인덱스 구축이 필요**하며 기존 인덱스를 혼용하면 안 된다. 원본·파싱이 같으면 parse는 재실행하지 않는다.
- 개인 .env는 자동 전환하지 않았다. A CLI를 사용할 때 A_EMBEDDING_MODEL에 후보 폴더명을 지정한다. build에도 A_GENERATION_MODEL 설정값은 필요하지만 생성 파일 로딩은 하지 않는다. ask/evaluate에는 별도 생성 담당 구현이 필요하다.
- 후속 배치한 모델은 `A_EMBEDDING_MODEL=default_model`로 연결한다. 기존 후보 이름의 인덱스는 모델명이 다르므로 `default_model` 이름으로 사용할 새 인덱스를 구축해야 한다.

모델 근거: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1).
