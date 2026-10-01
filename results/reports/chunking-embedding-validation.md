# 청킹·임베딩 구현 검증

## 상태

- 작업 브랜치: `feature/chunking-embedding`. `dev` 대상 리뷰용 변경이며 병합은 별도 진행.
- 구현·오프라인 테스트·**실제 HWP 3개 문서의 fixed/boundary 비교 완료**.
- 사용자가 개인 `.env`에 담당자를 설정한 뒤 기존 CLI로 파싱 완료. 자동 실험 ID는 `chy-0001`이며 번호 생성기를 우회하거나 수정하지 않음.
- 실제 OpenAI API 호출 0회. 실제 API 토큰·검색 Recall·답변 품질은 미측정. 청킹 구조와 로컬 실행 시간만 측정했으며 검색 성능 향상을 주장하지 않음.
- [상세 측정값·해시·오프라인 재현 코드](chunking-20261001T060125Z-1f09a0b4.md).
- Python 3.12.3의 개인 `.venv` 사용. 새 의존성 없음.

## 변경과 유지 조건

- `embedding.py`에서 실행 시 `RFP_CHUNKING_STRATEGY=fixed|boundary`를 읽음. 미설정 기본값은 fixed. 빈 값과 지원하지 않는 값은 오류.
- boundary는 빈 줄 → 줄바꿈 → 문장 종결 뒤 공백 순서로 분할 끝점을 선택. 같은 종류는 가장 뒤의 경계를 사용하고, 연속 빈 줄도 마지막 경계를 선택함.
- 후보 끝점의 최소 길이는 `max(ceil(chunk_size / 2), chunk_overlap + 1)`. 적합한 경계가 없으면 고정 길이 분할. 다음 시작점은 `end - chunk_overlap`.
- 원문을 재작성하지 않음. 모든 Chunk 본문은 원문의 start_char:end_char 슬라이스이며 Document/Chunk/Hit/Answer의 필수 키와 metadata를 보존.
- 공개 함수의 이름·인자 순서·기본값·반환 형식 유지. 문서 ID와 chunk_id 형식 유지.
- 기존 32개 순차 임베딩 배치, float32/L2 정규화, IndexFlatIP 유지. 정규화 전 벡터 스케일을 맞춰 극단적인 유한 값의 제곱합 overflow/underflow 방지.
- 빈 입력, 응답 index의 누락·중복·범위·타입 오류, 불일치 차원, NaN/Inf, 영벡터를 거부. 배치 간 차원을 검사하고 전체 임베딩 성공 전 파일을 저장하지 않음.
- 기존 index.faiss/chunks.json/config.json 및 config 필수 키 유지. config에 chunking_strategy와 chunking_version=1만 추가. 새 키가 없는 기존 인덱스도 로드 가능.
- run.py, 파싱·검색·생성·평가·추적·자동 ID 모듈 및 공통 의존성은 변경하지 않음. 새 설정은 기존 build 추적으로 전달되며 본문·질문·답변·인증 정보는 추가 기록하지 않음.

## 실제 호출 관계와 문서 차이

- 빌드: `run.py → build_index → chunk_documents → embed_texts → FAISS/JSON 저장`.
- 질의·평가: `run.py → load_index → retrieve → embed_texts → generate_answer → evaluate`.
- 질문 임베딩은 저장된 config의 모델을 사용. sources는 전달된 검색 결과 전체에 citation을 부여한 목록.
- 문서의 공개 계약은 실제 코드와 일치. 개발 가이드의 수동 실험 ID/경로 예시는 최신 자동 ID 방식과 혼동 가능. 경로 구분자와 run.py가 부여하는 자동 실험 ID를 구분해야 함.
- 기존 자동 평가 보고서는 새 전략 키를 출력하지 않으므로 비교 요약에 전략/version을 별도로 명시해야 함.

## 실행한 검증

- `.venv/bin/python -m pip check`: 통과.
- `.venv/bin/python -m unittest discover -s tests -v`: **31개 통과**(기존 13개 + 새 18개). PR 제출 전 재검증도 통과. 테스트 실행 시간은 실제 서비스 지연이 아님.
- `git diff --check`: 통과.
- 원문·전처리·인덱스·실험 결과 JSON 경로의 Git 제외 규칙 확인.

추가 테스트는 두 전략의 원문 위치·비공백 문자 커버리지·종료·Unicode·공백·짧은 문서·긴 무구분 문자열·overlap 극단값·문단 우선순위를 검증함. 기존 fixed 분할과 공백 건너뛰기 순번도 검증함.

모의 API로 역순 응답, 잘못된 index와 벡터, 65개 입력의 32/32/1 배치 순서, 중간 실패·차원 불일치 시 기존 인덱스 보존을 검증함. 두 전략과 기존 config의 저장→로드→필터 검색→생성→평가 연결 및 JSON 직렬화가 통과함. 모의 추적의 배치별 토큰 합산과 본문·파일명·기관명 제외도 확인함. 모의 API 평가 점수와 토큰은 실제 성능·사용량으로 보고하지 않음.

## 실제 문서 비교

동일 CSV 앞 3행의 HWP 3개를 한 번 파싱하여 문서 내용·순서를 고정했다. 원문 총 72,553자, 크기/중복 1000/150, chunking_version=1이다. 문서 해시·코드 커밋·dirty 상태·embedding.py 해시는 상세 보고서에 기록했으며 현재 코드와 산출물의 해시 일치도 확인했다.

| 지표 | fixed | boundary |
| --- | ---: | ---: |
| 청크 수 | 86 | 98 |
| 문단 경계 분할 비율 | 2.41% | 63.16% |
| 줄 경계 분할 비율 | 4.82% | 100.00% |
| 중복 문자 수 | 12,450 | 14,250 |
| 청킹 시간 중앙값 | 0.199ms | 3.343ms |
| 비공백 문자 커버리지 | 100% | 100% |
| 향후 build 요청 예상(32개 배치, 재시도 제외) | 3회 | 4회 |

문서 끝을 제외한 분할점에서 경계 비율을 측정했다. 줄 경계 비율에는 문단 경계가 포함된다. 시간은 전략별 워밍업 10회 후 순서를 교대하여 각 101회 측정했으며 I/O·파싱·임베딩을 제외했다. 변경 전 HEAD 함수와 새 fixed의 실제 청크 전체 일치를 확인했다. 두 전략 모두 원문 슬라이스·최대 길이·비공백 문자 커버리지 검증을 통과했다.

boundary는 문단·줄 경계에서 더 자주 분할했으나 청크 수·중복량·CPU 시간이 증가했다. 이는 검색 품질 향상을 의미하지 않는다. 실제 정답 평가셋을 통한 검색·생성 평가와 토큰·비용 비교는 남아 있다.

산출물은 `data/processed/chunking-20261001T060125Z-1f09a0b4` 및 `results/chunking-20261001T060125Z-1f09a0b4`에 분리 저장했다. 실제 인덱스는 생성하지 않았다. 원문 포함 JSON은 Git에서 제외하며 공유 대상은 원문 없는 Markdown 요약뿐이다.

## 재실행

아래 폴더 구분자는 자동 실험 ID가 아니다. 현재 개인 .env의 EXPERIMENT_OWNER를 사용하며, 기존 산출물과 충돌하지 않는 새 경로를 선택한다. 오프라인 검증의 parse에서만 Langfuse를 명시적으로 끈다.

```bash
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
git diff --check

RUN_DIR="chunking-$(date -u +%Y%m%dT%H%M%S)-$$"
export PROCESSED_DIR="data/processed/$RUN_DIR"
LANGFUSE_ENABLED=false .venv/bin/python run.py parse --limit 3 \
  --processed-dir "$PROCESSED_DIR"
```

후속 **유료** build/evaluate는 문서·질문 수와 예상 호출량을 협의한 뒤 실행한다. 동일 평가셋·top_k·답변 모델을 유지한다. data/eval.json은 아직 없으며 원문으로 정답을 검증하여 준비해야 한다.

```bash
for strategy in fixed boundary; do
  RFP_CHUNKING_STRATEGY="$strategy" .venv/bin/python run.py build \
    --processed-dir "$PROCESSED_DIR" --index-dir "indexes/$RUN_DIR/$strategy" \
    --chunk-size 1000 --chunk-overlap 150
  .venv/bin/python run.py evaluate \
    --processed-dir "$PROCESSED_DIR" --index-dir "indexes/$RUN_DIR/$strategy" \
    --results-dir "results/$RUN_DIR/$strategy" \
    --eval-file data/eval.json --top-k 5
done
```

## 한계와 재생성

boundary 적용은 새 인덱스 build가 필요하다. 기존 fixed 인덱스는 그대로 재사용 가능하며, 같은 파싱 결과가 있으면 parse를 반복하지 않는다.

경계 탐색은 CPU 시간과 청크 수·중복량을 늘릴 수 있다. 문자 overlap의 시작점은 문장 중간일 수 있고 표 구조 복원·토큰 상한·의미적 완결성을 보장하지 않는다. 원문 범위 확대 및 PDF 검증, 실제 검색·생성 평가가 남아 있다. 파일 저장의 원자적 교체는 이번 범위 밖이며 build 중 해당 인덱스를 조회하지 않는다.

원문이 들어 있는 산출물 JSON은 커밋하지 않는다. 공유는 본 보고서처럼 원문 없는 요약만 사용한다.
