# 개인 Langfuse 기록과 팀 평가 요약 공유

각자 무료 프로젝트의 키로 연결합니다. 상세 기록은 개인 프로젝트에, 공통 평가 요약은 로컬 파일에 저장됩니다. 개인 프로젝트의 기록을 자동으로 합치는 방식은 아닙니다.

## 개인 환경 설정

본인 계정의 프로젝트 폴더에서 기존 가상환경에 패키지를 설치합니다.

```bash
cd ~/rfp-rag
source .venv/bin/activate
python -m pip install -r requirements-server.lock.txt
python -m pip check
```

개인 `.env`에 본인 프로젝트의 설정을 넣습니다. 키는 커밋하지 않습니다.

```dotenv
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"
LANGFUSE_ENABLED=true
EXPERIMENT_OWNER="yjk"
```

BASE_URL은 Markdown 링크가 아닌 URL 문자열입니다. 미국 리전 프로젝트의 예시이며 본인 프로젝트의 리전 주소를 사용합니다. 터미널에 설정된 환경 변수는 `.env`보다 우선합니다.

키가 없거나 `LANGFUSE_ENABLED=false`이면 로컬 결과만 저장합니다. 설정·초기화 실패는 경고 후 실행을 계속하며, 전송 오류는 SDK 로그를 확인합니다. CLI의 “기록 활성화”는 초기화 상태이지 서버 저장 성공을 보장하지 않습니다. 종료 시 대기 중인 기록을 flush합니다.

## 실행

담당자 이니셜은 `.env`에 한 번만 설정합니다. 실험 ID는 실행마다 `yjk-0001`, `yjk-0002`처럼 자동 증가합니다. `EXPERIMENT_ID` 환경 변수는 사용하지 않으며 `--experiment-id` 옵션도 제거했습니다. `--owner`는 필요할 때 담당자를 지정하는 옵션입니다.

| 담당자 | 영문 이니셜 |
| --- | --- |
| 김도영 | dyk |
| 나상훈 | shn |
| 유찬혁 | chy |
| 김연주 | yjk |
| 박단비 | dbp |
| 김시현 | shk |

위 한글 이름을 owner에 넣어도 대응하는 이니셜로 변환합니다. 그 외에는 고유한 영문 2~16자를 설정합니다. 대문자는 소문자로 통일하고 owner가 없으면 설정 방법을 안내하며 실행을 중단합니다.

번호는 프로젝트의 `.experiment-state/experiments.sqlite3`에 owner별로 저장합니다. 결과 파일을 지워도 다음 번호가 유지되며, 동시 실행은 DB 잠금으로 서로 다른 번호를 받습니다. 실패한 실행도 번호를 소비하므로 번호가 건너뛰는 것은 정상입니다. parse/build/ask/all/evaluate는 같은 owner 번호를 공유하며 각 명령 실행마다 증가합니다. 질문별 평가 사례마다 증가하는 번호는 아닙니다.

상태 폴더는 Git에서 제외합니다. 삭제하면 번호가 처음부터 시작하므로 보존하세요. 같은 owner가 새 clone이나 다른 서버에서 별도 상태 파일로 실행하면 번호가 중복될 수 있으므로 기존 상태 파일을 실행이 없는 때 함께 이전하거나, 별도 작업 공간에는 다른 owner 식별자를 사용합니다. 팀원별 고유 이니셜을 정하면 서로 다른 계정의 첫 실행도 구분됩니다.

```bash
python run.py ask --question "주요 요구사항은 무엇인가요?" --index-dir indexes/retrieval-yj-001 --results-dir results/retrieval-yj-001
python run.py evaluate --eval-file data/eval.json --index-dir indexes/retrieval-yj-001 --results-dir results/retrieval-yj-001 --top-k 5
```

본인의 인덱스와 OpenAI 키가 필요하고 API 비용이 발생합니다. 평가셋은 원문에서 정답을 확인해 팀이 별도 준비해야 합니다. parse/build에서도 자동 실험 ID가 배정됩니다. 각 CLI 실행은 별도 trace입니다.

## 기록 위치와 내용

| 위치 | 기록 |
| --- | --- |
| 개인 Langfuse | 전체·단계별 시간, 오류 종류, 담당자·실험 ID·커밋, 데이터/평가셋/인덱스 해시, 검색 문서·청크 ID와 점수, 모델·토큰, 실행 전체 평가 점수 |
| 로컬 결과 JSON | 기존 질문별 답변·근거·평가 기록, 설정, experiment 정보, trace ID, 토큰 합계 |
| `results/reports/evaluate_*.md` | 담당자·실험 ID·커밋·해시, 비교 설정, 평균 지표, 모델별 토큰 합계 |

parse/build는 시간·파싱 성공/실패 수·인덱스 설정을 기록하며 기존 동작대로 결과 JSON이나 평가 보고서를 만들지 않습니다. ask/all은 로컬 결과 JSON, evaluate는 JSON과 공유용 Markdown을 만듭니다. 요약 폴더는 `--reports-dir`로 변경합니다.

RFP 외부 공유 제한 때문에 **질문·원문·답변 본문·파일명·필터 값은 Langfuse로 보내지 않습니다.** OpenAI 자동 래퍼 대신 필요한 항목만 기록합니다. 질문은 `question_sha256`로 구분하고 실제 답변 분석은 로컬 JSON에서 합니다. 오류 메시지도 예외 종류만 기록합니다.

Langfuse가 해당 모델의 가격 정보를 알고 있으면 토큰 기반 비용을 계산할 수 있습니다. 공유 요약은 비용을 직접 계산하지 않습니다. 토큰은 성공 응답 기준이며 실패·재시도 과금이나 다른 호출을 포함한 전체 청구액은 아닙니다. 예산 자동 차단 기능도 아닙니다.

## 팀 공유

Markdown을 검토한 뒤 필요한 파일만 PR에 추가합니다. 질문·근거·답변이 들어 있는 원본 JSON은 공유하지 않습니다.

```bash
git status --short
# 실제 생성된 파일명으로 변경
git add results/reports/evaluate_실제시각.md
```

같은 평가셋·문서 범위·비교 조건으로 baseline과 개선 실험을 비교합니다. 해시는 사용한 산출물을 구분하는 용도이며 파일을 보관해 주지는 않습니다. 데이터와 인덱스는 비공개 위치에 보존하세요. `dirty=true`는 커밋 외 작업 내용이 포함된 실행입니다.

개인 trace ID는 실행자가 기록을 찾는 용도이며 다른 팀원에게 열람 권한을 제공하지 않습니다. 팀은 공유 요약을 비교하고 실패 분석이 필요하면 실행자가 자신의 상세 기록을 확인해 설명합니다.

추적은 `observability.py`, 공유 요약은 `experiment_reports.py`, 실행 연결은 `run.py`에서 관리합니다. 기존 모듈 입출력은 유지합니다.

공식 문서: [추적 안내](https://langfuse.com/docs/observability/get-started), [토큰·비용](https://langfuse.com/docs/observability/features/token-and-cost-tracking).
