# Langfuse 추적 설정 (선택)

Langfuse는 질의응답 사용에 필수가 아닙니다. 키를 입력하지 않으면 외부 추적 없이 로컬 결과만 저장합니다. 패키지는 각 실행 방식의 설치 파일에 포함되어 있습니다.

## 연결 설정

본인의 Langfuse 프로젝트에서 발급받은 키와 프로젝트 주소를 사용할 실행 방식의 `.env.server` 또는 `.env.api`에 입력합니다.

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=본인_PUBLIC_KEY
LANGFUSE_SECRET_KEY=본인_SECRET_KEY
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

`LANGFUSE_BASE_URL`은 본인 프로젝트가 위치한 주소로 지정합니다. 키와 설정 파일은 Git에 올리지 않습니다. 추적을 끄려면 `LANGFUSE_ENABLED=false`로 설정합니다. 설정이 불완전하거나 초기화에 실패하면 경고를 출력하고 로컬 기록으로 실행을 계속합니다.

```bash
python3.12 run.py ask --question "사업의 주요 요구사항은 무엇인가요?"
```

## 저장되는 기록

| 위치 | 내용 |
| --- | --- |
| Langfuse | 실행·단계별 시간, 오류 종류, 코드 버전, 입력·인덱스 해시, 검색 ID·점수, 모델·토큰 사용량 |
| `results/server/`, `results/api/` | 질의응답 결과 JSON, 실행 설정, 실행 ID, trace ID, 토큰 사용량 |
| 각 실행 방식의 `reports/` | 선택적 평가 실행 시 생성되는 요약 보고서 |

질문·원문·답변 본문·파일명·필터 값은 Langfuse로 전송하지 않습니다. 질문은 해시로 구분합니다. 실제 질문과 답변 분석에는 로컬 결과 JSON을 사용하세요. OpenAI 임베딩·생성·평가 판정기에 전달하는 입력은 이 추적 설정과 별개입니다.

`EXPERIMENT_OWNER`는 선택 설정입니다. 비우면 `user-0001` 형식의 실행 ID를 사용하고, 영문 이름(2~16자)을 지정하면 사용자별로 실행 번호를 구분합니다. 실행 번호는 `.experiment-state/server/`와 `.experiment-state/api/`에 저장됩니다.

Langfuse가 모델 가격을 알고 있으면 기록된 토큰으로 비용을 계산할 수 있습니다. 이 값이 제공업체의 전체 청구액과 같다는 의미는 아닙니다. 로컬 결과에는 문서 근거가 포함될 수 있으므로 공유 전에 내용을 확인하세요.
