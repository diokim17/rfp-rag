#!/usr/bin/env bash
# 김연주: eval_v2 cross-encoder 실험을 GPU 서버에서 한 번에 실행합니다.
#   bash experiments/run_eval_v2_gpu_yjk.sh
# 준비: 저장소 루트에 .env(OPENAI_API_KEY, EXPERIMENT_OWNER), data/raw(CSV + files), data/eval_v2.json
# 비용: 인덱스가 없으면 빌드(약 $0.25)와 질문 임베딩 65건. 인덱스가 있으면 질문 임베딩만. 경로는 PROCESSED·INDEX 환경 변수로 바꿀 수 있습니다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
# 팀 공용 data/processed·indexes는 건드리지 않고 개인 경로를 씁니다.
PROCESSED=${PROCESSED:-data/processed/eval-v2-yjk}
INDEX=${INDEX:-indexes/eval-v2-structured-yjk}
# shn-0035·yjk-0005와 같은 문서(PR #8 파싱)의 LF 기준 documents_sha256
EXPECTED_DOCS=0e93af3a97c4f261ea7304aa8db76b7638ed02eadf021eb6babc279ef7a5842c
export PYTHONUTF8=1 LANGFUSE_ENABLED=${LANGFUSE_ENABLED:-false}

"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 없음: GPU 서버에서 실행하세요')"
"$PY" -c "import transformers" 2>/dev/null || "$PY" -m pip install transformers
[ -f data/eval_v2.json ] || { echo "data/eval_v2.json이 없습니다."; exit 1; }
[ -f data/raw/data_list.csv ] || { echo "data/raw/data_list.csv와 data/raw/files/가 필요합니다."; exit 1; }

if [ ! -f "$PROCESSED/documents.json" ]; then
  "$PY" run.py parse --processed-dir "$PROCESSED"
fi
DOCS=$("$PY" -c "from experiment_reports import text_file_hash; print(text_file_hash('$PROCESSED/documents.json'))")
if [ "$DOCS" != "$EXPECTED_DOCS" ]; then
  echo "경고: documents_sha256 ${DOCS:0:8}… 이 shn-0035(${EXPECTED_DOCS:0:8}…)와 다릅니다. 원본 데이터나 파서 버전을 확인하세요."
  [ "${ALLOW_DOC_MISMATCH:-0}" = 1 ] || exit 1
fi
if [ ! -f "$INDEX/config.json" ]; then
  RFP_CHUNKING_STRATEGY=structured RFP_EMBEDDING_CONTEXT=none \
    "$PY" run.py build --processed-dir "$PROCESSED" --index-dir "$INDEX"
fi

# k(top_k)는 팀 기준 5로 고정. 필터 표기 보정은 retrieve 기본 동작이라 +fuzzy는 붙이지 않습니다.
# 주 비교: 하이브리드 + cross-encoder + 상한 2 vs 하이브리드 + lexical + 상한 2 (후보 50, BM25 접두 켬/끔)
SETTINGS="hybrid-cap2 prefix-hybrid-cap2"
for prefix in "" "prefix-"; do
  SETTINGS="$SETTINGS ${prefix}hybrid-ce-c50-cap2 ${prefix}hybrid-lex-c50-cap2 ${prefix}hybrid-ce-c50"
done
"$PY" -u experiments/eval_v2_yjk.py --eval-file data/eval_v2.json --index-dir "$INDEX" --top-k 5 \
  --settings $SETTINGS --fresh-query-embeddings
ls -t results/reports/eval_v2_yjk_*.md | head -1
