#!/usr/bin/env bash
# 김연주: eval_v2 cross-encoder 실험을 GPU 서버에서 한 번에 실행합니다.
#   bash experiments/run_eval_v2_gpu_yjk.sh
# 준비: 저장소 루트에 .env(OPENAI_API_KEY, EXPERIMENT_OWNER), data/raw(CSV + files), data/eval_v2.json
# 비용: 인덱스가 없으면 빌드(약 $0.25)와 질문 임베딩 65건. 인덱스가 있으면 질문 임베딩만.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
INDEX=indexes/eval-v2-structured-yjk
export PYTHONUTF8=1 LANGFUSE_ENABLED=${LANGFUSE_ENABLED:-false}

"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 없음: GPU 서버에서 실행하세요')"
"$PY" -c "import transformers" 2>/dev/null || "$PY" -m pip install transformers

if [ ! -f data/processed/documents.json ]; then
  "$PY" run.py parse --processed-dir data/processed
fi
if [ ! -f "$INDEX/config.json" ]; then
  RFP_CHUNKING_STRATEGY=structured RFP_EMBEDDING_CONTEXT=none \
    "$PY" run.py build --processed-dir data/processed --index-dir "$INDEX"
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
