# KURE-v1 폴더명 변경 후 인덱스 재구축

- 자동 생성 실험 ID: `chy-0032`
- 모델: `nlpai-lab/KURE-v1`, 폴더명 `kure-v1`, revision `8b418a58414668e75532ed045c22d9ca018ae2b2`
- 기존 chy-0030과 동일한 100문서, fixed 1000자 / overlap 150자, embedding_context=none.
- 새 인덱스: `indexes/a-embedding/chy-0032/`
- 문서·평가셋 사본: `data/processed/a-embedding/chy-0032/`
- 로컬 검증 기록: `results/a-embedding/chy-0032/rebuild.json`
- 구축: 10,218청크 / 1024차원 / 136.41초.
- 문서 SHA-256: `97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a`
- 인덱스 SHA-256: `bc2e509f8cb38e0ec1a541a28040e2a3c38428b6d85414f7ece79556bdae789f`
- 기존 인덱스와 FAISS 파일 바이트 일치: True
- 기존 인덱스 3개 파일의 해시 불변 확인. 신규 config에서 embedding_model만 kure-v1로 변경됨.
- 평가셋 첫 3문항으로 로컬 질문 임베딩·검색 연결 확인. 전체 성능 평가는 재실행하지 않음.
- 검색 확인 결과(본문 제외): `[{"id": "e01", "hit_count": 5, "found_gold": true}, {"id": "e02", "hit_count": 5, "found_gold": true}, {"id": "e03", "hit_count": 5, "found_gold": true}]`
- 생성·OpenAI API 호출 없음. 기존 Langfuse 추적 유지, 본문·질문·키를 외부 기록에 추가하지 않음.
- .env는 변경하지 않음. 사용 시 A_EMBEDDING_MODEL=kure-v1과 위 index/processed 경로를 함께 지정.

## 동일 조건으로 새 경로에 재구축하는 명령

아래 NEW_INDEX_DIR은 아직 사용하지 않은 개인 인덱스 경로로 지정한다. 실험 ID는 run.py가 자동 생성한다.

```bash
RFP_SCENARIO=A A_EMBEDDING_MODEL=kure-v1 A_GENERATION_MODEL=unused-retrieval-only \
RFP_CHUNKING_STRATEGY=fixed RFP_EMBEDDING_CONTEXT=none RETRIEVAL_REWRITE=off \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
.venv/bin/python run.py build \
  --processed-dir data/processed/a-embedding/chy-0032 \
  --index-dir "$NEW_INDEX_DIR" --chunk-size 1000 --chunk-overlap 150
```
