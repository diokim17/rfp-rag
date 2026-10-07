# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

RFP-RAG: a retrieval-augmented QA pipeline over ~100 Korean public-procurement RFP documents (96 HWP, 4 PDF) plus a metadata CSV. Codeit AI team project (team 3, 6 people, 2026-09-29 to 2026-10-22). Only "Scenario B" (OpenAI API) is implemented; "Scenario A" (self-hosted models on GCP) is not.

Code comments, docstrings, CLI messages, docs and commit messages are written in **Korean**. Match that when editing existing files.

## Commands

Use the project venv (Python 3.12.3). There is no build step, linter, or formatter config.

```bash
.venv/bin/python -m pip install -r requirements-server.lock.txt   # full pinned env (requirements.txt = direct deps only)
.venv/bin/python -m unittest discover -s tests -v                 # all tests: offline, no .env, no API
.venv/bin/python -m unittest tests.test_retrieval_rerank -v       # one file
.venv/bin/python -m unittest tests.test_pipeline -k filter -v     # by name pattern
```

Pipeline CLI (`run.py`), run from the repo root:

```bash
python run.py parse --limit 3                      # free: no API, no key needed
python run.py build                                # embeds every chunk  -> costs money
python run.py ask --question "..." --top-k 5 --filter "발주 기관=한영대학" --max-per-doc 2
python run.py evaluate --eval-file data/eval.json  # one generation call per case -> costs money
python run.py all --limit 3 --question "..."       # parse + build + ask
```

Retrieval-only experiments (embedding calls only, no generation):

```bash
python exp_max_per_doc.py --eval-file data/eval_retrieval_yjk_e2e24.json --max-per-doc 2 3
python experiments/retrieval_eval_yjk.py run --index-dir ... --eval-file ... --results-dir ... --label baseline
python experiments/max_per_doc_effect_yjk.py --eval-file data/eval_retrieval_yjk.json --index-dir indexes
```

**API budget is a hard team constraint: $20 total, not enforced by code.** `build`, `ask`, `all`, `evaluate` and the experiment scripts call OpenAI. Do not run them without the user asking; prefer the offline tests to verify changes. Running `python run.py` with no arguments picks `ask` or `all` automatically and can trigger a full rebuild.

## Architecture

Flat layout, one module per pipeline stage, each owned by one teammate. `run.py` wires them together and is the only place that loads `.env` and creates the OpenAI client.

```
parsing.py  ->  embedding.py  ->  retrieval.py  ->  generation.py  ->  evaluation.py
(Document)      (Chunk, FAISS)    (Hit)             (Answer)           (summary, records)
```

| File | Owner | Role |
| --- | --- | --- |
| `run.py` | 김도영 (integration) | argparse CLI, experiment ID, tracing, result JSON |
| `parsing.py` | 나상훈 | HWP5 binary records via `olefile` + PDF via `pymupdf`; tables rebuilt as markdown; text cleanup; CSV join. Also home of `read_json` / `write_json`, used everywhere |
| `embedding.py` | 유찬혁 | character chunking (`fixed` or `boundary`), OpenAI embeddings, FAISS `IndexFlatIP` over L2-normalized vectors |
| `retrieval.py` | 김연주 | cosine search, metadata filters (exact match, falling back to a unique normalized-containment match), optional rerank (`lexical` char-bigram BM25, or local `cross-encoder`), per-document chunk cap |
| `generation.py` | 박단비 | OpenAI Responses API call with numbered citations |
| `evaluation.py` | 김시현 | doc-level Recall@k, keyword coverage, latency |
| `observability.py` | shared | optional Langfuse tracing: `Trace`, `@observed`, `model_call`, `record_retrieval` |
| `experiment_ids.py` | shared | per-owner run counter in `.experiment-state/experiments.sqlite3` |
| `experiment_reports.py` | shared | sanitized markdown summary of an `evaluate` run |
| `exp_max_per_doc.py`, `experiments/*_yjk.py` | 김연주 | personal retrieval experiments, not part of the pipeline |

`BM25.py` is an untracked scratch file (a shell heredoc is pasted at the end, so it is not valid Python). Ignore it.

### Data contracts

Everything passed between stages is plain JSON-serializable `dict`/`list`. No dataclasses, no DataFrames. Adding keys is fine; removing or renaming them is a contract change.

```
Document = {doc_id, text, metadata}
Chunk    = {chunk_id, doc_id, text, metadata + start_char, end_char}
Hit      = Chunk + {score}  (+ rerank_score when reranking)
Answer   = {question, answer, sources: [{citation, ...Hit}], model}
```

- `doc_id` = first 16 hex chars of SHA-256 of the NFC-normalized filename. Eval sets reference these IDs, so changing filenames or the ID rule invalidates them.
- `chunk_id` = `doc_id:N`. `text` must equal `document_text[start_char:end_char]`.
- `metadata` keeps the CSV's Korean column names (`사업명`, `발주 기관`, ...) as string values, plus `filename` and `source`. The CSV `텍스트` column is dropped; text always comes from the raw file.
- `score` is always cosine similarity, even after reranking.
- `sources` is the full hit list with 1-based `citation`, not only what the model cited. `evaluate` reads retrieved doc IDs from it, so do not trim it.
- `run.py` calls stage functions **positionally**; keep argument order. New options go in as keyword-only (see `retrieve(..., *, rerank, candidates, max_per_doc)`).

### Index

An index directory is three files that must stay in sync: `index.faiss`, `chunks.json`, `config.json`. FAISS vector *i* is `chunks[i]`; `load_index` checks counts and dimension. Queries are embedded with the model recorded in `config.json`, not the one in `.env`. The three files are not written atomically, so never query a directory while it is being built.

### Things that are easy to get wrong

- **`max_per_doc` + rerank**: without rerank the cap is applied while collecting hits; with rerank the full candidate pool is reranked first and the cap applied afterwards, so fewer than `top_k` hits can come back.
- **Filters or `max_per_doc`** make `retrieve` search the whole index (`index.ntotal`) and filter in Python.
- **Config via env vars, read at call time**: `RFP_CHUNKING_STRATEGY` (`fixed` | `boundary`), `RETRIEVAL_RERANK` (`none` | `lexical` | `cross-encoder`, default `cross-encoder`, falls back to `lexical` with a warning when torch/transformers are missing and it was not set explicitly), `RETRIEVAL_HYBRID` (default on), `RETRIEVAL_MAX_PER_DOC` (default 2, `none` disables), `RETRIEVAL_CANDIDATES` (default 50), `RETRIEVAL_HYBRID_VECTOR_K` / `RETRIEVAL_HYBRID_BM25_K` (default 100), `RETRIEVAL_RERANK_MODEL` (default `BAAI/bge-reranker-v2-m3`), `OPENAI_EMBEDDING_MODEL`, `OPENAI_GENERATION_MODEL` (only `gpt-5-mini` or `gpt-5-nano` are accepted), `EXPERIMENT_OWNER`, `LANGFUSE_*`.
- **Retrieval defaults** are hybrid + cross-encoder + `max_per_doc=2` (chosen from yjk-0017 on the team eval set). `tests/retrieval_baseline.py` provides a `setUpModule` that patches the module defaults back to plain cosine search so tests stay offline and deterministic; the retrieval tests import it, and `tests/test_retrieval_defaults.py` covers the real defaults with a fake cross-encoder. The other stage tests (`test_pipeline`, `test_run`, `test_observability`, `test_embedding`, `test_chunk_metadata`, `test_structured_embedding`) do not import it yet, so they run with the real defaults; they pass only because cross-encoder falls back to `lexical` when torch is absent. Adding the import there is up to each test's owner.
- **`cross-encoder`** needs `torch` and `transformers`, which are intentionally not in `requirements.txt`.
- **Every `run.py` invocation** needs an owner (`EXPERIMENT_OWNER` or `--owner`) and consumes an experiment number (`yjk-0007`), including `parse` and failed runs.
- **Imports must stay side-effect free**: no API calls, key loading, or file writes at import time. `run.py` imports everything at module scope (stdlib, third-party, project modules, in that order), so tests patch names on `run` itself. Optional heavy dependencies (`langfuse`, `torch`, `transformers`) are still imported lazily inside functions.
- **`limit`** in `parse_documents` means the first N CSV rows, not N successful documents. Per-file failures go to `parsing_errors.json` and the run continues.

## Privacy rules (enforced by design, keep them)

The RFP source documents may not be shared outside the team.

- Langfuse receives only IDs, scores, hashes, timings, token counts and exception *type names*. Never send questions, document text, answers, filenames, or filter values. That is why `observability.py` uses hand-picked fields instead of the OpenAI auto-instrumentation.
- `.gitignore` excludes `data/`, `indexes/`, `results/` (except `results/reports/*.md|*.csv`), `.env`, `.experiment-state/`. Result JSONs contain source text; only the sanitized reports are committable.
- Never print or log API keys. Never overwrite an existing `.env`.

## Working conventions

- Branch from `dev`, PR into `dev`. Branch names look like `feature/<topic>` or `retrieval/<topic>`. Commit messages are Korean, usually with a `feat:` / `fix:` / `docs:` / `merge:` prefix.
- Stay inside the owning module where possible. Changing a public signature, a required key, the `doc_id` rule, the index format, or shared dependency versions needs team agreement; update callers, tests and `docs/TEAM_DEVELOPMENT_GUIDE.md` together.
- Keep personal experiments in separate directories using `--processed-dir`, `--index-dir`, `--results-dir` (e.g. `indexes/parsing-v2-yjk`). Do not overwrite the shared default `data/processed` and `indexes`.
- Do not fold changes to `DEFAULT_COMMAND`, `DEFAULT_LIMIT`, `DEFAULT_QUESTION` or the default directories in `run.py` into feature PRs.
- Tests use fakes for the OpenAI client and must stay offline. Add edge-case tests for what you change; do not weaken existing ones.
- Add files to git by name, not `git add .`.

## Local state on this checkout (as of 2026-10-02)

Not derivable from git, and likely to drift:

- Branch `retrieval/max-per-doc` (PR #3 into `dev`). It carries the `parsing.py` rewrite from PR #2 (direct HWP5 record parsing with table reconstruction, `pymupdf` for PDF). `run.py` keeps the shared defaults `data/processed` and `indexes`; pass `--processed-dir data/processed/parsing-v2-yjk --index-dir indexes/parsing-v2-yjk` to use the table-restored data.
- `indexes/` (2,932 chunks, 98 docs) is the older baseline index; `indexes/parsing-v2-yjk/` (9,789 chunks, 100 docs) is built from the new parser. `exp_max_per_doc.py` defaults to the v2 index, `experiments/max_per_doc_effect_yjk.py` to the old one.
- `data/eval_retrieval_yjk.json` (325 cases) and `data/eval_retrieval_yjk_e2e24.json` (24 cases) are auto-generated from CSV metadata by `experiments/retrieval_eval_yjk.py make-evalset`. They are personal retrieval eval sets, not the team's shared eval set, which does not exist yet.
- `README.md` and `docs/TEAM_DEVELOPMENT_GUIDE.md` lag the code in places: they still describe `pyhwp`/`pypdf` parsing and say reranking is not implemented.

## Further reading

- `README.md`: project overview, run instructions, I/O contract table.
- `docs/TEAM_DEVELOPMENT_GUIDE.md`: server setup, ownership, contract rules, PR checklist.
- `docs/LANGFUSE_GUIDE.md`: tracing setup, experiment IDs, what is and is not recorded.
