"""김도영: 전처리 → 인덱스 → 검색 → 생성 → 평가 연결."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import sqlite3

from parsing import parse_documents, read_json, write_json

ROOT = Path(__file__).resolve().parent

# VS Code에서 인자 없이 실행할 때 사용할 설정입니다.
DEFAULT_COMMAND = "auto"  # 인덱스가 있으면 ask, 없으면 all. 재생성하려면 "all"
DEFAULT_LIMIT = 3
DEFAULT_QUESTION = "한영대학교 교육환경 구축 사업의 주요 요구사항은 무엇인가요?"


def main():
    parser = argparse.ArgumentParser(description="시나리오 B: 최소 RFP RAG 파이프라인")
    parser.add_argument("command", choices=["parse", "build", "ask", "all", "evaluate"])
    parser.add_argument("--question", help="ask/all 실행 질문")
    parser.add_argument("--owner", help="담당자 영문 이니셜 (기본: EXPERIMENT_OWNER)")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    parser.add_argument("--limit", type=int, help="전처리할 CSV 앞쪽 N행 (생략: 전체)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-doc", type=int, help="문서당 최대 청크 수 (생략: 제한 없음)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--eval-file", type=Path, help="수동 작성한 평가 JSON 파일")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    argv = sys.argv[1:]
    if not argv:
        command = DEFAULT_COMMAND
        if command == "auto":
            index_files = ("index.faiss", "chunks.json", "config.json")
            command = "ask" if all((ROOT / "indexes" / name).is_file() for name in index_files) else "all"
        argv = [command, "--question", DEFAULT_QUESTION]
        if DEFAULT_LIMIT is not None:
            argv += ["--limit", str(DEFAULT_LIMIT)]
        print(f"기본 실행: {command}")
        if command in {"parse", "all"}:
            print(f"처리할 문서 수: {DEFAULT_LIMIT or '전체'}")
        print(f"질문: {DEFAULT_QUESTION}")
    args = parser.parse_args(argv)
    if args.command in {"ask", "all"} and not (args.question or "").strip():
        parser.error("ask/all에는 --question이 필요합니다.")
    if args.command == "evaluate" and not args.eval_file:
        parser.error("evaluate에는 --eval-file이 필요합니다.")
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("top-k, limit은 1 이상이어야 합니다.")
    if args.max_per_doc is not None and args.max_per_doc < 1:
        parser.error("max-per-doc은 1 이상이어야 합니다.")    
    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("0 <= chunk-overlap < chunk-size 조건이 필요합니다.")
    filters = {}
    for value in args.filter:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            parser.error("--filter는 '발주 기관=기관명' 형태로 입력하세요.")
        filters[key] = item

    from dotenv import load_dotenv
    from observability import Trace, code_version
    from experiment_reports import file_hash
    from experiment_ids import owner_initials, next_experiment_id

    load_dotenv(ROOT / ".env", override=False)
    try:
        owner = owner_initials(args.owner or os.getenv("EXPERIMENT_OWNER", ""))
        experiment_id = next_experiment_id(owner, ROOT / ".experiment-state")
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment = {
        "experiment_id": experiment_id,
        "owner": owner,
        **code_version(ROOT),
        "evaluation_sha256": file_hash(args.eval_file) if args.eval_file else None,
        "documents_sha256": file_hash(args.processed_dir / "documents.json"),
        "index_sha256": file_hash(args.index_dir / "index.faiss"),
        "chunks_sha256": file_hash(args.index_dir / "chunks.json"),
        "index_config_sha256": file_hash(args.index_dir / "config.json"),
        "filters_sha256": hashlib.sha256(
            json.dumps(filters, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }
    trace = Trace.from_env({**experiment, "command": args.command, "top_k": args.top_k})
    print(f"실험 ID: {experiment_id} (담당자: {owner})")
    print("Langfuse: " + ("기록 활성화" if trace.client else "비활성화 (로컬 결과 저장)"))
    with trace.run(f"{experiment_id}-{args.command}"):
        run_pipeline(args, parser, filters, experiment, trace, timestamp)


def run_pipeline(args, parser, filters, experiment, trace, timestamp):
    if args.command in {"parse", "all"}:
        documents = parse_documents(args.raw_dir, args.processed_dir, args.limit)
        errors = read_json(args.processed_dir / "parsing_errors.json")
        print(f"전처리: 성공 {len(documents)}건, 실패 {len(errors)}건")
        from experiment_reports import file_hash
        experiment["documents_sha256"] = file_hash(args.processed_dir / "documents.json")
        with trace.span("parsing-summary", output={"documents": len(documents), "errors": len(errors)},
                        metadata={"documents_sha256": experiment["documents_sha256"]}):
            pass
        if args.command == "parse":
            return

    # 사용자가 실행할 때만 .env를 로드하며 키를 출력하거나 결과에 저장하지 않습니다.
    from openai import OpenAI, RateLimitError
    from embedding import build_index, load_index
    from retrieval import retrieve
    from generation import generate_answer

    if not os.getenv("OPENAI_API_KEY"):
        parser.error(".env 또는 환경 변수에 OPENAI_API_KEY를 설정하세요.")
    client = OpenAI(timeout=60.0, max_retries=2)
    generation_model = os.getenv("OPENAI_GENERATION_MODEL", "gpt-5-mini")
    if generation_model not in {"gpt-5-mini", "gpt-5-nano"}:
        parser.error("허용된 답변 모델은 gpt-5-mini, gpt-5-nano입니다. "
                     ".env의 OPENAI_GENERATION_MODEL을 수정하거나 삭제하세요.")
    if args.command in {"build", "all"}:
        documents = read_json(args.processed_dir / "documents.json")
        config = build_index(documents, client, args.index_dir,
                             os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                             args.chunk_size, args.chunk_overlap)
        print(f"인덱스 생성: {config['chunk_count']}개 청크")
        from experiment_reports import file_hash
        for key, filename in (("index_sha256", "index.faiss"), ("chunks_sha256", "chunks.json"),
                              ("index_config_sha256", "config.json")):
            experiment[key] = file_hash(args.index_dir / filename)
        with trace.span("build-settings", metadata={**config, **experiment}):
            pass
        if args.command == "build":
            return

    index, chunks, config = load_index(args.index_dir)
    with trace.span("index-settings", metadata={**config, "generation_model": generation_model,
                                               "top_k": args.top_k}):
        pass

    from observability import observed

    @observed("question-answer")
    def answer(question, case_filters=None):
        with trace.span("question-settings", metadata={
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "filter_keys": sorted({**filters, **(case_filters or {})})}):
            pass
        hits = retrieve(question, client, index, chunks, config, args.top_k,
                        {**filters, **(case_filters or {})}), , max_per_doc=args.max_per_doc)
        try:
            return generate_answer(question, hits, client, generation_model)
        except RateLimitError as exc:
            import re
            if re.search(r"RPM\).*?Limit\s+0(?:\D|$)", str(exc)):
                detail = "분당 요청 한도(RPM)가 0입니다. 대기하거나 질문을 줄여도 해결되지 않습니다."
            elif exc.code == "insufficient_quota":
                detail = "API 사용 가능 할당량이 부족합니다. 결제·크레딧·사용 한도를 확인하세요."
            else:
                detail = "요청/토큰 사용 한도에 도달했습니다. 한도를 확인하고 잠시 후 재시도하세요."
            parser.exit(1, f"\n답변 생성 실패 ({generation_model}): {detail}\n"
                        "OpenAI 프로젝트의 Limits에서 해당 모델의 사용 한도를 확인하세요.\n"
                        "다른 사용 가능한 모델로 바꾸려면 .env의 OPENAI_GENERATION_MODEL을 설정하세요.\n"
                        "인덱스는 저장되어 있으므로 기본 실행 ask로 재시도할 수 있습니다.\n")

    if args.command == "evaluate":
        from evaluation import evaluate
        result = evaluate(read_json(args.eval_file), answer)
        print(result["summary"])
    else:
        result = answer(args.question)
        print(result["answer"])
        for source in result["sources"]:
            print(f"[{source['citation']}] {source['metadata']['filename']} "
                  f"(doc_id={source['doc_id']}, score={source['score']:.3f})")
    result["settings"] = {**config, "generation_model": generation_model,
                          "top_k": args.top_k, "max_per_doc": args.max_per_doc, "filters": filters}
    result["experiment"] = {**experiment, "trace_id": trace.trace_id}
    result["token_usage"] = trace.usage
    if args.command == "evaluate":
        from experiment_reports import save_report
        trace.scores(result["summary"])
        report = save_report(args.reports_dir, timestamp, result["summary"],
                             result["settings"], result["experiment"], trace.usage)
        print(f"팀 공유용 요약: {report}")
    output = args.results_dir / f"{args.command}_{experiment['experiment_id']}_{timestamp}.json"
    write_json(output, result)
    print(f"결과 저장: {output}")


if __name__ == "__main__":
    main()
