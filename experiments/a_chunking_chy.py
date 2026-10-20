"""A 청킹 통제 실험. 생성/API 호출 없이 compare 또는 기존 인덱스 evaluate.

각 조건은 별도 프로세스·자동 실험 ID를 사용한다. 기본 파이프라인은 변경하지 않는다.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from pipelines.server.server_client import ScenarioAClient
from pipelines.server.server_embedding import build_index, chunk_documents, load_index
from evaluation.evaluation_metrics import _answer_chunk_hit
from tracking.tracking_ids import next_experiment_id, owner_initials
from tracking.tracking_reports import file_hash, text_file_hash
from observability.observability_trace import Trace, code_version
from parsing.parsing_documents import read_json, write_json
from pipelines.server.server_retrieval import retrieve, retrieval_options
from experiments.a_embedding_chy import verify_model, sha256
from experiments.a_retrieval_metrics import score_case
from config.config_paths import model_directory

MODEL_ROOT = model_directory('embedding')
RERANK_ROOT = model_directory('rerank')
MODEL_NAME = 'default_model'
MODEL_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
VARIANTS = {'fixed': ('fixed', '1'), 'boundary': ('boundary', '1'),
            'structured-v2': ('structured', '2'), 'structured-v3': ('structured', '3')}
COMMON = dict(rewrite='off', within_doc='off', dedup=False, min_score='none')
PROFILES = {
    'dense': dict(COMMON, rerank='none', hybrid=False, max_per_doc='none', expand=0,
                  bm25_prefix=False),
    'a-default': dict(COMMON, rerank='cross-encoder', hybrid=True, candidates=50,
                      hybrid_vector_k=100, hybrid_bm25_k=100, rrf_k=60,
                      max_per_doc=2, bm25_prefix=True, expand=5),
}


def configure(variant, context="none"):
    strategy, version = VARIANTS[variant]
    os.environ.update(RFP_CHUNKING_STRATEGY=strategy, RFP_STRUCTURED_CHUNKING_VERSION=version,
                      RFP_EMBEDDING_CONTEXT=context, RETRIEVAL_REWRITE='off',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def code_hashes():
    paths = list(ROOT.joinpath('pipelines/server').glob('server_*.py'))
    for folder in ('parsing', 'evaluation', 'observability', 'tracking', 'retrieval', 'config'):
        paths.extend(ROOT.joinpath(folder).glob('*.py'))
    paths.extend([Path(__file__), ROOT/'experiments/a_embedding_chy.py',
                  ROOT/'experiments/a_retrieval_metrics.py'])
    return {str(p.relative_to(ROOT)): text_file_hash(p) for p in sorted(paths)}


def validate_inputs(documents, cases):
    if not documents or not cases:
        raise ValueError('Empty documents or evaluation')
    ids = [d['doc_id'] for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate document IDs')
    for d in documents:
        if not isinstance(d['text'], str) or not isinstance(d['metadata'], dict):
            raise ValueError('Invalid Document contract')
    case_ids = [c['id'] for c in cases]
    if any(not isinstance(i, str) or not i for i in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError('Invalid or duplicate case IDs')
    for c in cases:
        gold = c.get('expected_doc_ids', [])
        if not isinstance(c.get('question'), str) or not c['question'].strip():
            raise ValueError('Missing question')
        if not isinstance(gold, list) or not set(gold) <= set(ids):
            raise ValueError('Missing gold documents')
        if c.get('answerable', True) and not gold:
            raise ValueError('Answerable case requires gold documents')
    if not any(c.get('answerable', True) for c in cases):
        raise ValueError('No answerable cases')


def geometry(documents, chunks, chunk_size=1000, chunk_overlap=150):
    """원문 기반 독립 검사. 청킹 구현의 경계 검출 함수를 재사용하지 않는다."""
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c['doc_id']].append(c)
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Duplicate chunk IDs')
    if set(grouped) - {d['doc_id'] for d in documents}:
        raise ValueError('Unknown chunk document')
    lengths = [len(c['text']) for c in chunks]
    result = dict(chunk_count=len(chunks), length_min=min(lengths), length_median=statistics.median(lengths),
                  length_p95=float(np.percentile(lengths, 95)), length_max=max(lengths),
                  text_chars=sum(lengths), overlap_chars=0, short_under_150=sum(n < 150 for n in lengths),
                  small_tables=0, small_tables_not_whole=0, table_midrow_boundaries=0,
                  section_crossings=0)
    for d in documents:
        text = d['text']; items = grouped[d['doc_id']]
        covered = bytearray(len(text)); last_start, last_end = -1, 0
        boundaries = set()
        for c in items:
            start, end = c['metadata']['start_char'], c['metadata']['end_char']
            if not (0 <= start < end <= len(text) and end-start <= chunk_size
                    and start > last_start and end > last_end and max(0, last_end-start) <= chunk_overlap
                    and c['text'] == text[start:end] and c['text'].strip()
                    and c['chunk_id'].startswith(d['doc_id']+':')
                    and all(c['metadata'].get(k) == v for k,v in d['metadata'].items())):
                raise ValueError('Chunk coordinate/metadata contract violated')
            covered[start:end] = b'\1'*(end-start)
            result['overlap_chars'] += max(0,last_end-start)
            result['section_crossings'] += sum(start < s[0] < end for s in d.get('sections', [])
                                               if isinstance(s,(list,tuple)) and len(s)==2 and type(s[0]) is int)
            boundaries.update((start,end)); last_start,last_end=start,end
        if any(not covered[i] for i,char in enumerate(text) if not char.isspace()):
            raise ValueError('Uncovered source text')
        for m in re.finditer(r'<!-- table:(T\d+) -->.*?<!-- /table:\1 -->',text,re.S):
            if m.end()-m.start() <= chunk_size:
                result['small_tables'] += 1
                result['small_tables_not_whole'] += not any(
                    c['metadata']['start_char'] <= m.start() and c['metadata']['end_char'] >= m.end() for c in items)
            result['table_midrow_boundaries'] += sum(
                m.start()<pos<m.end() and text[pos-1]!='\n' for pos in boundaries)
    if json.loads(json.dumps(chunks,ensure_ascii=False)) != chunks:
        raise ValueError('Chunk JSON round-trip changed data')
    return result


def case_record(case, hits, repetition, seconds, error=None):
    answerable=case.get('answerable',True)
    metrics=score_case(case['expected_doc_ids'],[h['doc_id'] for h in hits],5) if answerable else dict(recall=None,mrr=None)
    return dict(id=case['id'], repetition=repetition, answerable=answerable, seconds=seconds,
                error=error, **metrics, answer_chunk_hit=_answer_chunk_hit(case,hits) if answerable else None,
                hit_count=len(hits), top_score=hits[0]['score'] if hits else None,
                hits=[{k:h[k] for k in ('doc_id','chunk_id','score','rerank_score') if k in h} for h in hits])


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(records):
    positive=[r for r in records if r['answerable']]
    negative=[r for r in records if not r['answerable']]
    times=[r['seconds'] for r in records]
    return dict(recall_at_5=mean([r['recall'] for r in positive]),
                mrr_at_5=mean([r['mrr'] for r in positive]),
                answer_chunk_hit_rate=mean([r['answer_chunk_hit'] for r in positive]),
                answerable_cases=len({r['id'] for r in positive}),
                chunk_metric_cases=len({r['id'] for r in positive if r['answer_chunk_hit'] is not None}),
                unanswerable_cases=len({r['id'] for r in negative}),
                negative_empty_rate=mean([float(r['hit_count']==0) for r in negative if r['error'] is None]),
                negative_mean_hit_count=mean([r['hit_count'] for r in negative if r['error'] is None]),
                negative_mean_top_score=mean([r['top_score'] for r in negative if r['error'] is None]),
                mean_seconds=mean(times), median_seconds=statistics.median(times),
                p95_seconds=float(np.percentile(times,95)), failures=sum(r['error'] is not None for r in records))


def rank_key(result):
    s=result['summary']
    if s['failures']:
        raise ValueError('Cannot rank a failed condition')
    return (s['recall_at_5'],s['answer_chunk_hit_rate'] if s['answer_chunk_hit_rate'] is not None else -1,
            s['mrr_at_5'],-result['config']['chunk_count'])


def fingerprints(index_dir):
    return {name:(file_hash if name=='index.faiss' else text_file_hash)(index_dir/name)
            for name in ('index.faiss','chunks.json','config.json')}


def validate_reuse(index_dir, manifest, document_hash, evaluation_hash, model_manifest_hash, implementation):
    config=read_json(index_dir/'config.json')
    expected={'documents_sha256':document_hash, 'evaluation_sha256':evaluation_hash,
              'model_manifest_sha256':model_manifest_hash, 'code_hashes':implementation}
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError('Saved input/model/code provenance differs; rebuild or use original snapshot')
    if manifest.get('index_hashes') != fingerprints(index_dir) or config.get('documents_sha256') != document_hash:
        raise ValueError('Saved index or documents changed')
    return config


def report_run(result, path):
    # Explicit allowlist: no case text, source text, filenames, filters, or arbitrary metadata.
    s=result['summary']; cfg=result['config']
    lines=['# A 청킹 검색 실험', '', '| 항목 | 값 |','| --- | --- |']
    fields={'experiment_id':result['experiment']['experiment_id'], 'variant':result['variant'],
            'profile':result['profile'], 'commit':result['experiment']['commit'],
            'embedding_model':cfg['embedding_model'], 'chunk_count':cfg['chunk_count'],
            'dimension':cfg['dimension'], 'chunk_size':cfg.get('chunk_size'),
            'chunk_overlap':cfg.get('chunk_overlap'), 'embedding_context':cfg.get('embedding_context'), **s, **result['timing']}
    lines += [f'| {k} | {v} |' for k,v in fields.items()]
    lines += ['', '로컬 토큰·실패 통계:', '```json',json.dumps(result['stats'],indent=2), '```',
              '', '근거 청크 적중률은 기존 값·키워드 기반 대리지표이며 답변 정확도가 아닙니다.',
              '답변 불가 문항의 빈 검색 결과 비율은 거절 정확도가 아닙니다. 생성·판정 API 호출은 0회입니다.', '']
    path.write_text('\n'.join(lines),encoding='utf-8')


def wait_for_gpu(torch, profile, wait_seconds=600):
    if not torch.cuda.is_available():
        raise RuntimeError('This GPU comparison requires CUDA')
    deadline=time.monotonic()+wait_seconds
    required=(6 if profile=='dense' else 8)*1024**3
    while True:
        free,_=torch.cuda.mem_get_info()
        if free>=required:
            return
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise RuntimeError('Insufficient free GPU memory; retry when shared GPU is available')
        print(f'Waiting for shared GPU: {free//1024**2} MiB free; {required//1024**2} MiB needed',flush=True)
        time.sleep(min(30,remaining))


def run_condition(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.a',override=False)
    owner=owner_initials(os.getenv('EXPERIMENT_OWNER',''))
    if os.getenv('A_EMBEDDING_MODEL') != MODEL_NAME:
        raise ValueError('This experiment pins A_EMBEDDING_MODEL=default_model')
    if args.repeat < 1:
        raise ValueError('repeat must be positive')
    source_docs=args.processed_dir/'documents.json' if args.command=='evaluate' else args.documents
    source_eval=args.processed_dir/'eval.json' if args.command=='evaluate' else args.eval_file
    input_hashes=(text_file_hash(source_docs),text_file_hash(source_eval))
    documents,cases=read_json(source_docs),read_json(source_eval)
    validate_inputs(documents,cases)
    model_manifest=verify_model(MODEL_ROOT/MODEL_NAME)
    if model_manifest['repo_id']!='BAAI/bge-m3' or model_manifest['revision']!=MODEL_REVISION:
        raise ValueError('Pinned embedding model revision differs')
    model_hash=file_hash(MODEL_ROOT/MODEL_NAME/'local_manifest.json')
    implementation=code_hashes()
    if args.command=='evaluate':
        old=read_json(args.processed_dir/'manifest.json')
        config=validate_reuse(args.index_dir,old,text_file_hash(source_docs),text_file_hash(source_eval),model_hash,implementation)
        variant=old['variant']
        chunk_size, chunk_overlap = config['chunk_size'], config['chunk_overlap']
        context = config.get('embedding_context', 'none')
    else:
        variant=args.variant
        chunk_size, chunk_overlap = args.chunk_size, args.chunk_overlap
        context = args.embedding_context
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError('Require chunk_size > chunk_overlap >= 0')
    configure(variant, context)
    expected=getattr(args,'expected',None)
    if expected:
        for actual,wanted in zip((text_file_hash(source_docs),text_file_hash(source_eval),model_hash,
                                  hashlib.sha256(json.dumps(implementation,sort_keys=True).encode()).hexdigest()),expected):
            if actual!=wanted:
                raise ValueError('Comparison inputs or code changed between conditions')
    profile=args.profile
    rerank_files={}
    if profile=='a-default':
        if os.getenv('A_RERANK_MODEL')!='bge-reranker-v2-m3':
            raise ValueError('This profile pins A_RERANK_MODEL=bge-reranker-v2-m3')
        p=RERANK_ROOT/'bge-reranker-v2-m3'
        rerank_files={str(f.relative_to(p)):sha256(f) for f in sorted(p.rglob('*'))
                      if f.is_file() and '.cache' not in f.relative_to(p).parts}
        if not rerank_files:
            raise ValueError('Reranker files missing')
    import torch
    torch.set_num_threads(4)
    wait_for_gpu(torch,profile,args.gpu_wait_seconds)
    experiment_id=next_experiment_id(owner,ROOT/'.experiment-state/a')
    paths={name:ROOT/base/'a-chunking'/experiment_id for name,base in
           (('processed','data/processed'),('index','indexes'),('results','results'))}
    for name in ('processed','results'):
        paths[name].mkdir(parents=True,exist_ok=False)
    if args.command!='evaluate': paths['index'].mkdir(parents=True,exist_ok=False)
    for source,name in ((source_docs,'documents.json'),(source_eval,'eval.json')):
        (paths['processed']/name).write_bytes(source.read_bytes())
    documents,cases=read_json(paths['processed']/'documents.json'),read_json(paths['processed']/'eval.json')
    validate_inputs(documents,cases)
    if ((text_file_hash(paths['processed']/'documents.json'),text_file_hash(paths['processed']/'eval.json'))
            != input_hashes):
        raise ValueError('Input changed during snapshot')
    experiment=dict(experiment_id=experiment_id,owner=owner,**code_version(ROOT),
                    measured_at_utc=datetime.now(timezone.utc).isoformat(),
                    documents_sha256=text_file_hash(paths['processed']/'documents.json'),
                    evaluation_sha256=text_file_hash(paths['processed']/'eval.json'),
                    model_manifest_sha256=model_hash,code_hashes=implementation)
    print(f'{experiment_id}: {variant}/{profile}; {len(documents)} documents, {len(cases)} cases',flush=True)
    trace=Trace.from_env(experiment)
    manifest=dict(experiment,variant=variant,model_files=model_manifest['files'],rerank_files=rerank_files)
    write_json(paths['processed']/'manifest.json',manifest)
    try:
        with trace.run('a-chunking-comparison'):
            started=time.perf_counter()
            client=ScenarioAClient({'A_EMBEDDING_MODEL':MODEL_NAME,'A_GENERATION_MODEL':'unused'},
                                   MODEL_ROOT,MODEL_ROOT,need_generation=False)
            timing={'model_load_seconds':time.perf_counter()-started}
            print(f'{experiment_id}: model loaded',flush=True)
            torch.cuda.reset_peak_memory_stats()
            index_dir=args.index_dir if args.command=='evaluate' else paths['index']
            stats_before=dict(client.embedding.stats)
            start=time.perf_counter()
            if args.command!='evaluate':
                config=build_index(documents,client,index_dir,MODEL_NAME,chunk_size,chunk_overlap,
                                   documents_sha256=experiment['documents_sha256'])
            timing['build_seconds']=time.perf_counter()-start if args.command!='evaluate' else None
            built=dict(client.embedding.stats)
            index,chunks,config=load_index(index_dir)
            print(f'{experiment_id}: index ready ({len(chunks)} chunks)',flush=True)
            start=time.perf_counter()
            expected_chunks=chunk_documents(documents,chunk_size,chunk_overlap)
            timing['chunking_seconds']=time.perf_counter()-start
            if chunks!=expected_chunks: raise ValueError('Saved chunks differ from current chunker')
            shape=geometry(documents,chunks,chunk_size,chunk_overlap)
            for start in range(0,index.ntotal,1024):
                vectors=index.reconstruct_n(start,min(1024,index.ntotal-start))
                if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
                    raise ValueError('Invalid stored vector normalization')
            manifest['index_hashes']=fingerprints(index_dir)
            write_json(paths['processed']/'manifest.json',manifest)
            options=PROFILES[profile]
            first=next(c for c in cases if c.get('answerable',True))
            start=time.perf_counter()
            retrieve(first['question'],client,index,chunks,config,5,first.get('filters'),**options)
            timing['warmup_seconds']=time.perf_counter()-start
            warm=dict(client.embedding.stats)
            records=[]
            for repetition in range(args.repeat):
                for c in cases:
                    torch.cuda.synchronize();start=time.perf_counter();error=None
                    try:
                        hits=retrieve(c['question'],client,index,chunks,config,5,c.get('filters'),**options)
                    except Exception as exc:
                        hits=[];error=type(exc).__name__
                    torch.cuda.synchronize()
                    records.append(case_record(c,hits,repetition,time.perf_counter()-start,error))
                print(f'{experiment_id}: repetition {repetition+1}/{args.repeat}',flush=True)
            summary=summarize(records)
            trace.scores(summary)
            delta=lambda after,before:{k:after[k]-before[k] for k in after}
            stats={'build':delta(built,stats_before),'warmup':delta(warm,built),
                   'evaluation':delta(client.embedding.stats,warm),'final_batch_size':client.embedding.batch_size}
            if fingerprints(index_dir)!=manifest['index_hashes'] or code_hashes()!=implementation:
                raise ValueError('Index/code changed during evaluation')
            if verify_model(MODEL_ROOT/MODEL_NAME)!=model_manifest:
                raise ValueError('Model changed during evaluation')
            for relative,digest in rerank_files.items():
                if sha256(RERANK_ROOT/'bge-reranker-v2-m3'/relative)!=digest:
                    raise ValueError('Reranker changed during evaluation')
            result=dict(experiment={**experiment,'trace_id':trace.trace_id},variant=variant,profile=profile,
                        config=config,options=retrieval_options(**options),geometry=shape,summary=summary,
                        repetitions=[summarize([r for r in records if r['repetition']==i]) for i in range(args.repeat)],
                        timing=timing,stats=stats,token_usage=trace.usage,records=records,
                        paths={'processed':str(paths['processed']),'index':str(index_dir),'results':str(paths['results'])},
                        model=dict(repo_id=model_manifest['repo_id'],revision=model_manifest['revision'],
                                   device=str(client.embedding.model.device),dtype=str(client.embedding.model.dtype),
                                   max_length=client.embedding.model.max_seq_length,cpu_threads=torch.get_num_threads()),
                        packages={p:importlib.metadata.version(p) for p in
                                  ('torch','transformers','sentence-transformers','numpy','faiss-cpu')},
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),rerank_files=rerank_files)
        write_json(paths['results']/'evaluation.json',result)
        report_path=ROOT/'results/reports'/f'a_chunking_{experiment_id}.md'
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_run(result,report_path)
        if summary['failures']: raise RuntimeError('Search failures; condition cannot be selected')
        if args.result_pointer: Path(args.result_pointer).write_text(str(paths['results']/'evaluation.json'))
        print(json.dumps(summary),flush=True)
        return result
    except Exception as exc:
        write_json(paths['results']/'failure.json',{'experiment':experiment,'error':type(exc).__name__})
        raise


def compare(args):
    from dotenv import load_dotenv
    import tempfile
    load_dotenv(ROOT/'.env.a',override=False)
    implementation=code_hashes()
    expected=[text_file_hash(args.documents),text_file_hash(args.eval_file),
              file_hash(MODEL_ROOT/MODEL_NAME/'local_manifest.json'),
              hashlib.sha256(json.dumps(implementation,sort_keys=True).encode()).hexdigest()]
    results=[]
    def child(command):
        with tempfile.TemporaryDirectory(prefix='a-chunking-') as temp:
            pointer=Path(temp)/'result.txt'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),*command,'--repeat',str(args.repeat),
                            '--expected',*expected,'--gpu-wait-seconds',str(args.gpu_wait_seconds),'--result-pointer',str(pointer)],check=True,cwd=ROOT)
            return read_json(Path(pointer.read_text()))
    for variant in VARIANTS:
        results.append(child(['run','--variant',variant,'--profile','dense','--documents',str(args.documents.resolve()),
                              '--eval-file',str(args.eval_file.resolve())]))
    candidate=max(results[1:],key=rank_key)
    for saved in (results[0],candidate):
        results.append(child(['evaluate','--processed-dir',saved['paths']['processed'],
                              '--index-dir',saved['paths']['index'],'--profile','a-default']))
    if results[-2]['rerank_files']!=results[-1]['rerank_files']:
        raise ValueError('Reranker changed between conditions')
    report=ROOT/'results/reports'/f"a_chunking_comparison_{results[0]['experiment']['experiment_id']}.md"
    lines=['# A 청킹 비교 결과', '', '| 조건 | 검색 | ID | Recall@5 | MRR@5 | 근거 청크 적중률 | 평균 ms | p95 ms | 청크 |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        s=r['summary']
        lines.append(f"| {r['variant']} | {r['profile']} | {r['experiment']['experiment_id']} | {s['recall_at_5']:.6f} | {s['mrr_at_5']:.6f} | {s['answer_chunk_hit_rate']} | {s['mean_seconds']*1000:.2f} | {s['p95_seconds']*1000:.2f} | {r['config']['chunk_count']} |")
    lines += ['',f"추가 검증 후보: {candidate['variant']}. 기본 설정은 변경하지 않았습니다.",
              f"답변 가능 {results[0]['summary']['answerable_cases']}문항으로 검색 품질을 집계합니다. 근거 청크 적중률은 적용 가능한 문항에 한정됩니다.",
              '세 반복은 시간 측정용이며 독립 평가 문항 수를 늘리지 않습니다. API 호출 0회, 답변 품질은 미측정입니다.',
              '', '## 재현 입력', '', '```json',json.dumps({
                  'documents_sha256':expected[0],'evaluation_sha256':expected[1],
                  'model_manifest_sha256':expected[2],'code_hashes':implementation},indent=2),'```','']
    report.write_text('\n'.join(lines),encoding='utf-8')
    write_json(Path(results[0]['paths']['results'])/'comparison.json',
               {'conditions':[str(Path(r['paths']['results'])/'evaluation.json') for r in results],
                'candidate':candidate['variant']})
    print(f'COMPARISON_REPORT={report}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('compare','run','evaluate'):
        p=sub.add_parser(name)
        p.add_argument('--repeat',type=int,default=3)
        p.add_argument('--gpu-wait-seconds',type=int,default=600)
        if name in ('compare','run'):
            p.add_argument('--documents',type=Path,default=ROOT/'data/processed/documents.json')
            p.add_argument('--eval-file',type=Path,default=ROOT/'data/eval_v4.json')
        if name=='run':
            p.add_argument('--variant',choices=VARIANTS,required=True)
            p.add_argument('--chunk-size',type=int,default=1000)
            p.add_argument('--chunk-overlap',type=int,default=150)
            p.add_argument('--embedding-context', choices=('none','project','project_blend','table','section'), default='none')
        if name=='evaluate':
            p.add_argument('--processed-dir',type=Path,required=True)
            p.add_argument('--index-dir',type=Path,required=True)
        if name!='compare':
            p.add_argument('--profile',choices=PROFILES,default='dense')
            p.add_argument('--result-pointer',help=argparse.SUPPRESS)
            p.add_argument('--expected',nargs=4,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.repeat<1: parser.error('repeat must be positive')
    if args.gpu_wait_seconds<0: parser.error('gpu-wait-seconds must be nonnegative')
    if args.command=='compare': compare(args)
    else: run_condition(args)


if __name__=='__main__':
    main()
