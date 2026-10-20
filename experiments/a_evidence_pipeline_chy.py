"""기존 A 인덱스 3조건: 최신 기본 검색, 단계별 근거 누락 진단. API 호출 없음."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from dotenv import load_dotenv
import pipelines.server.server_retrieval as retrieval
from pipelines.server.server_embedding import load_index
from pipelines.server.server_client import ScenarioAClient
from evaluation.evaluation_metrics import _answer_chunk_hit
from experiments import a_chunking_chy as base

OPTIONS = dict(rerank='cross-encoder', candidates=50, max_per_doc=2, hybrid=True,
               hybrid_vector_k=100, hybrid_bm25_k=100, rrf_k=60, bm25_prefix=True,
               rewrite='off', within_doc='off', expand=5, dedup=False, min_score=.53,
               rerank_section=True, uncap_named=False)
SOURCES = [('fixed-none', 'chy-0011'), ('fixed-blend20', 'chy-0013'),
           ('structured-v3-none', 'chy-0005')]


def read_source(name, source_id):
    result_path = ROOT/'results/a-chunking'/source_id/'evaluation.json'
    result = base.read_json(result_path)
    paths = {k:Path(v) for k,v in result['paths'].items()}
    manifest = base.read_json(paths['processed']/'manifest.json')
    if base.fingerprints(paths['index']) != manifest['index_hashes']:
        raise ValueError('Source index hash mismatch')
    for file, key in [('documents.json','documents_sha256'), ('eval.json','evaluation_sha256')]:
        if base.text_file_hash(paths['processed']/file) != result['experiment'][key]:
            raise ValueError('Source input hash mismatch')
    index, chunks, config = load_index(paths['index'])
    if config != result['config'] or config['documents_sha256'] != result['experiment']['documents_sha256']:
        raise ValueError('Source config mismatch')
    expected=('structured',3,'none') if name=='structured-v3-none' else ('fixed',1,'project_blend' if name=='fixed-blend20' else 'none')
    if (config['chunking_strategy'],config['chunking_version'],config['embedding_context']) != expected:
        raise ValueError('Source condition mismatch')
    if name=='fixed-blend20' and (config['embedding_context_version'] != 1 or config.get('project_context_weight',.2) != .2):
        raise ValueError('Source blend weight mismatch')
    if (config['chunk_size'],config['chunk_overlap'],config['embedding_model']) != (1000,150,base.MODEL_NAME):
        raise ValueError('Source geometry/model mismatch')
    vectors=index.reconstruct_n(0,index.ntotal)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
        raise ValueError('Invalid source vectors')
    return dict(name=name, source_result=str(result_path), result=result, paths=paths,
                index=index, chunks=chunks, config=config, index_hashes=manifest['index_hashes'])


@contextmanager
def capture_stages():
    """원래 함수를 그대로 호출하며 입력·출력 목록을 변경하지 않습니다."""
    captured={}
    expand, rerank = retrieval._expand_candidates, retrieval._rerank
    def expanded(*args, **kwargs):
        captured['fused'] = list(args[1])
        result=expand(*args,**kwargs)
        captured['expanded'] = list(result)
        return result
    def ranked(*args, **kwargs):
        captured['candidates'] = list(args[1])
        result=rerank(*args,**kwargs)
        captured['ranked'] = list(result)
        return result
    with patch.object(retrieval,'_expand_candidates',expanded), patch.object(retrieval,'_rerank',ranked):
        yield captured


def diagnose(case, chunks, documents, captured, hits):
    """기존 값/키워드 대리지표의 단계 추적. 원인 확정은 근거 원문 확인 필요."""
    gold=set(case.get('expected_doc_ids',[]))
    gold_chunks=[c for c in chunks if c['doc_id'] in gold]
    matching={c['chunk_id'] for c in gold_chunks if _answer_chunk_hit(case,[c])==1}
    applicable=_answer_chunk_hit(case,[]) is not None and case.get('answerable',True)
    result={'applicable':applicable,'matching_chunk_ids':sorted(matching)}
    for stage in ('fused','expanded','candidates','ranked'):
        rows=captured.get(stage,[])
        result[stage+'_ids']=[c['chunk_id'] for c in rows]
        result[stage+'_evidence_rank']=next((i for i,c in enumerate(rows,1) if c['chunk_id'] in matching),None)
    result['final_ids']=[h['chunk_id'] for h in hits]
    result['final_evidence_rank']=next((i for i,h in enumerate(hits,1) if h['chunk_id'] in matching),None)
    result['uncapped_top5_hit']=any(c['chunk_id'] in matching for c in captured.get('ranked',[])[:5])
    if not applicable:
        category='not_scored'
    elif result['final_evidence_rank'] is not None:
        category='hit'
    elif not matching:
        category='chunk_coverage_or_annotation' if _answer_chunk_hit(case,documents)==1 else 'annotation_or_source'
    elif not captured.get('ranked'):
        category='filter_or_threshold'
    elif result['candidates_evidence_rank'] is None:
        category='candidate_miss'
    elif result['uncapped_top5_hit']:
        category='document_cap_loss'
    else:
        category='rerank_below_top5'
    result['category']=category
    return result


def rerank_token_counts(question, candidates):
    """같은 tokenizer로 입력 쌍의 토큰 수만 재계산. 추론 없이 측정 시간 밖에서 실행."""
    if not candidates:
        return dict(pairs=0,input_tokens=0)
    tokenizer,_,_=retrieval._cross_encoders[retrieval.rerank_model_source()]
    encoded=tokenizer([question]*len(candidates),
        [retrieval._with_project(c,True) for c in candidates],padding=False,
        truncation='only_second',max_length=1024)
    return dict(pairs=len(candidates),input_tokens=sum(map(len,encoded['input_ids'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat',type=int,default=3)
    args=parser.parse_args()
    if args.repeat<1:parser.error('repeat must be positive')
    load_dotenv(ROOT/'.env',override=False)
    load_dotenv(ROOT/'.env.a',override=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
    if os.getenv('A_RERANK_MODEL')!='bge-reranker-v2-m3':
        raise ValueError('Expected local bge-reranker-v2-m3')
    owner=base.owner_initials(os.getenv('EXPERIMENT_OWNER',''))
    sources=[read_source(*s) for s in SOURCES]
    for key in ('documents_sha256','evaluation_sha256','model_manifest_sha256'):
        if len({s['result']['experiment'][key] for s in sources})!=1:
            raise ValueError('Source inputs/models differ')
    model=base.verify_model(base.MODEL_ROOT/base.MODEL_NAME)
    if base.file_hash(base.MODEL_ROOT/base.MODEL_NAME/'local_manifest.json')!=sources[0]['result']['experiment']['model_manifest_sha256']:
        raise ValueError('Current model differs from source')
    rerank_path=Path(retrieval.rerank_model_source())
    rerank_files={str(p.relative_to(rerank_path)):base.sha256(p) for p in sorted(rerank_path.rglob('*'))
                  if p.is_file() and '.cache' not in p.relative_to(rerank_path).parts}
    import torch
    torch.set_num_threads(4)
    base.wait_for_gpu(torch,'a-default',0)
    implementation=base.code_hashes()
    implementation[str(Path(__file__).relative_to(ROOT))]=base.text_file_hash(Path(__file__))
    experiment_id=base.next_experiment_id(owner,ROOT/'.experiment-state/a')
    paths={k:ROOT/p/'a-evidence-pipeline'/experiment_id for k,p in
           [('processed','data/processed'),('results','results')]}
    for path in paths.values():path.mkdir(parents=True,exist_ok=False)
    for name in ('documents.json','eval.json'):
        (paths['processed']/name).write_bytes((sources[0]['paths']['processed']/name).read_bytes())
    documents=base.read_json(paths['processed']/'documents.json')
    cases=base.read_json(paths['processed']/'eval.json')
    base.validate_inputs(documents,cases)
    experiment={k:sources[0]['result']['experiment'][k] for k in
                ('documents_sha256','evaluation_sha256','model_manifest_sha256')}
    for filename,key in [('documents.json','documents_sha256'),('eval.json','evaluation_sha256')]:
        if base.text_file_hash(paths['processed']/filename)!=experiment[key]:raise ValueError('Snapshot mismatch')
    experiment.update(experiment_id=experiment_id,owner=owner,**base.code_version(ROOT))
    trace=base.Trace.from_env(experiment)
    conditions=[]
    print(f'{experiment_id}: validated three existing indexes; {len(cases)} cases',flush=True)
    with trace.run('a-evidence-pipeline'):
        client=ScenarioAClient({'A_EMBEDDING_MODEL':base.MODEL_NAME,'A_GENERATION_MODEL':'unused'},
                               base.MODEL_ROOT,base.MODEL_ROOT,need_generation=False)
        # 각 인덱스의 BM25/리랭커 워밍업. 평가 기록과 분리.
        for source in sources:
            start=time.perf_counter()
            retrieval.retrieve(cases[0]['question'],client,source['index'],source['chunks'],source['config'],5,
                               cases[0].get('filters'),**OPTIONS)
            source['warmup_seconds']=time.perf_counter()-start
            conditions.append(dict(name=source['name'],config=source['config'],source_result=source['source_result'],
                index_dir=str(source['paths']['index']),index_hashes=source['index_hashes'],
                records=[],diagnostics=[],query_stats=Counter(),rerank_stats=Counter(),
                warmup_seconds=source['warmup_seconds']))
        for repetition in range(args.repeat):
            # 실행 순서를 회전하여 순차 실행 편향을 줄인다.
            for i in [(repetition+j)%len(sources) for j in range(len(sources))]:
                source,result=sources[i],conditions[i]
                for case in cases:
                    before=dict(client.embedding.stats)
                    with capture_stages() as captured:
                        torch.cuda.synchronize();start=time.perf_counter()
                        hits=retrieval.retrieve(case['question'],client,source['index'],source['chunks'],source['config'],
                                                5,case.get('filters'),**OPTIONS)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    result['records'].append(base.case_record(case,hits,repetition,elapsed,None))
                    result['query_stats'].update({k:v-before[k] for k,v in client.embedding.stats.items()})
                    result['rerank_stats'].update(rerank_token_counts(case['question'],captured.get('candidates',[])))
                    if repetition==0:
                        result['diagnostics'].append(dict(id=case['id'],**diagnose(case,source['chunks'],documents,captured,hits)))
                print(f'{experiment_id}: {source["name"]} repetition {repetition+1}/{args.repeat}',flush=True)
                # 중단되더라도 완료된 기록을 보존. 본문 없이 로컬에만 저장.
                base.write_json(paths['results']/'partial.json',dict(experiment=experiment,conditions=conditions))
        for result in conditions:
            result['summary']=base.summarize(result['records'])
            result['diagnosis_counts']=dict(Counter(d['category'] for d in result['diagnostics']))
            with trace.span('condition-summary',metadata={'condition':result['name']},output=result['summary']):pass
    for source in sources:
        if base.fingerprints(source['paths']['index'])!=source['index_hashes']:raise ValueError('Source changed')
    if any(base.text_file_hash(ROOT/p)!=h for p,h in implementation.items()):raise ValueError('Code changed')
    if base.verify_model(base.MODEL_ROOT/base.MODEL_NAME)!=model:raise ValueError('Model changed')
    if any(base.sha256(rerank_path/p)!=h for p,h in rerank_files.items()):raise ValueError('Reranker changed')
    output=dict(experiment=experiment,measured_at_utc=datetime.now(timezone.utc).isoformat(),
                conditions=conditions,options=retrieval.retrieval_options(**OPTIONS),code_hashes=implementation,
                source_experiments=[s['result']['experiment'] for s in sources],rerank_files=rerank_files,
                model_manifest=model,packages={p:importlib.metadata.version(p) for p in
                ('torch','transformers','sentence-transformers','numpy','faiss-cpu')},
                token_usage=trace.usage,query_stats_including_warmup=client.embedding.stats,trace_id=trace.trace_id,
                paths={k:str(v) for k,v in paths.items()},repeat=args.repeat)
    base.write_json(paths['results']/'evaluation.json',output)
    rows=['# A 기본 검색: 청킹·project blend 근거 검증','',f'실험 ID: {experiment_id}',
          f'실행: {output["measured_at_utc"]}; commit {experiment["commit"]} (미커밋 변경 포함)',
          '', '100문서 / eval_v4 50문항 / fixed 또는 structured-v3 1000/150 / BGE-M3 / 동일 최신 검색 설정.',
          '기존 인덱스 읽기 전용 재사용. 재파싱·재임베딩·생성·판정 API 호출 없음.',
          '', '| 조건 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p95 ms | 답변불가 빈 결과율 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in conditions:
        s=r['summary'];rows.append(f'| {r["name"]} | {s["recall_at_5"]:.6f} | {s["mrr_at_5"]:.6f} | {s["answer_chunk_hit_rate"]:.2f} | {s["mean_seconds"]*1000:.2f} | {s["p95_seconds"]*1000:.2f} | {s["negative_empty_rate"]:.2f} |')
    rows+=['','Recall/MRR 분모 42문항, 근거 지표 20문항, 답변불가 8문항. 근거 지표는 값/키워드 대리지표.',
           '반복은 독립 표본을 늘리지 않음. 실행 순서를 반복별 회전함. 시간에는 단계 캡처의 작은 오버헤드 포함.',
           '', '## 단계별 근거 누락(첫 반복)', '']
    for r in conditions:
        rows += [f'### {r["name"]}', '',f'분류: `{json.dumps(r["diagnosis_counts"])}`', '',
                 '| 문항 ID | 분류 | RRF 후보 순위 | 확장 후보 순위 | 재정렬 순위 |',
                 '| --- | --- | ---: | ---: | ---: |']
        for d in r['diagnostics']:
            if d['applicable'] and d['category']!='hit':
                rows.append(f'| {d["id"]} | {d["category"]} | {d["fused_evidence_rank"]} | {d["candidates_evidence_rank"]} | {d["ranked_evidence_rank"]} |')
        rows += ['',f'질문 토큰/실패 통계: `{json.dumps(r["query_stats"])}`',
                 f'리랭커 토큰 재계산: `{json.dumps(r["rerank_stats"])}`','']
    rows += ['## 고정 옵션·재현 정보','','```json',json.dumps(dict(options=output['options'],experiment=experiment,
             code_hashes=implementation,source_results=[s['source_result'] for s in sources]),indent=2),'```',
             '', '```bash','.venv-a/bin/python experiments/a_evidence_pipeline_chy.py --repeat 3','```',
             '', '리랭커 토큰은 동일 tokenizer 설정(only_second, 최대 1024)으로 측정 구간 밖에서 다시 계산한 padding 제외 입력 토큰이며 API 과금량이 아님.',
             '기존 추적·ID 생성·공개 계약을 유지. 다른 담당 모듈은 수정하지 않음. 독립 평가셋·답변 생성은 미검증.','']
    report=ROOT/'results/reports'/f'a_evidence_pipeline_{experiment_id}.md'
    report.write_text('\n'.join(rows),encoding='utf-8')
    print(f'REPORT={report}',flush=True)


if __name__=='__main__':main()
