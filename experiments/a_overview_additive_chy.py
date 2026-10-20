"""fixed 청크 유지 + 개요 추가: fixed/stable 비교, 기존 벡터 재사용."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
import numpy as np
import a_embedding as embedding
import a_retrieval as retrieval
from a_client import ScenarioAClient
from a_evaluation import _answer_chunk_hit
from experiments import a_chunking_chy as base
from experiments.a_evidence_pipeline_chy import OPTIONS, read_source, capture_stages, diagnose, rerank_token_counts


def evidence_rank(ids, matching):
    return next((i for i,cid in enumerate(ids,1) if cid in matching),None)


def strict_evidence(case,hits):
    spans=case.get('evidence_spans')
    if not spans:return None
    return float(all(any(h['doc_id']==s['doc_id'] and h['metadata']['start_char']<=s['start_char']
                        and s['end_char']<=h['metadata']['end_char'] for h in hits) for s in spans))


@contextmanager
def capture(chunks):
    original=retrieval._rrf
    with capture_stages() as stages:
        def fused(rankings,constant):
            stages['vector_ids']=[chunks[i]['chunk_id'] for i in rankings[0]]
            stages['bm25_ids']=[chunks[i]['chunk_id'] for i in rankings[1]]
            result=original(rankings,constant)
            stages['rrf_ids']=[chunks[i]['chunk_id'] for i,_ in result]
            return result
        with patch.object(retrieval,'_rrf',fused):yield stages


def chunk_key(chunk):
    return chunk['doc_id'], chunk['metadata']['start_char'], chunk['metadata']['end_char']


def validate_additive(docs, fixed, chunks):
    # additive 개요는 fixed 안에 포함될 수 있으므로 단조 end/overlap 검증은 fixed에 적용.
    geometry = base.geometry(docs, fixed)
    old = {chunk_key(c): c for c in fixed}
    actual = {chunk_key(c): c for c in chunks}
    if len(actual) != len(chunks) or len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Duplicate chunks')
    for key, c in old.items():
        other = actual[key]
        if c['text'] != other['text'] or c['metadata'] != other['metadata']:
            raise ValueError('Fixed chunk changed')
    allowed = {(d['doc_id'], a, b) for d in docs for a, b in embedding._overview_spans(d,1000,embedding._table_spans(d['text']))}
    if set(actual) - set(old) != allowed - set(old):
        raise ValueError('Unexpected added spans')
    bydoc = {d['doc_id']: d for d in docs}
    for (docid,a,b), c in actual.items():
        if not 0 <= a < b <= len(bydoc[docid]['text']) or b-a > 1000 or c['text'] != bydoc[docid]['text'][a:b]:
            raise ValueError('Invalid source span')
    return dict(fixed_geometry=geometry, chunk_count=len(chunks), fixed_preserved=len(old), added=len(actual)-len(old))


def build_reusing_fixed(docs, client, path, fixed, documents_sha256):
    cfg = fixed['config']
    if (cfg['chunking_strategy'],cfg['embedding_context'],cfg['embedding_model'],cfg['chunk_size'],cfg['chunk_overlap'],cfg['documents_sha256']) != ('fixed','none',base.MODEL_NAME,1000,150,documents_sha256):
        raise ValueError('Reuse configuration mismatch')
    chunks = embedding.chunk_documents(docs,1000,150)
    validate_additive(docs,fixed['chunks'],chunks)
    vectors = fixed['index'].reconstruct_n(0,fixed['index'].ntotal)
    cached = {chunk_key(c): v for c,v in zip(fixed['chunks'],vectors)}
    added = [c for c in chunks if chunk_key(c) not in cached]
    for start in range(0,len(added),32):
        batch = added[start:start+32]
        new = embedding._document_vectors(batch,client,base.MODEL_NAME,'none')
        cached.update({chunk_key(c): v for c,v in zip(batch,new)})
    def reuse(batch, *args):
        return np.stack([cached[chunk_key(c)] for c in batch])
    with patch.object(embedding,'_document_vectors',reuse):
        result = embedding.build_index(docs,client,path,base.MODEL_NAME,1000,150,documents_sha256=documents_sha256)
    index, saved, _ = embedding.load_index(path)
    actual = index.reconstruct_n(0,index.ntotal)
    for i,c in enumerate(saved):
        if not np.array_equal(actual[i],cached[chunk_key(c)]):
            raise ValueError('Stored reused vector differs')
    return result, dict(reused=len(fixed['chunks']),newly_embedded=len(added),vectors_exact=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-cases',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--gpu-wait-seconds',type=int,default=600)
    args=parser.parse_args()
    if args.repeat<1:parser.error('repeat must be positive')
    if args.gpu_wait_seconds<0:parser.error('gpu wait must be nonnegative')
    load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT/'.env.a',override=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
                      RFP_EMBEDDING_CONTEXT='none',RFP_CHUNKING_STRATEGY='overview_additive')
    owner=base.owner_initials(os.getenv('EXPERIMENT_OWNER',''))
    if os.getenv('A_RERANK_MODEL')!='bge-reranker-v2-m3':raise ValueError('Pinned local reranker required')
    sources=[read_source('fixed-none','chy-0011')]
    prior_path=ROOT/'results/a-overview-stable/chy-0019/evaluation.json'
    prior=base.read_json(prior_path)
    old=next(c for c in prior['conditions'] if c['name']=='overview-stable-none')
    old_path=Path(old['index_dir'])
    if base.fingerprints(old_path)!=old['index_hashes']:raise ValueError('Overview source hash mismatch')
    for key in ('documents_sha256','evaluation_sha256','new_cases_sha256'):
        expected=base.text_file_hash(args.new_cases) if key=='new_cases_sha256' else sources[0]['result']['experiment'][key]
        if prior['experiment'][key]!=expected:raise ValueError('Overview input mismatch')
    if prior['options']!=retrieval.retrieval_options(**OPTIONS):raise ValueError('Retrieval options changed')
    old_index,old_chunks,old_cfg=embedding.load_index(old_path)
    if old_cfg!=old['config']:raise ValueError('Overview config mismatch')
    if prior['model_manifest']!=base.verify_model(base.MODEL_ROOT/base.MODEL_NAME):raise ValueError('Overview model mismatch')
    sources.append(dict(name='overview-stable-none',index=old_index,chunks=old_chunks,config=old_cfg,paths={'index':old_path},index_hashes=old['index_hashes'],result=sources[0]['result'],source_result=str(prior_path)))
    for key in ('documents_sha256','evaluation_sha256','model_manifest_sha256'):
        if len({s['result']['experiment'][key] for s in sources})!=1:raise ValueError('Provenance differs')
    model=base.verify_model(base.MODEL_ROOT/base.MODEL_NAME)
    if base.file_hash(base.MODEL_ROOT/base.MODEL_NAME/'local_manifest.json')!=sources[0]['result']['experiment']['model_manifest_sha256']:
        raise ValueError('Query model differs')
    rerank_path=Path(retrieval.rerank_model_source())
    rerank_files={str(p.relative_to(rerank_path)):base.sha256(p) for p in sorted(rerank_path.rglob('*')) if p.is_file() and '.cache' not in p.parts}
    if prior['rerank_files']!=rerank_files:raise ValueError('Overview reranker mismatch')
    new_cases=base.read_json(args.new_cases)
    original_cases=base.read_json(sources[0]['paths']['processed']/'eval.json')
    original_docs=base.read_json(sources[0]['paths']['processed']/'documents.json')
    base.validate_inputs(original_docs,original_cases+new_cases)
    old_gold={d for c in original_cases for d in c.get('expected_doc_ids',[])}
    if any(old_gold & set(c['expected_doc_ids']) for c in new_cases):raise ValueError('New case gold overlaps existing evaluation')
    by_doc={d['doc_id']:d for d in original_docs}
    for c in new_cases:
        if not c.get('evidence_spans'):raise ValueError('New case requires manually reviewed spans')
        for s in c['evidence_spans']:
            if s['doc_id'] not in c['expected_doc_ids'] or not 0<=s['start_char']<s['end_char']<=len(by_doc[s['doc_id']]['text']):
                raise ValueError('Invalid evidence span')
        if _answer_chunk_hit(c,[by_doc[d] for d in c['expected_doc_ids']])!=1:raise ValueError('New case label not found')
    import torch
    torch.set_num_threads(4);base.wait_for_gpu(torch,'a-default',args.gpu_wait_seconds)
    eid=base.next_experiment_id(owner,ROOT/'.experiment-state/a')
    paths={k:ROOT/p/'a-overview-additive'/eid for k,p in [('processed','data/processed'),('index','indexes'),('results','results')]}
    for p in paths.values():p.mkdir(parents=True,exist_ok=False)
    for name in ('documents.json','eval.json'):
        (paths['processed']/name).write_bytes((sources[0]['paths']['processed']/name).read_bytes())
    (paths['processed']/'new_cases.json').write_bytes(args.new_cases.read_bytes())
    docs=base.read_json(paths['processed']/'documents.json');cases=base.read_json(paths['processed']/'eval.json')
    new_cases=base.read_json(paths['processed']/'new_cases.json');all_cases=cases+new_cases
    for name,key in [('documents.json','documents_sha256'),('eval.json','evaluation_sha256')]:
        if base.text_file_hash(paths['processed']/name)!=sources[0]['result']['experiment'][key]:raise ValueError('Snapshot mismatch')
    if base.text_file_hash(paths['processed']/'new_cases.json')!=base.text_file_hash(args.new_cases):raise ValueError('New cases changed')
    implementation=base.code_hashes()
    for name in ('experiments/a_overview_additive_chy.py','experiments/a_evidence_pipeline_chy.py'):
        implementation[name]=base.text_file_hash(ROOT/name)
    experiment=dict(experiment_id=eid,owner=owner,**base.code_version(ROOT),
        documents_sha256=base.text_file_hash(paths['processed']/'documents.json'),
        evaluation_sha256=base.text_file_hash(paths['processed']/'eval.json'),
        new_cases_sha256=base.text_file_hash(paths['processed']/'new_cases.json'))
    trace=base.Trace.from_env(experiment)
    print(f'{eid}: frozen rules; {len(docs)} documents; {len(cases)} existing + {len(new_cases)} targeted new cases',flush=True)
    results=[]
    with trace.run('a-overview-chunking'):
        client=ScenarioAClient({'A_EMBEDDING_MODEL':base.MODEL_NAME,'A_GENERATION_MODEL':'unused'},base.MODEL_ROOT,base.MODEL_ROOT,need_generation=False)
        before=dict(client.embedding.stats);start=time.perf_counter()
        cfg,reuse=build_reusing_fixed(docs,client,paths['index'],sources[0],experiment['documents_sha256'])
        build_seconds=time.perf_counter()-start;build_stats={k:v-before[k] for k,v in client.embedding.stats.items()}
        index,chunks,cfg=embedding.load_index(paths['index'])
        geometry=validate_additive(docs,sources[0]["chunks"],chunks)
        selected=[dict(doc_id=d['doc_id'],start_char=a,end_char=b) for d in docs for a,b in embedding._overview_spans(d,1000,embedding._table_spans(d['text']))]
        base.write_json(paths['processed']/'selected_spans.json',selected)
        sources.append(dict(name='overview-additive-none',index=index,chunks=chunks,config=cfg,paths={'index':paths['index']},index_hashes=base.fingerprints(paths['index'])))
        print(f'{eid}: candidate built; {len(chunks)} chunks; {build_seconds:.2f}s',flush=True)
        for source in sources:
            retrieval.retrieve(cases[0]['question'],client,source['index'],source['chunks'],source['config'],5,cases[0].get('filters'),**OPTIONS)
            results.append(dict(name=source['name'],config=source['config'],index_dir=str(source['paths']['index']),index_hashes=source['index_hashes'],records=[],diagnostics=[],query_stats=Counter(),rerank_stats=Counter()))
        for repetition in range(args.repeat):
            for i in [(repetition+j)%len(sources) for j in range(len(sources))]:
                source,result=sources[i],results[i]
                for case in all_cases:
                    before=dict(client.embedding.stats)
                    with capture(source['chunks']) as stages:
                        torch.cuda.synchronize();start=time.perf_counter()
                        hits=retrieval.retrieve(case['question'],client,source['index'],source['chunks'],source['config'],5,case.get('filters'),**OPTIONS)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    row=base.case_record(case,hits,repetition,elapsed,None)
                    row.update(split='existing' if case in cases else 'new',strict_evidence=strict_evidence(case,hits))
                    result['records'].append(row)
                    result['query_stats'].update({k:v-before[k] for k,v in client.embedding.stats.items()})
                    result['rerank_stats'].update(rerank_token_counts(case['question'],stages.get('candidates',[])))
                    if repetition==0:
                        d=diagnose(case,source['chunks'],docs,stages,hits)
                        for stage in ('vector','bm25','rrf'):
                            d[stage+'_ids']=stages.get(stage+'_ids',[])
                            d[stage+'_evidence_rank']=evidence_rank(d[stage+'_ids'],set(d['matching_chunk_ids']))
                        matching=[c for c in source['chunks'] if c['chunk_id'] in set(d['matching_chunk_ids'])]
                        d['matching_geometry']=[{'chunk_id':c['chunk_id'],'start_char':c['metadata']['start_char'],'end_char':c['metadata']['end_char'],'length':len(c['text'])} for c in matching]
                        d.update(id=case['id'],split=row['split'],strict_evidence=row['strict_evidence'])
                        result['diagnostics'].append(d)
                print(f'{eid}: {source["name"]} repetition {repetition+1}/{args.repeat}',flush=True)
                base.write_json(paths['results']/'partial.json',dict(experiment=experiment,conditions=results))
        for result in results:
            result['summary']={split:base.summarize([r for r in result['records'] if r['split']==split]) for split in ('existing','new')}
            result['strict_new_evidence_rate']=base.mean([r['strict_evidence'] for r in result['records'] if r['split']=='new'])
            with trace.span('condition-summary',metadata={'condition':result['name']},output=result['summary']):pass
    for source in sources:
        if base.fingerprints(source['paths']['index'])!=source['index_hashes']:raise ValueError('Index changed')
    if any(base.text_file_hash(ROOT/p)!=h for p,h in implementation.items()):raise ValueError('Code changed')
    if base.verify_model(base.MODEL_ROOT/base.MODEL_NAME)!=model:raise ValueError('Model changed')
    if any(base.sha256(rerank_path/p)!=h for p,h in rerank_files.items()):raise ValueError('Reranker changed')
    output=dict(experiment=experiment,measured_at_utc=datetime.now(timezone.utc).isoformat(),repeat=args.repeat,
        conditions=results,options=retrieval.retrieval_options(**OPTIONS),code_hashes=implementation,
        model_manifest=model,rerank_files=rerank_files,source_results=[s.get('source_result') for s in sources[:2]],
        geometry=geometry,selected_spans_count=len(selected),selected_document_count=len({s['doc_id'] for s in selected}),
        build_seconds=build_seconds,build_stats=build_stats,vector_reuse=reuse,token_usage=trace.usage,trace_id=trace.trace_id,
        paths={k:str(v) for k,v in paths.items()})
    base.write_json(paths['results']/'evaluation.json',output)
    rows=['# fixed 청크 유지 및 개요 추가 실험','',f'실험 ID: {eid}; {output["measured_at_utc"]}',
      '', '변경: overview-additive v1. fixed/overview-stable 대비 임베딩 문맥 none, 최신 A 검색 옵션 동일.',
      '규칙: 다음 절까지 1000자 이내, 사업명/기간/예산/계약 방식 중 2종류 이상. 개요 인식 규칙은 v1과 동일. 모든 fixed 청크를 유지하고 개요 청크만 추가. 동일 좌표만 중복 제거. 원본 fixed 벡터 재사용.',
      '질문·정답·문서 ID를 청킹 규칙에 사용하지 않음. 기존 기본값·공개 함수·원문 좌표·3파일 인덱스 형식 유지.',
      '', '| 평가 | 조건 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p95 ms |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for split in ('existing','new'):
        for r in results:
            s=r['summary'][split];rows.append(f'| {split} | {r["name"]} | {s["recall_at_5"]:.6f} | {s["mrr_at_5"]:.6f} | {s["answer_chunk_hit_rate"]:.3f} | {s["mean_seconds"]*1000:.2f} | {s["p95_seconds"]*1000:.2f} |')
    rows+=['','기존 eval_v4 50문항(답변 가능 42, 근거 대리지표 20)과 새 질문 6문항은 별도 집계.',
      '새 6문항은 기존 평가 정답 문서와 겹치지 않는 6문서의 개요 기간/예산 질문. 규칙 고정 후 작성했으나 개요가 인식되는 문서만 선택한 목표 검증이므로 대표성 있는 독립 평가셋이 아님.',
      '새 질문은 원문 근거 좌표로 전체 항목이 최종 청크에 포함되는지도 검사. 답변 생성·판정 미실행.',
      '',f'선택 문서 {output["selected_document_count"]}, 구간 {len(selected)}, 후보 청크 {len(chunks)}.',
      f'실제 후보 build {build_seconds:.2f}초, 로컬 임베딩 통계 `{json.dumps(build_stats)}`.',
      '', '## 재현','', '```bash',f'.venv-a/bin/python experiments/a_overview_additive_chy.py --new-cases {paths["processed"].relative_to(ROOT)}/new_cases.json --repeat 1','```',
      '', '```json',json.dumps(dict(experiment=experiment,options=output['options'],code_hashes=implementation),indent=2),'```','']
    report=ROOT/'results/reports'/f'a_overview_additive_{eid}.md';report.write_text('\n'.join(rows),encoding='utf-8')
    print(f'REPORT={report}',flush=True)


if __name__=='__main__':main()
