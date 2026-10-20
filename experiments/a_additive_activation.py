"""Validate and activate a verified additive index in this personal checkout.

No paid judge calls. Existing default paths must be absent; reruns never overwrite them.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from tracking.tracking_ids import next_experiment_id, owner_initials
from tracking.tracking_reports import text_file_hash
from observability.observability_trace import Trace, code_version
from pipelines.server.server_run import read_scenario_settings
from pipelines.server.server_client import ScenarioAClient
import pipelines.server.server_embedding as emb
import pipelines.server.server_retrieval as retrieval
from experiments import a_chunking_chy as base
from experiments.a_evidence_pipeline_chy import OPTIONS


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    for path in (ROOT/'indexes/a', ROOT/'data/processed/a'):
        if path.exists() or path.is_symlink():
            raise ValueError(f'Default path already exists; refusing to overwrite: {path}')
    load_dotenv(ROOT/'.env', override=False)
    settings = read_scenario_settings(ROOT/'.env.a')
    if emb._chunking_strategy() != 'overview_additive' or emb._embedding_context() != 'none':
        raise ValueError('Personal environment overrides the verified default strategy/context')
    if settings['A_EMBEDDING_MODEL'] != 'default_model':
        raise ValueError('Personal embedding model differs from verified model')
    if retrieval.retrieval_options() != retrieval.retrieval_options(**OPTIONS):
        raise ValueError('Personal default retrieval options differ from verified settings')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1')
    prep = ROOT/'results/pr37-full-rebuild-prep'
    source = prep/'runs/20261020T002355Z'
    prior = read(source/'evaluation.json')
    baseline = read(prep/'inputs/baseline.json')
    documents = read(prep/'inputs/documents.json')
    cases, targeted = read(prep/'inputs/eval.json'), read(prep/'inputs/new_cases.json')
    model = base.verify_model(base.MODEL_ROOT/base.MODEL_NAME)
    if model != baseline['model_manifest']:
        raise ValueError('Model manifest changed')
    for name, digest in baseline['rerank_files'].items():
        if base.sha256(Path(retrieval.rerank_model_source())/name) != digest:
            raise ValueError('Reranker changed')
    for condition in prior['conditions']:
        if base.fingerprints(Path(condition['index_dir'])) != condition['index_hashes']:
            raise ValueError('Verified source index changed')
    candidate = read(source/'overview_additive/chunks.json')
    if emb.chunk_documents(documents) != candidate:
        raise ValueError('Default chunk output differs; rebuild and revalidate before activation')
    for name,key in [('documents.json','documents_sha256'),('eval.json','evaluation_sha256'),('new_cases.json','new_cases_sha256')]:
        if text_file_hash(prep/'inputs'/name) != baseline['experiment'][key]:
            raise ValueError('Input snapshot changed')
    owner = owner_initials(os.getenv('EXPERIMENT_OWNER',''))
    eid = next_experiment_id(owner, ROOT/'.experiment-state/a')
    result_dir = ROOT/'results/a-additive-activation'/eid
    processed = ROOT/'data/processed/a-additive-deploy'/eid
    index_dir = ROOT/'indexes/a-additive-deploy'/eid
    for path in (result_dir, processed, index_dir):
        path.mkdir(parents=True, exist_ok=False)
    for name in ('documents.json','eval.json','new_cases.json'):
        shutil.copyfile(prep/'inputs'/name, processed/name)
    for name in ('index.faiss','chunks.json','config.json'):
        shutil.copyfile(source/'overview_additive'/name, index_dir/name)
    cfg = read(index_dir/'config.json')
    if cfg['documents_sha256'] != text_file_hash(processed/'documents.json'):
        raise ValueError('Deployment documents mismatch')
    if base.fingerprints(index_dir) != base.fingerprints(source/'overview_additive'):
        raise ValueError('Copied index mismatch')
    experiment = dict(experiment_id=eid, owner=owner, **code_version(ROOT),
                      documents_sha256=text_file_hash(processed/'documents.json'),
                      evaluation_sha256=text_file_hash(processed/'eval.json'))
    trace = Trace.from_env(experiment)
    output = dict(experiment=experiment, options=retrieval.retrieval_options(),
                  reused_from=str(source), code_hashes=base.code_hashes(),
                  index_dir=str(index_dir), processed_dir=str(processed),
                  rollback_index=str(source/'fixed'), conditions=[])
    save(result_dir/'evaluation.json', output)
    print(eid + ': inputs, code output, models, deployment copies verified', flush=True)
    import torch
    torch.set_num_threads(4)
    with trace.run('additive-activation-validation'):
        client = ScenarioAClient(settings, base.MODEL_ROOT, base.MODEL_ROOT, need_generation=False)
        for name,path in [('fixed',source/'fixed'),('overview_additive',index_dir)]:
            index,chunks,config = emb.load_index(path)
            # Count stored document tokens offline; this is not a new embedding inference usage measurement.
            tokens=truncated=0
            for start in range(0,len(chunks),64):
                texts=[c['text'].strip() for c in chunks[start:start+64]]
                if getattr(client.embedding.model[0], 'do_lower_case', False):
                    texts=[t.lower() for t in texts]
                lengths=[len(x) for x in client.embedding.model.tokenizer(texts,padding=False,truncation=False,verbose=False)['input_ids']]
                limit=client.embedding.model.max_seq_length
                tokens+=sum(min(n,limit) for n in lengths)
                truncated+=sum(n>limit for n in lengths)
            retrieval.retrieve(cases[0]['question'],client,index,chunks,config,5,cases[0].get('filters'))
            before=dict(client.embedding.stats)
            rows=[]
            for split,batch in [('existing',cases),('targeted',targeted)]:
                for case in batch:
                    torch.cuda.synchronize(); start=time.perf_counter()
                    hits=retrieval.retrieve(case['question'],client,index,chunks,config,5,case.get('filters'))
                    torch.cuda.synchronize()
                    rows.append(dict(base.case_record(case,hits,0,time.perf_counter()-start),split=split))
            condition=dict(name=name,records=rows,
                           summary={s:base.summarize([r for r in rows if r['split']==s]) for s in ('existing','targeted')},
                           query_stats={k:v-before[k] for k,v in client.embedding.stats.items()},
                           document_token_count_offline=tokens, document_truncated_count_offline=truncated)
            old=next(c for c in prior['conditions'] if c['name']==name)
            expected={r['id']:r for r in old['records']}
            for row in rows:
                if any(row[k]!=expected[row['id']][k] for k in ('recall','mrr','answer_chunk_hit')):
                    raise ValueError('Search metrics differ; do not activate')
            output['conditions'].append(condition)
            save(result_dir/'evaluation.json',output)
            print(name+': all 56 per-case metrics reproduced',flush=True)
    output['usage']=trace.usage
    output['trace_id']=trace.trace_id
    # Both destinations are absent, so no active readers can observe a mixed old/new pair.
    for target,link in ((processed,ROOT/'data/processed/a'),(index_dir,ROOT/'indexes/a')):
        link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)
    output['activated']=True
    save(result_dir/'evaluation.json',output)
    print('Default A paths activated; starting two local generation smoke cases',flush=True)
    for case in [c for c in cases if c['id'] in ('yB058','h10')]:
        with (result_dir/(case['id']+'-ask.log')).open('w') as log:
            subprocess.run([sys.executable,'-m','pipelines.server.server_run','ask','--question',case['question'],
                            '--results-dir',str(result_dir/'smoke'), '--reports-dir',str(result_dir/'reports')],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(case['id']+': default-path ask completed',flush=True)
    print('RESULT_DIR='+str(result_dir),flush=True)

if __name__ == '__main__':
    main()
