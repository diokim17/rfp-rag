"""Fixed 청킹 크기 통제 비교: 로컬 임베딩/검색만 실행."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import a_chunking_chy as base
from dotenv import load_dotenv


def main():
    load_dotenv(ROOT / '.env.a', override=False)
    documents = ROOT / 'data/processed/documents.json'
    evaluation = ROOT / 'data/eval_v4.json'
    implementation = base.code_hashes()
    implementation[str(Path(__file__).relative_to(ROOT))] = base.text_file_hash(Path(__file__))
    expected = [base.text_file_hash(documents), base.text_file_hash(evaluation),
                base.file_hash(base.MODEL_ROOT / base.MODEL_NAME / 'local_manifest.json'),
                hashlib.sha256(json.dumps(base.code_hashes(), sort_keys=True).encode()).hexdigest()]
    results = []
    for size in (500, 1000, 1500):
        with tempfile.TemporaryDirectory(prefix='chy-size-') as folder:
            pointer = Path(folder) / 'result.txt'
            subprocess.run([sys.executable, str(ROOT / 'experiments/a_chunking_chy.py'),
                'run', '--variant', 'fixed', '--profile', 'dense',
                '--chunk-size', str(size), '--chunk-overlap', '150',
                '--documents', str(documents), '--eval-file', str(evaluation),
                '--repeat', '3', '--gpu-wait-seconds', '0',
                '--expected', *expected, '--result-pointer', str(pointer)], check=True, cwd=ROOT)
            results.append(base.read_json(Path(pointer.read_text())))
    first = results[0]['experiment']['experiment_id']
    report = ROOT / 'results/reports' / f'a_chunk_size_comparison_{first}.md'
    lines = ['# A 청킹 크기 비교', '',
        '고정 조건: fixed v1, overlap 150자, BAAI/bge-m3, 문맥 none, dense 검색, top-k 5.',
        '같은 100문서 및 eval_v4 50문항, 3회 반복. 생성/판정 미실행.', '',
        '| ID | 크기 | 청크 수 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p95 ms | build 초 | 임베딩 토큰 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        s, c = r['summary'], r['config']
        lines.append(f"| {r['experiment']['experiment_id']} | {c['chunk_size']} | {c['chunk_count']} | {s['recall_at_5']:.6f} | {s['mrr_at_5']:.6f} | {s['answer_chunk_hit_rate']:.2f} | {s['mean_seconds']*1000:.2f} | {s['p95_seconds']*1000:.2f} | {r['timing']['build_seconds']:.2f} | {r['stats']['build']['input_tokens']} |")
    lines += ['', 'Recall/MRR: 답변 가능 42문항. 근거 적중률: 적용 가능한 20문항의 대리지표로 답변 정확도가 아님.',
        '50문항을 3회 반복한 시간 측정이며 독립 표본 150개가 아님. 공유 GPU 순차 실행으로 시간 차이 해석에 주의.',
        'overlap 절대 길이를 고정하므로 크기별 중복 비율은 다름. 이 평가셋 내 비교이며 기본 설정은 변경하지 않음.',
        '', '## 재현 기록', '', '```json', json.dumps(dict(input_hashes=expected[:3], code_hashes=implementation,
        conditions=[str(Path(r['paths']['results'])/'evaluation.json') for r in results]), indent=2), '```', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    base.write_json(Path(results[0]['paths']['results'])/'size_comparison.json', dict(
        conditions=[str(Path(r['paths']['results'])/'evaluation.json') for r in results], code_hashes=implementation))
    print(f'COMPARISON_REPORT={report}', flush=True)


if __name__ == '__main__':
    main()
