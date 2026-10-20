# A 청킹 크기 비교

고정 조건: fixed v1, overlap 150자, BAAI/bge-m3, 문맥 none, dense 검색, top-k 5.
같은 100문서 및 eval_v4 50문항, 3회 반복. 생성/판정 미실행.

| ID | 크기 | 청크 수 | Recall@5 | MRR@5 | 근거 적중률 | 평균 ms | p95 ms | build 초 | 임베딩 토큰 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chy-0008 | 500 | 24740 | 0.932540 | 0.908730 | 0.70 | 51.32 | 76.28 | 168.23 | 6991055 |
| chy-0009 | 1000 | 10218 | 0.928571 | 0.888889 | 0.65 | 34.76 | 45.33 | 139.47 | 5735042 |
| chy-0010 | 1500 | 6452 | 0.908730 | 0.910714 | 0.75 | 29.64 | 36.11 | 140.40 | 5408849 |

Recall/MRR: 답변 가능 42문항. 근거 적중률: 적용 가능한 20문항의 대리지표로 답변 정확도가 아님.
50문항을 3회 반복한 시간 측정이며 독립 표본 150개가 아님. 공유 GPU 순차 실행으로 시간 차이 해석에 주의.
overlap 절대 길이를 고정하므로 크기별 중복 비율은 다름. 이 평가셋 내 비교이며 기본 설정은 변경하지 않음.

## 재현 기록

```json
{
  "input_hashes": [
    "97bb7a5afa3219566047e817fa6125617d7fd18fe3df52285785f070eed9807a",
    "ded1e0b3be82f4567f2a5f9ad91c8e2329f56cc79cc3715bf2f866a4f2fdbf4a",
    "c08690d7741b680b8a382f643588ba1909e6ae6e7cffee7c030afe15f4879611"
  ],
  "code_hashes": {
    "a_client.py": "cfbe717d833350b49af63bc5b4a3de0f12e86b41f160f565279522657142bc23",
    "a_embedding.py": "fc7834766ce400262e160373c9ae9177bca387be9abdd5a0dc8ef0ad50535c76",
    "a_evaluation.py": "32045ef786d0dd49f49d5c78058e48b65a66da06cf11c6eaafa216c40688ff85",
    "a_experiment_ids.py": "291a21e4991babe98389c66226a29338b876c886478f1666c5a5fec0a0c3934a",
    "a_experiment_reports.py": "47ad3b7440e8e0dbfabb64ac02a459b668078d7d3d67ee58c8968e583667f570",
    "a_generation.py": "cfc093478d87e9f7c6d93bd739fc946dc5aa8dce1ef309e0ab7080c5021a3d27",
    "a_local_embedding.py": "db55b9486f173ebf6c98d876a32cdbd3cfcb3b932e87b3ac66af2b6e710bcac3",
    "a_local_generation.py": "e8dfd22d7ba236c0f3a241039af1e1e7b1472fc157682b76695ee99b73876b20",
    "a_observability.py": "7b2408f92face6a98ba797fd10b4f1e80d1bcaa4b86e8815404403f6f6d35c24",
    "a_parsing.py": "746eedb11d5d15d8b11f93e780cb5c9f108eeb65a6bbd024143f97952cd13f8a",
    "a_query_rewrite.py": "8476037b592dda2ce23bd42e68f3287c0b0c8a1a1724d19cc87da736e705f697",
    "a_retrieval.py": "402e23c7325360144116d8ad161d8511dea8c6a274cccad15aa7718afdcbdd28",
    "a_run.py": "1c08c2e21dcd4d7d5084abe1071bdbdec5388d04173d51cebc38666e797f9a20",
    "experiments/a_chunking_chy.py": "554a0976d9c17a6f34f7efd64b1b775c7cedddc128c8b19616ba53ee987c813d",
    "experiments/a_embedding_chy.py": "10e372551c1248f4f725788ba6a05c5f4ed7e444b402505fede3b5079ab8c3cd",
    "experiments/a_retrieval_metrics.py": "8c83849a097b08ed9fd505370717fbb1bac92139c369c65c65416a7a47c85354",
    "experiments/a_chunk_size_chy.py": "6e4ac890fe8e95626cb2655e60018a9bb43aaa6b05949989bdf393b2987b923e"
  },
  "conditions": [
    "/home/spai1304/rfp-rag/results/a-chunking/chy-0008/evaluation.json",
    "/home/spai1304/rfp-rag/results/a-chunking/chy-0009/evaluation.json",
    "/home/spai1304/rfp-rag/results/a-chunking/chy-0010/evaluation.json"
  ]
}
```

## 검증 및 해석

- 최신 origin/dev `49e3e4e` 반영 후 세 조건을 새로 실행. 전체 테스트 520개 및 pip check 통과.
- 저장된 450개 검색 기록의 Recall/MRR/근거 적중률 재계산 일치. 입력·모델·코드·검색 설정 동일, 인덱스 해시 검증 통과.
- 모든 조건 검색/임베딩 실패, 잘림, OOM 재시도 0건.
- 500자: 1000자 대비 Recall +0.397%p, MRR +0.019841, 근거 적중률 +5%p. 청크 수 약 2.42배.
- 1500자: 1000자 대비 Recall -1.984%p, MRR +0.021825, 근거 적중률 +10%p. 청크 수 약 36.9% 감소.
- Recall 우선이면 500자가 후속 검증 후보지만 개선 폭이 작고, 더 긴 청크가 근거 키워드를 포함하기 쉬우므로 근거 적중률만으로 우열을 단정하지 않음. 기본 크기 1000자는 유지.
