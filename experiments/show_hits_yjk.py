"""evaluate 결과 JSON에서 문항별로 검색해 온 상위 k개 청크 원문을 출력합니다 (검색 결과 눈으로 확인용).

API·모델 호출 없음. 원문이 출력되므로 화면이나 파일로만 보고 팀 밖으로 공유하지 마세요.

    python experiments/show_hits_yjk.py results/a/evaluate_yjk-0040_....json
    python experiments/show_hits_yjk.py results/a/evaluate_yjk-0040_....json --case u06 x07 > hits.txt
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("result_file")
parser.add_argument("--case", nargs="*", help="볼 문항 ID (생략: 전체)")
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()

records = json.load(open(args.result_file, encoding="utf-8"))["records"]
for record in records:
    if args.case and record["case_id"] not in args.case:
        continue
    print("=" * 100)
    print(f"[{record['case_id']}] {record['question']}")
    # 답변 점수(1 정답·0.5 부분·0 오답)와 거절 여부, 정답 문서
    print(f"답변 점수: {record['answer_score']}  거절: {record['refused']}  상태: {record['status']}  "
          f"정답 문서: {record['expected_doc_ids']}")
    print(f"답변: {record['answer']}")
    if not record["sources"]:
        print("(검색 결과 없음: 범위 밖 차단 min_score에 걸림)")
    for source in record["sources"][:args.top_k]:
        # ★는 정답 문서에서 나온 청크
        mark = "★" if source["doc_id"] in record["expected_doc_ids"] else " "
        print("-" * 100)
        print(f"{mark} [{source['citation']}] {source['chunk_id']}  {source['metadata'].get('사업명')}")
        print(f"  코사인 {source['score']:.3f}  재정렬 {source.get('rerank_score')}")
        print(source["text"])
