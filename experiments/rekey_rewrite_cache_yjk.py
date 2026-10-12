"""김연주: 질문 재작성 캐시를 새 키 형식으로 바꿉니다 (API 호출 없음, 한 번만 실행).

  python experiments/rekey_rewrite_cache_yjk.py
  python experiments/rekey_rewrite_cache_yjk.py --cache results/cache/query_rewrites.json --eval-files data/eval_v2.json

커밋 86616c5에서 캐시 키가 sha256("1\\n모델\\n질문")에서 sha256("모델\\n질문")으로 바뀌어
그 전에 만든 캐시는 하나도 맞지 않습니다. 키는 해시라서 질문을 되살릴 수 없으므로,
평가셋 질문으로 옛 키를 다시 계산해 맞는 항목만 새 키로 옮깁니다.
- 원본은 <캐시>.v1-backup.json으로 먼저 복사합니다. 백업이 이미 있으면 덮어쓰지 않습니다(두 번 실행해도 원본 보존).
- 평가셋에 없는 질문의 항목은 지우지 않고 그대로 둡니다. 재작성문 내용은 바꾸지 않습니다.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parsing import read_json, write_json

MODELS = ("gpt-5-mini", "gpt-5-nano")  # OPENAI_GENERATION_MODEL로 허용되는 두 모델


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=ROOT / "results/cache/query_rewrites.json")
    parser.add_argument("--eval-files", type=Path, nargs="+",
                        default=[ROOT / "data/eval_v2.json", ROOT / "data/eval_withindoc_yjk.json"])
    args = parser.parse_args()

    backup = args.cache.with_name(args.cache.stem + ".v1-backup.json")
    if not backup.exists():
        shutil.copyfile(args.cache, backup)
    old = read_json(args.cache)
    questions = {case["question"] for path in args.eval_files for case in read_json(path)}

    new, moved = {}, set()
    for question in questions:
        for model in MODELS:
            old_key = hashlib.sha256(f"1\n{model}\n{question}".encode()).hexdigest()
            if old_key in old:
                new[hashlib.sha256(f"{model}\n{question}".encode()).hexdigest()] = old[old_key]
                moved.add(old_key)
    new.update({key: value for key, value in old.items() if key not in moved})  # 못 찾은 항목은 그대로 둠
    write_json(args.cache, new)
    print(f"전체 {len(old)}건 중 {len(moved)}건을 새 키로 옮김, {len(old) - len(moved)}건은 그대로 둠 (백업: {backup})")


if __name__ == "__main__":
    main()
