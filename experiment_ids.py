"""담당자 이니셜별 실행 번호를 로컬 SQLite에 영구 저장합니다."""

from pathlib import Path
from contextlib import closing
import re
import sqlite3

TEAM_INITIALS = {
    "김도영": "dyk", "나상훈": "shn", "유찬혁": "chy",
    "김연주": "yjk", "박단비": "dbp", "김시현": "shk",
}


def owner_initials(value):
    value = value.strip()
    if value in TEAM_INITIALS:
        return TEAM_INITIALS[value]
    if not re.fullmatch(r"[A-Za-z]{2,16}", value):
        raise ValueError("EXPERIMENT_OWNER 또는 --owner에 고유한 영문 이니셜(2~16자)을 설정하세요. 예: dyk, yjk")
    return value.lower()


def next_experiment_id(owner, state_dir):
    """동시 실행에도 같은 owner에 서로 다른 번호 배정. 실패한 실행 번호도 유지."""
    owner = owner_initials(owner)
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(directory / "experiments.sqlite3", timeout=30)) as connection:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS counters (owner TEXT PRIMARY KEY, number INTEGER NOT NULL)")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT number FROM counters WHERE owner = ?", (owner,)).fetchone()
            number = row[0] + 1 if row else 1
            connection.execute("INSERT INTO counters VALUES (?, ?) ON CONFLICT(owner) DO UPDATE SET number = excluded.number",
                               (owner, number))
    return f"{owner}-{number:04d}"
