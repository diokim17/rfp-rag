# 이전 파이프라인 보관

이 폴더는 A/B 분리 이전 구현과 해당 구현 전용 테스트를 보관합니다.
현재 개발·실행에는 최상위 `a_run.py`, `b_run.py`와 같은 접두사 모듈을 사용하세요.

보관된 설정 회귀 테스트: `python -m unittest discover -s legacy/tests -v`
보관된 CLI: 저장소 루트에서 `python -m legacy.legacy_run check-config`
기존 CLI는 이전 공유 경로를 사용하므로 현재 A/B 산출물 작업에는 사용하지 마세요.
