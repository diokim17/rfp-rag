"""원본 파일 없이 HWP 표 복원 로직을 검증합니다."""

import struct
import unittest

from parsing import (_cell_text, _clean_text, _drop_toc, _fields, _merge_pages, _number_tables, _reading_order,
                     _render_table, _section_text, _sections, _won, section_of)


def record(tag, level, payload):
    return struct.pack("<I", tag | (level << 10) | (len(payload) << 20)) + payload


def para(level, text):
    return record(67, level, text.encode("utf-16-le"))


def cell(col, row, colspan=1, rowspan=1):
    return record(72, 1, bytes(8) + struct.pack("<4H", col, row, colspan, rowspan) + bytes(8))


class RenderTableTest(unittest.TestCase):
    def test_rowspan_repeated_with_header_line(self):
        cells = [(0, 0, 2, 1, "구분"), (0, 1, 1, 1, "a"), (1, 1, 1, 1, "b")]
        self.assertEqual(_render_table(2, 2, cells), "<!-- table -->\n| 구분 | a |\n|---|---|\n| 구분 | b |\n<!-- /table -->")

    def test_colspan_header_repeated_but_long_text_not(self):
        cells = [(0, 0, 2, 1, "구분"), (0, 1, 1, 2, "예산"), (1, 1, 1, 1, "국비"), (1, 2, 1, 1, "지방비"),
                 (2, 0, 1, 1, "비고"), (2, 1, 1, 2, "가" * 31)]
        self.assertEqual(_render_table(3, 3, cells),
                         "<!-- table -->\n| 구분 | 예산 | 예산 |\n|---|---|---|\n| 구분 | 국비 | 지방비 |\n"
                         f"| 비고 | {'가' * 31} |  |\n<!-- /table -->")

    def test_single_row_table_becomes_paragraph(self):
        self.assertEqual(_render_table(1, 4, [(0, 0, 1, 1, "7"), (0, 2, 1, 1, "기타사항")]), "7 기타사항")

    def test_rows_identical_after_rowspan_repeat_dropped(self):
        cells = [(0, 0, 1, 1, "목차"), (0, 1, 2, 1, "목 차"), (1, 0, 1, 1, "목차"), (2, 0, 1, 1, "Ⅰ"), (2, 1, 1, 1, "개요")]
        self.assertEqual(_render_table(3, 2, cells),
                         "<!-- table -->\n| 목차 | 목 차 |\n|---|---|\n| Ⅰ | 개요 |\n<!-- /table -->")


class CellTextTest(unittest.TestCase):
    def test_letter_spaced_word_joined_but_normal_phrase_kept(self):
        self.assertEqual(_cell_text(" 사 업 명 "), "사업명")
        self.assertEqual(_cell_text("사업 기간 a|b"), "사업 기간 a/b")


class NumberTablesTest(unittest.TestCase):
    def test_adjacent_tables_get_separate_ids(self):
        t = "<!-- table -->\n| a | b |\n<!-- /table -->"
        self.assertEqual(_number_tables(f"글\n{t}\n{t}"),
                         "글\n<!-- table:T1 -->\n| a | b |\n<!-- /table:T1 -->"
                         "\n<!-- table:T2 -->\n| a | b |\n<!-- /table:T2 -->")


class ReadingOrderTest(unittest.TestCase):
    def test_two_column_page_reads_left_then_right(self):
        items = [(10, 90, y, f"L{y}") for y in (1, 2, 3)] + [(110, 190, y, f"R{y}") for y in (1, 2, 3)]
        self.assertEqual([it[3] for it in sorted(items, key=_reading_order(items, 200))],
                         ["L1", "L2", "L3", "R1", "R2", "R3"])

    def test_single_column_page_reads_top_down(self):
        items = [(10, 190, 2, "b"), (10, 190, 1, "a"), (110, 190, 3, "c")]
        self.assertEqual([it[3] for it in sorted(items, key=_reading_order(items, 200))], ["a", "b", "c"])


class CleanTextTest(unittest.TestCase):
    def test_form_noise_removed_and_checkbox_bullet_restored(self):
        raw = ("합계 | 77 |\n(이 하 여 백)\n다음\n현상태사용[\n]\n□\n√적용) 보고서\nŸ\nGPA 분석"
               "\n\u00ad 주소 | FAX 02\u00ad6312")
        self.assertEqual(_clean_text(raw),
                         "합계 | 77 |\n다음\n현상태사용[ ]\n☑적용) 보고서\n• GPA 분석\n• 주소 | FAX 02-6312")

    def test_spaced_label_before_colon_joined_but_form_run_kept(self):
        self.assertEqual(_clean_text("○ 사 업 비: 금150,000,000원\n○ 기 간 : 180일\n년 월 일 주 소 :"),
                         "○ 사업비: 금150,000,000원\n○ 기간 : 180일\n년 월 일 주 소 :")


class SectionTextTest(unittest.TestCase):
    def test_table_records_become_markdown(self):
        data = (para(0, "본문")
                + record(71, 0, b" lbt") + record(77, 1, struct.pack("<IHH", 0, 2, 2))
                + cell(0, 0) + para(2, "항목") + cell(1, 0) + para(2, "금액")
                + cell(0, 1) + para(2, "예산") + cell(1, 1) + para(2, "100원")
                + para(0, "다음 문단"))
        self.assertEqual(_section_text(data),
                         ["본문", "<!-- table -->\n| 항목 | 금액 |\n|---|---|\n| 예산 | 100원 |\n<!-- /table -->",
                          "다음 문단"])


class MergePagesTest(unittest.TestCase):
    def test_continued_tables_joined_cards_kept(self):
        pages = [
            ["본문", [["구분", "내용"], ["가", "1"]]],
            [[["구분", "내용"], ["나", "2"]]],          # 헤더 반복 → 한 번만
            [[["다", "3"]], [["ID", "SFR-001"]]],     # 헤더 없이 이어짐
            [[["ID", "SFR-002"]]],                    # 일부만 같은 첫 행 → 별개 카드
        ]
        self.assertEqual(_merge_pages(pages), [
            "본문", [["구분", "내용"], ["가", "1"], ["나", "2"], ["다", "3"]],
            [["ID", "SFR-001"]], [["ID", "SFR-002"]]])

class SectionsTest(unittest.TestCase):
    def test_heading_path_skips_toc_lists_and_tables(self):
        text = "\n".join([
            "Ⅰ. 사업개요 4",                 # 목차(쪽번호) → 제외
            "Ⅰ. 사업개요", "1. 사업 개요", "가. 추진 배경", "나. 사업기간: 12개월",  # 콜론 → 목록
            "| 1 사업 | 표 |", "3. 건너뛴 번호",   # 표 줄, 순서 안 맞는 번호 → 제외
            "2. 사업 범위", "Ⅱ 제안요청 내용", "1 상세 요구사항",
            "1. 입찰참가자격", "1.1 참가 자격", "1.1.1 세부 자격",          # 장 제목 없이 번호 재시작 → 상위 비움
            "Ⅳ 기타",                                    # 장 번호 건너뜀(Ⅲ 누락) 허용
        ])
        doc = {"sections": _sections(text)}
        self.assertEqual([p for _, p in doc["sections"]], [
            "Ⅰ. 사업개요", "Ⅰ. 사업개요 > 1. 사업 개요", "Ⅰ. 사업개요 > 1. 사업 개요 > 가. 추진 배경",
            "Ⅰ. 사업개요 > 2. 사업 범위", "Ⅱ 제안요청 내용", "Ⅱ 제안요청 내용 > 1 상세 요구사항",
            "1. 입찰참가자격", "1. 입찰참가자격 > 1.1 참가 자격",
            "1. 입찰참가자격 > 1.1 참가 자격 > 1.1.1 세부 자격", "Ⅳ 기타"])
        self.assertEqual(section_of(doc, 0), "")
        self.assertEqual(section_of(doc, text.index("나. 사업기간")), "Ⅰ. 사업개요 > 1. 사업 개요 > 가. 추진 배경")


class FieldsTest(unittest.TestCase):
    def test_won_units(self):
        self.assertEqual(_won("금 130,000,000원(VAT 포함)"), 130_000_000)
        self.assertEqual(_won("40,000천원"), 40_000_000)
        self.assertEqual(_won("196백만원"), 196_000_000)
        self.assertEqual(_won("1억 5천만 원"), 150_000_000)
        self.assertIsNone(_won("추후 공지"))

    def test_first_valid_value_per_field(self):
        text = "\n".join([
            "사업비 5억 원 미만으로 감리 비대상",               # ':' 없음 → 무시
            "| 계약기간 | 계약금액 |",                        # 값 확인 실패(표 헤더) → 다음 후보
            "ㅇ (사업예산) 1억원 미만",                        # 범위 표현 → 제외
            "3) 사업 예산 : 220,000천원(VAT 포함)",
            "□ 사업기간 : 계약일로부터 3개월",
            "| 입찰방식 | 제한경쟁입찰(협상에 의한 계약) |",
        ])
        self.assertEqual(_fields(text), {
            "원문 사업 예산": "220,000천원(VAT 포함)", "원문 사업 기간": "계약일로부터 3개월",
            "원문 계약 방법": "제한경쟁입찰(협상에 의한 계약)", "원문 사업 금액": "220000000"})


class DropTocTest(unittest.TestCase):
    def test_toc_at_front_dropped_list_at_back_kept(self):
        toc = ("목 차\nⅠ. 사업개요 1\n1. 추진배경 1\n2. 사업범위 3\n[양식 1] 제안서 표지 72\n"
               "Ⅱ. 추진방안 4 1. 추진목표 4 2. 추진체계 5 3. 추진일정 6 4. 협상내용과 범위 7\n")  # 한 문단 목차
        body = "본문 " * 200 + "\n1. 웹서버 2\n2. DB서버 2\n3. 백업서버 1\n"
        self.assertEqual(_drop_toc("제안요청서\n" + toc + body), "제안요청서\n" + body)


if __name__ == "__main__":
    unittest.main()
