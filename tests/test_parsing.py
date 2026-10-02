"""원본 파일 없이 HWP 표 복원 로직을 검증합니다."""

import struct
import unittest

from parsing import _cell_text, _render_table, _section_text


def record(tag, level, payload):
    return struct.pack("<I", tag | (level << 10) | (len(payload) << 20)) + payload


def para(level, text):
    return record(67, level, text.encode("utf-16-le"))


def cell(col, row, colspan=1, rowspan=1):
    return record(72, 1, bytes(8) + struct.pack("<4H", col, row, colspan, rowspan) + bytes(8))


class RenderTableTest(unittest.TestCase):
    def test_rowspan_repeated_with_header_line(self):
        cells = [(0, 0, 2, 1, "구분"), (0, 1, 1, 1, "a"), (1, 1, 1, 1, "b")]
        self.assertEqual(_render_table(2, 2, cells), "| 구분 | a |\n|---|---|\n| 구분 | b |")

    def test_single_row_table_becomes_paragraph(self):
        self.assertEqual(_render_table(1, 4, [(0, 0, 1, 1, "7"), (0, 2, 1, 1, "기타사항")]), "7 기타사항")


class CellTextTest(unittest.TestCase):
    def test_letter_spaced_word_joined_but_normal_phrase_kept(self):
        self.assertEqual(_cell_text(" 사 업 명 "), "사업명")
        self.assertEqual(_cell_text("사업 기간 a|b"), "사업 기간 a/b")


class SectionTextTest(unittest.TestCase):
    def test_table_records_become_markdown(self):
        data = (para(0, "본문")
                + record(71, 0, b" lbt") + record(77, 1, struct.pack("<IHH", 0, 2, 2))
                + cell(0, 0) + para(2, "항목") + cell(1, 0) + para(2, "금액")
                + cell(0, 1) + para(2, "예산") + cell(1, 1) + para(2, "100원")
                + para(0, "다음 문단"))
        self.assertEqual(_section_text(data),
                         ["본문", "| 항목 | 금액 |\n|---|---|\n| 예산 | 100원 |", "다음 문단"])


if __name__ == "__main__":
    unittest.main()
