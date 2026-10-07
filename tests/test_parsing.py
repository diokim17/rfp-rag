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

    def test_blank_page_marker_removed(self):
        self.assertEqual(_clean_text("시스템 현황\n- 본 페이지  -\nⅢ 사업추진 계획"), "시스템 현황\nⅢ 사업추진 계획")

    def test_supplementary_pua_removed(self):
        self.assertEqual(_clean_text(chr(0xF02EF) + "업무 " + chr(0xF0832) * 3), "업무")

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

    def test_new_card_not_joined_to_previous_continuation(self):
        pages = [
            [[["", None, ""], ["", "", "앞 카드 이어짐"]]],
            [[["요구사항 분류", None, "기능 요구사항"], ["요구사항 고유번호", None, "SFR-010"]]],
        ]
        self.assertEqual(len(_merge_pages(pages)), 2)

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

    def test_letter_spaced_heading_joined_in_path(self):
        text = "Ⅰ. 제 안 안 내\n1. 사 업 개 요\n가. 총 괄 표 (20점 만점)\n2. 사업 범위"
        self.assertEqual([p for _, p in _sections(text)], [
            "Ⅰ. 제안안내", "Ⅰ. 제안안내 > 1. 사업개요", "Ⅰ. 제안안내 > 1. 사업개요 > 가. 총괄표 (20점 만점)",
            "Ⅰ. 제안안내 > 2. 사업 범위"])

    def test_roman_numeral_on_own_line_joined_with_title(self):
        text = "Ⅰ\n사업개요\n1. 사업 목적\nⅡ\n사업 추진 방안"
        self.assertEqual([p for _, p in _sections(text)], [
            "Ⅰ 사업개요", "Ⅰ 사업개요 > 1. 사업 목적", "Ⅱ 사업 추진 방안"])

    def test_appendix_heading_starts_new_path(self):
        text = "Ⅴ 제안 안내\n1. 비밀 유지\n[붙임4] 소프트웨어 개발사업의 적정 사업기간 종합 산정서\n1. 일반현황\n【별지 제6호 서식】"
        self.assertEqual([p for _, p in _sections(text)], [
            "Ⅴ 제안 안내", "Ⅴ 제안 안내 > 1. 비밀 유지", "[붙임4] 소프트웨어 개발사업의 적정 사업기간 종합 산정서",
            "[붙임4] 소프트웨어 개발사업의 적정 사업기간 종합 산정서 > 1. 일반현황", "【별지 제6호 서식】"])

    def test_appendix_list_and_reference_ignored(self):
        text = ("[붙임 1] 입찰참가신청서\n[붙임 2] 제안회사 일반\n[붙임 3] 용역실적\n1 사업개요\n"
                "(첨부1 적정 사업기간 산정서 참조)\n【서식 제2호】서 약 서")
        self.assertEqual([p for _, p in _sections(text)], ["1 사업개요", "【서식 제2호】서약서"])

    def test_body_numbering_resumes_after_inline_form(self):
        text = ("Ⅴ 각종 기준\n1. 평가 기준\n2. 배점\n[양식 3]\n1. 성명\n3. 전문가파견 가이드라인\n"
                "【붙임 1】조항에 따름\n4. 기타\n[붙임3] 영향평가 검토결과서\n[붙임4] 적정 사업기간 산정서")
        self.assertEqual([p for _, p in _sections(text)], [
            "Ⅴ 각종 기준", "Ⅴ 각종 기준 > 1. 평가 기준", "Ⅴ 각종 기준 > 2. 배점", "[양식 3]", "[양식 3] > 1. 성명",
            "Ⅴ 각종 기준 > 3. 전문가파견 가이드라인", "Ⅴ 각종 기준 > 4. 기타",
            "[붙임3] 영향평가 검토결과서", "[붙임4] 적정 사업기간 산정서"])


class FieldsTest(unittest.TestCase):
    def test_won_units(self):
        self.assertEqual(_won("금 130,000,000원(VAT 포함)"), 130_000_000)
        self.assertEqual(_won("40,000천원"), 40_000_000)
        self.assertEqual(_won("196백만원"), 196_000_000)
        self.assertEqual(_won("1억 5천만 원"), 150_000_000)
        self.assertIsNone(_won("추후 공지"))
        self.assertEqual(_won("일금 육천만원정(￦ 60,000,000 ; 부가세 포함)"), 60_000_000)
        self.assertEqual(_won("50,000,000(금 오천만원/VAT포함)"), 50_000_000)
        self.assertIsNone(_won("2024. 10. 31."))  # 쉼표 없는 숫자는 금액 아님

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

    def test_value_cut_before_next_inline_item(self):
        text = "○ 사업기간 : 계약일로부터 5개월 이내 ○ 용역금액 : 입찰공고문 참조 ○ 계약방법 : 제한경쟁입찰 * 국가계약법 제7조"
        out = _fields(text)
        self.assertEqual(out["원문 사업 기간"], "계약일로부터 5개월 이내")
        self.assertEqual(out["원문 계약 방법"], "제한경쟁입찰")

    def test_value_on_next_bullet_line_and_new_labels(self):
        text = "1.4 사업기간\n- 착수일로부터 ∼ 2024. 10. 31.\nㅇ 낙찰방식 : 협상에 의한 계약"
        out = _fields(text)
        self.assertEqual(out["원문 사업 기간"], "착수일로부터 ∼ 2024. 10. 31.")
        self.assertEqual(out["원문 계약 방법"], "협상에 의한 계약")


class DropTocTest(unittest.TestCase):
    def test_toc_at_front_dropped_list_at_back_kept(self):
        toc = ("목 차\nⅠ. 사업개요 1\n1. 추진배경 1\n2. 사업범위 3\n[양식 1] 제안서 표지 72\n"
               "Ⅱ. 추진방안 4 1. 추진목표 4 2. 추진체계 5 3. 추진일정 6 4. 협상내용과 범위 7\n")  # 한 문단 목차
        body = "본문 " * 200 + "\n1. 웹서버 2\n2. DB서버 2\n3. 백업서버 1\n"
        self.assertEqual(_drop_toc("제안요청서\n" + toc + body), "제안요청서\n" + body)


if __name__ == "__main__":
    unittest.main()
