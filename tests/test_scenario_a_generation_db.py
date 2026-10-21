"""박단비: 시나리오 A 생성 모듈을 실제 모델 없이 검증합니다 (_generate만 가짜로 바꿔 끼움)."""

import unittest

import pipelines.server.server_generation as generation
from pipelines.server.server_local_generation import A_REMINDER, A_RULES, LocalGenerator

HITS = [{"chunk_id": "a:0", "doc_id": "a", "text": "예산 100원", "score": .9, "metadata": {"사업명": "a 사업"}}]


class FakeGenerator(LocalGenerator):
    """실제 모델 대신 정해진 답을 돌려주고, 모델에 들어간 입력(messages)을 기록합니다."""

    def __init__(self, text="a 사업 예산은 100원입니다.【1】", truncated=False):
        super().__init__(tokenizer=None, model=None)
        self.text, self.truncated, self.messages = text, truncated, None

    def _generate(self, messages):
        self.messages = messages
        return self.text, self.truncated


class LocalGeneratorTests(unittest.TestCase):
    def answer(self, generator, hits=HITS):
        return generator.generate_answer("예산은?", hits, "generation_model")

    def test_b_prompt_plus_a_rules_and_same_answer_format(self):
        generator = FakeGenerator()
        result = self.answer(generator)
        sources = [{"citation": 1, **HITS[0]}]
        self.assertEqual(result, {"question": "예산은?", "answer": "a 사업 예산은 100원입니다.[1]", "sources": sources,
                                  "model": "generation_model", "status": "ok"})
        # 지시문은 B 지시문 + A 전용 규칙, 입력은 B 입력 뒤에 질문 재확인 문장
        self.assertEqual(generator.messages, [{"role": "system", "content": generation.INSTRUCTIONS + A_RULES},
                                              {"role": "user", "content": generation._prompt("예산은?", sources)
                                               + A_REMINDER.format(question="예산은?")}])

    def test_no_hits_skips_model(self):
        generator = FakeGenerator()
        self.assertEqual(self.answer(generator, hits=[])["status"], "no_hits")
        self.assertIsNone(generator.messages)

    def test_empty_and_truncated(self):
        self.assertEqual(self.answer(FakeGenerator(text=""))["status"], "empty")
        self.assertEqual(self.answer(FakeGenerator(text="100원이며", truncated=True))["status"], "truncated")


if __name__ == "__main__":
    unittest.main()
