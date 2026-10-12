"""박단비: 시나리오 A 생성 모듈을 실제 모델 없이 검증합니다 (_generate만 가짜로 바꿔 끼움)."""

import unittest

import generation
from scenario_a_generation import LocalGenerator

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

    def test_same_prompt_and_answer_format_as_b(self):
        generator = FakeGenerator()
        result = self.answer(generator)
        sources = [{"citation": 1, **HITS[0]}]
        self.assertEqual(result, {"question": "예산은?", "answer": "a 사업 예산은 100원입니다.[1]", "sources": sources,
                                  "model": "generation_model", "status": "ok"})
        self.assertEqual(generator.messages, [{"role": "system", "content": generation.INSTRUCTIONS},
                                              {"role": "user", "content": generation._prompt("예산은?", sources)}])

    def test_no_hits_skips_model(self):
        generator = FakeGenerator()
        self.assertEqual(self.answer(generator, hits=[])["status"], "no_hits")
        self.assertIsNone(generator.messages)

    def test_empty_and_truncated(self):
        self.assertEqual(self.answer(FakeGenerator(text=""))["status"], "empty")
        self.assertEqual(self.answer(FakeGenerator(text="100원이며", truncated=True))["status"], "truncated")


if __name__ == "__main__":
    unittest.main()
