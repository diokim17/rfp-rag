"""박단비: 시나리오 A 생성. 서버 GPU에 올린 로컬 모델(EXAONE)로 답변을 만듭니다.

지시문·입력·결과 형식은 시나리오 B(generation.py)와 같아서, A/B를 모델만 다른 조건으로 비교할 수 있습니다.
scenario_a.py가 load_model을 한 번 부르고, 돌려받은 객체의 generate_answer를 질문마다 다시 씁니다.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation import INSTRUCTIONS, NO_HITS_ANSWER, _finish, _prompt, _result

MAX_NEW_TOKENS = 1024  # 답변 최대 길이(토큰)


class LocalGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer, self.model = tokenizer, model

    def generate_answer(self, question, hits, model):
        """B의 generate_answer와 같은 입력·결과 형식."""
        sources = [{"citation": i, **hit} for i, hit in enumerate(hits, 1)]
        if not hits:
            return _result(question, sources, model, NO_HITS_ANSWER, "no_hits")
        messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": _prompt(question, sources)}]
        text, truncated = self._generate(messages)
        return _finish(question, sources, model, text, truncated)

    def _generate(self, messages):
        """모델의 대화 형식으로 바꿔 답변을 생성합니다. 반환: (답변, 길이 한도로 잘렸는지)."""
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                    return_dict=True).to(self.model.device)
        with torch.no_grad():
            # do_sample=False: 매번 같은 답이 나오게 해서 평가 결과를 다시 재현할 수 있게 함
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]  # 입력 부분을 빼고 새로 만든 답변만
        truncated = len(new_tokens) == MAX_NEW_TOKENS and new_tokens[-1].item() != self.tokenizer.eos_token_id
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), truncated


def load_model(model_path):
    """모델 폴더(.env의 A_GENERATION_MODEL)를 GPU에 올립니다."""
    # EXAONE은 모델 폴더에 든 전용 코드로 동작해서 trust_remote_code=True가 필요합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="cuda")
    return LocalGenerator(tokenizer, model)
