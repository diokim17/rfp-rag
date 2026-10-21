"""
지시문·입력·결과 형식은 분리 시점의 시나리오 B와 같아서, A/B를 모델만 다른 조건으로 비교할 수 있습니다.
pipelines/server/server_client.py가 load_model을 한 번 부르고, 돌려받은 객체의 generate_answer를 질문마다 다시 씁니다.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipelines.server.server_generation import INSTRUCTIONS, NO_HITS_ANSWER, _finish, _prompt, _result

MAX_NEW_TOKENS = 1024  # 답변 최대 길이(토큰)

# B 지시문에 덧붙이는 A(EXAONE) 전용 규칙. eval_v4 기준선에서 A만 틀린 유형을 겨냥합니다.
A_RULES = (
    "\n- 없는 정보: 낙찰 금액·낙찰 업체·계약 결과처럼 공고 이후에 정해지는 정보나 검색 자료에 없는 사업은 "
    "RFP에서 확인할 수 없으니 추측하지 말고 확인되지 않는다고 답하세요. 비슷한 다른 항목(예: 사업 예산)을 대신 답하지 마세요."
    "\n- 빠짐없이: 질문이 여러 항목을 물으면 하나도 빠뜨리지 말고 각각 답하고, 자료에 있는 조건·기준·수치는 생략하지 말고 구체적으로 쓰세요."
    "\n- 원문 표기: 금액·날짜·수치는 검색 자료에 적힌 표기 그대로 쓰세요."
)

# 긴 검색 자료 뒤에 질문을 한 번 더 붙입니다. 작은 모델은 자료가 길면 앞쪽 질문을 놓쳐서
# 자료에 없는 정보(낙찰 금액 등)를 비슷한 항목으로 대신 답하는 경우가 있었습니다.
A_REMINDER = ("\n\n다시 질문: {question}\n자료에 있는 내용은 빠짐없이 답하세요. 질문이 직접 묻는 항목이 자료에 없을 때만 "
              "확인되지 않는다고 답하고, 다른 항목으로 대신 답하지 마세요. 질문이 묻지 않은 항목은 언급하지 마세요.")


class LocalGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer, self.model = tokenizer, model

    def generate_answer(self, question, hits, model):
        """B의 generate_answer와 같은 입력·결과 형식."""
        sources = [{"citation": i, **hit} for i, hit in enumerate(hits, 1)]
        if not hits:
            return _result(question, sources, model, NO_HITS_ANSWER, "no_hits")
        messages = [{"role": "system", "content": INSTRUCTIONS + A_RULES},
                    {"role": "user", "content": _prompt(question, sources) + A_REMINDER.format(question=question)}]
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
    """모델 폴더(.env의 SERVER_GENERATION_MODEL)를 GPU에 올립니다."""
    # EXAONE은 모델 폴더에 든 전용 코드로 동작해서 trust_remote_code=True가 필요합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="cuda")
    return LocalGenerator(tokenizer, model)
