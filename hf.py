import os
from typing import Any, List, Mapping, Optional

from text_generation import Client
from langchain.llms.base import LLM


LLM_HOST = os.environ.get('LLM_HOST', '0.0.0.0')
LLM_PORT = os.environ.get('LLM_PORT', 6018)
client = Client(f"http://{LLM_HOST}:{LLM_PORT}")


class CustomLLM(LLM):
    name: str
    temperature: float = 0.8
    max_new_tokens: int = 100
    stream: bool = False

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        if not self.stream:
            reply = client.generate(prompt, max_new_tokens=self.max_new_tokens).generated_text
            # print(reply)
            return reply
        else:
            raise NotImplementedError

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"name": self.name}


if __name__ == "__main__":

    query = 'Question: How old is Barack Obama? Answer:'
    llm = CustomLLM(name='local_llm')
    resp = llm(query)
    print(resp)
