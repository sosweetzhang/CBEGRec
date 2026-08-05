from typing import Any, List, Dict


class OpenAILLMClient:

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def complete(self, role: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=60,
        )
        return completion.choices[0].message.content.strip()


def call_llm(client: Any, role: str, messages: List[Dict[str, str]], temperature: float = 0.0, model: str = None) -> str:
    if client is None:
        raise RuntimeError(f"No LLM client configured for {role}")
    if hasattr(client, "complete"):
        try:
            return client.complete(role, messages, temperature)
        except TypeError:
            return client.complete(role=role, messages=messages, temperature=temperature)
    if hasattr(client, "chat") and model:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=60,
        )
        return completion.choices[0].message.content.strip()
    raise TypeError(f"Unsupported LLM client for {role}: {type(client)}")
