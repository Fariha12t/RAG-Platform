"""LLM providers: Anthropic, OpenAI, and an offline extractive fallback (no API key needed)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from app.config import get_settings
from app.services.costs import estimate_tokens
from app.textutils import split_sentences, tokenize

NOT_FOUND = "I could not find this information in the provided documents."


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int):
        import anthropic
        self.client = anthropic.Anthropic()     # reads ANTHROPIC_API_KEY
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, messages: list[dict], **_) -> LLMResult:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        temperature=0, system=system, messages=messages)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResult(text, r.usage.input_tokens, r.usage.output_tokens, self.model)


class OpenAILLM:
    def __init__(self, model: str, max_tokens: int):
        from openai import OpenAI
        self.client = OpenAI()                  # reads OPENAI_API_KEY
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, messages: list[dict], **_) -> LLMResult:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, *messages])
        return LLMResult(r.choices[0].message.content or "", r.usage.prompt_tokens,
                         r.usage.completion_tokens, self.model)


class ExtractiveLLM:
    """Offline fallback: returns the best-matching source sentences with [n] citations.
    Deterministic and free, so the whole stack (and CI) runs without any API key."""
    model = "extractive"

    def __init__(self, min_coverage: float = 0.3, max_sentences: int = 4):
        self.min_coverage, self.max_sentences = min_coverage, max_sentences

    def generate(self, system: str, messages: list[dict], *, sources: Optional[list] = None,
                 question: str = "", **_) -> LLMResult:
        q_tokens = set(tokenize(question))
        scored = []
        for src in sources or []:
            for sent in split_sentences(src["text"]):
                st = set(tokenize(sent))
                if not q_tokens or not st:
                    continue
                cov = len(q_tokens & st) / len(q_tokens)
                scored.append((cov, -src["n"], src["n"], sent))
        scored.sort(reverse=True)
        picked = [x for x in scored if x[0] >= self.min_coverage][: self.max_sentences]
        if not picked:
            text = NOT_FOUND
        else:
            picked.sort(key=lambda x: x[2])          # present in source order
            seen, lines = set(), []
            for _, _, n, sent in picked:
                if sent not in seen:
                    seen.add(sent)
                    lines.append(f"- {sent} [{n}]")
            text = "\n".join(lines)
        prompt = system + " ".join(m["content"] for m in messages)
        return LLMResult(text, estimate_tokens(prompt), estimate_tokens(text), self.model)


@lru_cache(maxsize=1)
def get_llm():
    s = get_settings()
    if s.llm_provider == "anthropic":
        return AnthropicLLM(s.anthropic_model, s.max_output_tokens)
    if s.llm_provider == "openai":
        return OpenAILLM(s.openai_model, s.max_output_tokens)
    return ExtractiveLLM()
