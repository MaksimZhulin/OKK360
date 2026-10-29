# -*- coding: utf-8 -*-
"""
Диагностика доступа к моделям облачного шлюза (AgentPlatform).

    venv\\Scripts\\python check_models.py

Спрашивает API-ключ (ввод скрыт, ключ никуда не сохраняется), показывает модели,
которые шлюз отдаёт этому ключу, и пробует крошечный запрос к нашим кандидатам.
Стоимость проверки — доли копейки.
"""
import os
import sys
import getpass

from openai import OpenAI

from config import LLM_BASE_URL
from llm_analysis import api_key_problem

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

CANDIDATES = [
    "google/gemini-2.5-flash", "gemini-2.5-flash",
    "google/gemini-2.5-flash-lite", "google/gemini-2.5-pro",
    "deepseek/deepseek-chat", "openai/gpt-4.1-mini", "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.5",
]


def main():
    # Ключ: из переменной OKK_API_KEY или скрытым вводом. В консоли Windows при скрытом
    # вводе Ctrl+V часто НЕ вставляет текст — вставляйте правой кнопкой мыши.
    key = os.environ.get("OKK_API_KEY", "").strip()
    if not key:
        key = getpass.getpass("API-ключ AgentPlatform (ввод не отображается; "
                              "вставка — правой кнопкой мыши): ")
    raw_len = len(key)
    key = "".join(ch for ch in key if ch.isprintable()).strip()
    if not key:
        print("Ключ не введён.")
        return
    # Что реально получено — без раскрытия ключа: длина и края
    print(f"Получен ключ: {len(key)} символов, {key[:4]}…{key[-4:]}"
          + (f" (убрано невидимых символов: {raw_len - len(key)})" if raw_len != len(key) else ""))
    problem = api_key_problem(key)
    if problem:
        print(f"❌ {problem}")
        return
    client = OpenAI(api_key=key, base_url=LLM_BASE_URL, timeout=60, max_retries=0)

    print(f"\nШлюз: {LLM_BASE_URL}")
    try:
        ids = sorted(m.id for m in client.models.list())
        print(f"Моделей, доступных этому ключу по списку шлюза: {len(ids)}")
        for mid in ids:
            print("  ", mid)
    except Exception as e:
        ids = []
        print(f"Список моделей получить не удалось: {type(e).__name__}: {e}")

    print("\nПробный запрос к кандидатам:")
    for model in CANDIDATES:
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Ответь одним словом: ок"}],
                max_tokens=16, temperature=0)
            text = (r.choices[0].message.content or "").strip()
            print(f"  ✅ {model:22} работает (ответ: {text[:20]!r})")
        except Exception as e:
            msg = str(e).replace("\n", " ")
            print(f"  ❌ {model:22} {type(e).__name__}: {msg[:160]}")


if __name__ == "__main__":
    main()
