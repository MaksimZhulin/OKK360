# -*- coding: utf-8 -*-
"""
Диагностика доступа к моделям облачного шлюза (AgentPlatform).

    venv\\Scripts\\python check_models.py

Спрашивает API-ключ (ввод скрыт, ключ никуда не сохраняется), показывает модели,
которые шлюз отдаёт этому ключу, и пробует крошечный запрос к нашим кандидатам.
Стоимость проверки — доли копейки.
"""
import sys
import getpass

from openai import OpenAI

from config import LLM_BASE_URL

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

CANDIDATES = [
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro",
    "deepseek-chat", "gpt-4.1-mini", "gpt-4o-mini", "claude-haiku-4.5", "claude-sonnet-4.5",
]


def main():
    key = getpass.getpass("API-ключ AgentPlatform (ввод не отображается): ").strip()
    if not key:
        print("Ключ не введён.")
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
