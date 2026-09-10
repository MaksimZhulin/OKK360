# -*- coding: utf-8 -*-
"""
Работа с LLM: коррекция терминов и ролей спикеров, учёт стоимости вызовов.
Вынесено из web_app.py.
"""
import re
import streamlit as st

from config import LLM_BASE_URL

# Показывать сырой ответ LLM в консоли/логе (для проверки, отдаёт ли tokengate готовую
# стоимость). Поставь False, когда наладишь и не нужно спамить лог.
SHOW_RAW_LLM = True

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text):
    """Убирает блок рассуждений <think>...</think> у reasoning-моделей (Qwen3 и пр.),
    чтобы он не попадал в транскрипт/JSON. Для обычных моделей текст не меняется."""
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def nothink_suffix(model):
    """Для Qwen3 мягко отключаем режим размышлений через маркер /no_think (быстрее,
    чище JSON). Для остальных моделей (gemini и т.д.) — пустая строка, ничего не меняем."""
    return " /no_think" if model and "qwen" in str(model).lower() else ""


# Контекст для локальных моделей Ollama. Дефолт Ollama ~4096 токенов обрезал бы наш
# большой промпт (критерии + транскрипт до 10к симв. + примеры ≈ 8-9к токенов) —
# расширяем, иначе анализ идёт по обрезанному тексту. Для облака параметр не шлём.
LOCAL_NUM_CTX = 12288


def ollama_options(local_mode):
    """kwargs для create(): локальным моделям задаём num_ctx (расширенный контекст).
    Облаку — пусто, вызов остаётся байт-в-байт прежним (gemini не трогаем)."""
    if not local_mode:
        return {}
    return {"extra_body": {"options": {"num_ctx": LOCAL_NUM_CTX}}}


COST_CURRENCY = "₽"
# Тариф в ₽ за 1000 токенов (вход/выход). tokengate даёт цену за 1М — делим на 1000.
LLM_PRICES = {
    "google/gemini-2.5-flash": {"in": 0.0324, "out": 0.27},   # РЕАЛЬНЫЙ тариф: 32.4 / 270 ₽ за 1М
    "deepseek/deepseek-chat":  {"in": 0.05, "out": 0.20},     # примерно — уточни у tokengate
    "openai/gpt-4o":           {"in": 0.55, "out": 2.20},     # примерно — уточни у tokengate
    "mistralai/mistral-nemo":  {"in": 0.02, "out": 0.05},     # примерно — уточни у tokengate
}
LLM_PRICE_DEFAULT = {"in": 0.05, "out": 0.20}

def _extract_real_cost(response):
    """Пытается достать ГОТОВУЮ стоимость из ответа прокси (если tokengate её отдаёт).
    Проверяет несколько известных мест. Возвращает число или None."""
    # 1) litellm кладёт сюда при проксировании
    hp = getattr(response, "_hidden_params", None)
    if isinstance(hp, dict) and hp.get("response_cost") is not None:
        return float(hp["response_cost"])
    # 2) нестандартные поля в теле ответа
    for obj in (getattr(response, "model_extra", None), response):
        for key in ("response_cost", "cost"):
            val = None
            if isinstance(obj, dict):
                val = obj.get(key)
            else:
                val = getattr(obj, key, None)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
    return None


def add_llm_cost(model, response):
    """Считает стоимость вызова и копит её в session_state для текущего файла.
    Приоритет: готовая стоимость из ответа tokengate (если есть) -> иначе по токенам.
    Вызывать после каждого запроса к LLM."""
    try:
        usage = getattr(response, "usage", None)
        # Кэшированные входные токены (если провайдер отдаёт) — из-за них реальная
        # стоимость заметно ниже оценки по токенам: кэш-префикс биллится ~в 4 раза дешевле.
        cached = 0
        det = getattr(usage, "prompt_tokens_details", None) if usage else None
        if det is not None:
            cached = getattr(det, "cached_tokens", None) or (
                det.get("cached_tokens") if isinstance(det, dict) else 0) or 0

        real = _extract_real_cost(response)
        if real is not None:
            cost, source = real, "tokengate (реальная)"
        elif usage:
            p = LLM_PRICES.get(model, LLM_PRICE_DEFAULT)
            cost = (usage.prompt_tokens / 1000.0) * p["in"] + \
                   (usage.completion_tokens / 1000.0) * p["out"]
            source = "оценка по токенам (кэш не учтён)"
        else:
            return

        prev = st.session_state.get("_file_llm_cost", 0.0)
        st.session_state["_file_llm_cost"] = prev + cost

        # Диагностика: по одной читаемой строке на вызов — видно источник цены,
        # токены (в т.ч. кэш) и накопленную сумму по файлу. SHOW_RAW_LLM=False отключает.
        if SHOW_RAW_LLM and usage:
            print(f"💰 [{model}] вход={usage.prompt_tokens} (кэш {cached}) "
                  f"выход={usage.completion_tokens} | цена={cost:.4f} ₽ "
                  f"[{source}] | по файлу: {prev + cost:.4f} ₽")
    except Exception as e:
        print(f"⚠️ Учёт стоимости: {e}")

def smart_text_correction(transcript_text, analysis_model, deepseek_key, local_mode=False):
    """Использует LLM для умной коррекции слов по контексту без изменения сути"""
    from openai import OpenAI
    
    if local_mode:
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    else:
        client = OpenAI(api_key=deepseek_key, base_url=LLM_BASE_URL)
    
    prompt = f"""Исправь ошибки транскрибации в тексте звонка. 
    Особенно обрати внимание на названия компаний и термины.
    Контекст: Это разговор с компанией "СтальМетУрал" (также может быть "СМУ", "Стальмет").
    
    🛑 СТРОГИЕ ПРАВИЛА (ЕСЛИ ТЫ ИХ НАРУШИШЬ, СИСТЕМА УПАДЕТ):
    1. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО перефразировать текст, менять порядок слов или удалять предложения.
    2. СОХРАНИ всю разговорную речь, паузы, корявые формулировки и обрывки фраз ровно так, как они есть. Не делай текст "литературным".
    3. Выполни только замену следующих искажений (если встретишь):
    
    Частые ошибки транскрибации которые нужно исправить:
    - "Сальметро", "Стермит Урал", "Дальмед Урал", "Тимотров", "Сталин металл", "не в термометрах", "Альметрол" → "СтальМетУрал"
    - "не взрослые, не взрослые" → "меня зовут Арсений, здравствуйте"
    - "Стартиковая" → "Пластиковая"
    - "ПМД", "ПМД, да, труба нажала" → "ПНД труба нужна"
    - "Да, не в термометрах" → "Данил, СтальМетУрал"
    - "физ лицо", "физическое лицо" → "физлицо"
    - "юр лицо", "юридическое лицо" → "юрлицо"
    - "темень собака металл ру", "темень собака металл.ру" → "tmn@stalmetural.ru"
    - "реквизиты", "карта предприятия" → оставить как есть
    - "ватсап", "whatsapp", "WhatsApp" → "WhatsApp"
    - "интернет сайт", "веб сайт" → "сайт"
    - "копир лица" → "физлица"
    - "я помощник МГУ", "помощник МГУ" → "чем помочь могу"
    - "протечный лист" → "просечный лист"
    - "Кто же у вас тут метрал?", "Кто же у вас тут СтальМетУрал?", "Кто же у вас тут металл?" → "Слушаю вас, СтальМетУрал"
    - "Слышал, что ты набрал?" → [ЭТУ ФРАЗУ НУЖНО ПОЛНОСТЬЮ УДАЛИТЬ, ЭТО ШУМ ГУДКОВ]
    - "Запись началась.", "Запись началась" → [ЭТУ ФРАЗУ НУЖНО ПОЛНОСТЬЮ УДАЛИТЬ ИЗ ТЕКСТА]
    
    Текст:
    {transcript_text[:10000]}
    Верни текст слово в слово, изменив только указанные термины. НЕ добавляй пояснений:"""
    
    try:
        response = client.chat.completions.create(
            model=analysis_model,
            messages=[
                {"role": "system", "content": "Ты бездушный алгоритм автозамены. Ты никогда не удаляешь оригинальные слова и не меняешь грамматику. Исправляй только ошибки, где слова сильно зажеванны и не представляются доступными для прочтения, сохраняй структуру." + nothink_suffix(analysis_model)},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000,
            **ollama_options(local_mode)
        )

        if not local_mode:
            add_llm_cost(analysis_model, response)
        corrected_text = strip_think(response.choices[0].message.content)

        if "Менеджер:" in corrected_text or "Клиент:" in corrected_text:
            return corrected_text
        else:
            return transcript_text
            
    except Exception as e:
        print(f"⚠️ Ошибка коррекции текста: {e}")
        return transcript_text

def correct_speaker_roles(transcript_text, analysis_model, deepseek_key, local_mode=False):
    """Использует LLM для глубокой коррекции ролей спикеров в транскрипции"""
    from openai import OpenAI
    
    if local_mode:
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    else:
        client = OpenAI(api_key=deepseek_key, base_url=LLM_BASE_URL)
    
    prompt = f"""Перед тобой транскрипция телефонного звонка в компанию "СтальМетУрал". 
Из-за технических особенностей записи нейросеть могла:
1. Оставить теги в формате "SPEAKER_01 / SPEAKER_02" или перепутать менеджера и клиента.
2. Склеить фразы двух разных людей в один длинный абзац.

ТВОЯ ЗАДАЧА — ВЫСТУПИТЬ РЕДАКТОРОМ И ВОССТАНОВИТЬ ЛОГИКУ ДИАЛОГА.

🛑 СТРОЖАЙШИЕ ПРАВИЛА (ШТРАФ ЗА НАРУШЕНИЕ):
1. В САМОМ ТЕКСТЕ РЕПЛИК НЕЛЬЗЯ УДАЛЯТЬ НИ ОДНОГО СЛОВА! Сохраняй все ошибки распознавания (например, "копир лица").
2. Твоя главная цель — расставить правильные теги "👨‍💼 Менеджер:" и "👤 Клиент:".
3. ОБЯЗАТЕЛЬНО переноси строки и разделяй спикеров, если в одном абзаце склеились фразы двух людей! Внимательно ищи логические стыки. Например, если в тексте идет рассказ клиента ("полиэтиленовая 100 мм"), а дальше сразу вопрос менеджера ("ПНД труба нужна?"), ты ОБЯЗАН разорвать этот абзац на две отдельные реплики "👤 Клиент:" и "👨‍💼 Менеджер:".

КАК ОПРЕДЕЛЯТЬ РОЛИ (СМОТРИ НА СМЫСЛ):
- МЕНЕДЖЕР (👨‍💼): 
  * Принимает звонок. Если звучит фраза "Меня Юля зовут, чем могу помочь", "Здравствуйте, СтальМетУрал" — это 100% Менеджер!
  * Проверяет наличие ("сейчас посмотрю", "уточню на складе", "посчитаю").
  * Запрашивает реквизиты ("скиньте карту предприятия", "продиктуйте номер").
- КЛИЕНТ (👤): 
  * Звонит, чтобы купить ("мне нужны листы", "есть ли трубы?"). 
  * Спрашивает цену ("сколько будет стоить?"). 
  * Диктует свой телефон или почту, называет свой город.

Транскрипция для исправления:
{transcript_text[:10000]}

Верни логичный, правильный диалог в формате "👨‍💼 Менеджер: ..." и "👤 Клиент: ...". НЕ добавляй никаких пояснений от себя, только текст диалога."""

    try:
        response = client.chat.completions.create(
            model=analysis_model,
            messages=[
                {"role": "system", "content": "Ты логический редактор. Твоя задача — распутать диалог, переставив теги ролей там, где это необходимо по смыслу. Ты не меняешь слова, но можешь разбивать склеенные абзацы." + nothink_suffix(analysis_model)},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
            **ollama_options(local_mode)
        )

        if not local_mode:
            add_llm_cost(analysis_model, response)
        corrected_text = strip_think(response.choices[0].message.content)

        if "👨‍💼 Менеджер:" in corrected_text and "👤 Клиент:" in corrected_text:
            return corrected_text
        else:
            return transcript_text
            
    except Exception as e:
        print(f"⚠️ Ошибка коррекции ролей: {e}")
        return transcript_text
