# -*- coding: utf-8 -*-
"""
Работа с LLM: коррекция терминов и ролей спикеров, учёт стоимости вызовов.
Вынесено из web_app.py.
"""
import re
import threading

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


# Контекст для локальных моделей Ollama. Дефолт Ollama ~4096 токенов обрезал бы наш
# большой промпт (критерии + транскрипт до 10к симв. + примеры ≈ 8-9к токенов) —
# расширяем, иначе анализ идёт по обрезанному тексту. Для облака параметр не шлём.
LOCAL_NUM_CTX = 12288


OLLAMA_URL = "http://localhost:11434/api/chat"


def ollama_native_chat(messages, model, temperature=0.3, max_tokens=2048,
                       num_ctx=LOCAL_NUM_CTX, timeout=1800):
    """Вызов локальной модели через НАТИВНЫЙ /api/chat Ollama (а не OpenAI /v1).
    Почему нативный: (1) он блокирующе ждёт загрузку модели (нет гонки 503 на
    холодном старте, как в /v1); (2) поддерживает think=False — единственный
    надёжный способ выключить «размышления» Qwen3 (маркер /no_think в тексте на
    Ollama 0.34 не работает, из-за чего content приходил пустым, а JSON не парсился).
    Возвращает строку content."""
    import requests
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,   # выключаем reasoning — иначе пустой/битый content
        "keep_alive": "15m",
        "options": {"num_ctx": num_ctx, "temperature": temperature,
                    "num_predict": max_tokens},
    }
    r = requests.post(OLLAMA_URL, json=body, timeout=timeout, proxies=NO_PROXY_LOCAL)
    r.raise_for_status()
    return (r.json().get("message", {}) or {}).get("content", "") or ""


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_VERSION_URL = "http://localhost:11434/api/version"


# Все локальные вызовы к Ollama идут МИМО системного прокси. Иначе requests при
# заданном HTTP(S)_PROXY гонит запрос к localhost через прокси -> тот до 127.0.0.1
# не достучится (таймаут/отказ), и сервер ошибочно считается недоступным.
NO_PROXY_LOCAL = {"http": None, "https": None}


def _ollama_alive(timeout=2):
    """Отвечает ли локальный сервер Ollama."""
    import requests
    try:
        return requests.get(OLLAMA_VERSION_URL, timeout=timeout, proxies=NO_PROXY_LOCAL).ok
    except Exception:
        return False


def _find_ollama_exe():
    """Ищет ollama.exe: сперва в PATH, потом стандартная установка в LOCALAPPDATA."""
    import shutil
    import os
    exe = shutil.which("ollama")
    if exe:
        return exe
    cand = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    return cand if os.path.exists(cand) else None


def ensure_ollama_running(wait=40):
    """Гарантирует, что локальный сервер Ollama поднят — БЕЗ системного прокси.
    Зачем без прокси: при активном HTTP(S)_PROXY Ollama 0.34 виснет на загрузке модели
    (сетевой вызов уходит на прокси) -> 503 -> локальный анализ пустой. Раньше это решал
    ручной start_ollama.bat в отдельном окне; теперь приложение поднимает сервер само,
    фоном и без окна. Если сервер уже отвечает — ничего не делаем.
    Возвращает True, если Ollama доступна к моменту выхода."""
    import os
    import time
    import subprocess

    if _ollama_alive():
        return True

    exe = _find_ollama_exe()
    if not exe:
        print("⚠️ ollama.exe не найден — установи Ollama или запусти сервер вручную")
        return False

    # Копия окружения БЕЗ прокси — ключевой момент (иначе Ollama зависнет на загрузке модели).
    env = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    env["NO_PROXY"] = "*"
    env["OLLAMA_NO_CLOUD"] = "true"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"

    creationflags = 0
    if os.name == "nt":  # фон без консольного окна, переживает закрытие родителя
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        subprocess.Popen([exe, "serve"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=creationflags)
    except Exception as e:
        print(f"⚠️ Не удалось запустить Ollama serve: {e}")
        return False

    for _ in range(wait):
        if _ollama_alive():
            print("✅ Ollama поднят автоматически (без прокси)")
            return True
        time.sleep(1)
    print("⚠️ Ollama не ответил за отведённое время")
    return False


def unload_ollama_model(model):
    """Просит Ollama немедленно выгрузить модель из VRAM (keep_alive=0).
    Нужно перед Фазой транскрибации: если Qwen висит с прошлого прогона (keep_alive 15m),
    WhisperX не влезет в 12 ГБ. Ошибки глушим — если Ollama не запущена/модели нет, не страшно."""
    import requests
    try:
        requests.post(OLLAMA_GENERATE_URL,
                      json={"model": model, "keep_alive": 0}, timeout=30,
                      proxies=NO_PROXY_LOCAL)
        print(f"🔄 Ollama: запрошена выгрузка {model} из VRAM")
    except Exception as e:
        print(f"⚠️ Выгрузка Ollama ({model}): {e}")


COST_CURRENCY = "₽"
# Тариф в ₽ за 1000 токенов (вход/выход). tokengate даёт цену за 1М — делим на 1000.
LLM_PRICES = {
    # тарифы AgentPlatform (каталог моделей, октябрь 2026): ₽ за 1М / 1000
    "gemini-2.5-flash":      {"in": 0.0324, "out": 0.27},    # 32.4 / 270 ₽ за 1М
    "gemini-2.5-flash-lite": {"in": 0.0108, "out": 0.0432},  # 10.8 / 43.2 ₽ за 1М
    "deepseek-chat":         {"in": 0.03456, "out": 0.09612},  # 34.56 / 96.12 ₽ за 1М
    "gpt-4o":                {"in": 0.27, "out": 1.08},      # 270 / 1080 ₽ за 1М
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


# Стоимость вызовов по текущему файлу. Обработка идёт в фоновом потоке (не в Streamlit),
# поэтому копим не в session_state, а здесь; прогон в процессе всегда один.
_file_cost = 0.0
_cost_lock = threading.Lock()


def reset_file_cost():
    """Обнуляет счётчик перед новым файлом."""
    global _file_cost
    with _cost_lock:
        _file_cost = 0.0


def get_file_cost():
    """Сколько уже потрачено на текущий файл (₽)."""
    with _cost_lock:
        return _file_cost


def add_llm_cost(model, response):
    """Считает стоимость вызова и копит её в счётчике текущего файла.
    Приоритет: готовая стоимость из ответа tokengate (если есть) -> иначе по токенам.
    Вызывать после каждого запроса к LLM."""
    global _file_cost
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

        with _cost_lock:
            prev = _file_cost
            _file_cost = prev + cost

        # Диагностика: по одной читаемой строке на вызов — видно источник цены,
        # токены (в т.ч. кэш) и накопленную сумму по файлу. SHOW_RAW_LLM=False отключает.
        if SHOW_RAW_LLM and usage:
            print(f"💰 [{model}] вход={usage.prompt_tokens} (кэш {cached}) "
                  f"выход={usage.completion_tokens} | цена={cost:.4f} ₽ "
                  f"[{source}] | по файлу: {prev + cost:.4f} ₽")
    except Exception as e:
        print(f"⚠️ Учёт стоимости: {e}")

# Облачные вызовы: SDK по умолчанию ждёт ответа до 10 минут и сам повторяет 2 раза —
# на подвисшем шлюзе файл висел десятки минут. Ждём меньше и повторяем сами, с логом.
CLOUD_TIMEOUT = 180            # сек на один запрос
CLOUD_RETRY_PAUSES = (10, 30)  # паузы перед 2-й и 3-й попыткой


class LLMCancelled(Exception):
    """Прогон остановлен пользователем — повторные попытки не делаем."""


def llm_chat(messages, model, api_key, local_mode=False, temperature=0.3, max_tokens=2500,
             label="", cancel=None):
    """Единая точка вызова LLM: локально — нативный Ollama, в облаке — OpenAI-совместимый
    прокси с таймаутом и повторами при сетевых сбоях / перегрузке шлюза.
    Стоимость облачного вызова копится в счётчике файла (add_llm_cost).
    cancel — threading.Event «Стоп»: прерывает паузу между попытками (сам запрос
    дорабатывает до ответа/таймаута — оборвать его на полпути нельзя).
    Возвращает текст ответа без <think>; при исчерпании попыток пробрасывает исключение."""
    import time
    t0 = time.time()
    if local_mode:
        text = strip_think(ollama_native_chat(messages, model, temperature=temperature,
                                              max_tokens=max_tokens))
        print(f"⏱️ [{label}] {time.time() - t0:.0f} с (локально, {model})")
        return text

    from openai import (OpenAI, APITimeoutError, APIConnectionError,
                        RateLimitError, InternalServerError)
    retryable = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL,
                    timeout=CLOUD_TIMEOUT, max_retries=0)
    attempts = len(CLOUD_RETRY_PAUSES) + 1
    for attempt in range(1, attempts + 1):
        t_try = time.time()
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        except retryable as e:
            print(f"⚠️ [{label}] попытка {attempt}/{attempts} не удалась через "
                  f"{time.time() - t_try:.0f} с: {type(e).__name__}: {e}")
            if attempt == attempts:
                raise
            pause = CLOUD_RETRY_PAUSES[attempt - 1]
            print(f"   повтор через {pause} с")
            if cancel is not None:
                if cancel.wait(pause):
                    raise LLMCancelled("остановлено пользователем")
            else:
                time.sleep(pause)
            continue
        add_llm_cost(model, response)
        print(f"⏱️ [{label}] {time.time() - t0:.0f} с ({model}, попытка {attempt})")
        return strip_think(response.choices[0].message.content or "")


def check_model_access(model, api_key):
    """Пробный крошечный запрос к облачной модели ДО начала прогона: неверный ключ,
    закрытая для ключа модель или лежащий шлюз обнаруживаются за секунды, а не после
    часа транскрибации (когда все расшифровки пропадают). None — всё в порядке,
    иначе — понятное описание проблемы."""
    from openai import AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError
    try:
        llm_chat([{"role": "user", "content": "Ответь одним словом: ок"}], model, api_key,
                 temperature=0, max_tokens=16, label=f"проверка доступа к {model}")
        return None
    except AuthenticationError:
        return ("API-ключ не принят шлюзом (401). Проверьте ключ в боковой панели — "
                "его можно скопировать в AgentPlatform → Доступ к моделям → Ключи.")
    except PermissionDeniedError as e:
        return (f"Ключу не разрешена модель «{model}» (403: {e}). Разрешите эту модель для ключа "
                "в AgentPlatform → Доступ к моделям → Ключи (или создайте ключ с доступом к ней), "
                "либо выберите другую модель.")
    except NotFoundError:
        return f"Шлюз не знает модель «{model}» (404) — проверьте название в каталоге моделей."
    except BadRequestError as e:
        return f"Шлюз отклонил запрос к модели «{model}» (400): {e}"
    except Exception as e:
        return (f"Шлюз ИИ недоступен ({type(e).__name__}: {e}). Транскрибацию не начинаю — "
                "без анализа её результаты всё равно пропадут. Попробуйте позже.")


def smart_text_correction(transcript_text, analysis_model, deepseek_key, local_mode=False, label="", cancel=None):
    """Использует LLM для умной коррекции слов по контексту без изменения сути"""
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
    
    messages = [
        {"role": "system", "content": "Ты бездушный алгоритм автозамены. Ты никогда не удаляешь оригинальные слова и не меняешь грамматику. Исправляй только ошибки, где слова сильно зажеванны и не представляются доступными для прочтения, сохраняй структуру."},
        {"role": "user", "content": prompt}
    ]
    try:
        corrected_text = llm_chat(messages, analysis_model, deepseek_key, local_mode,
                                  temperature=0.1, max_tokens=4000,
                                  label=f"коррекция текста: {label}", cancel=cancel)

        if "Менеджер:" in corrected_text or "Клиент:" in corrected_text:
            return corrected_text
        else:
            print(f"⚠️ [коррекция текста: {label}] в ответе нет меток ролей — оставляю исходный текст")
            return transcript_text

    except Exception as e:
        print(f"⚠️ [коррекция текста: {label}] не удалась, оставляю исходный текст: {e}")
        return transcript_text

def correct_speaker_roles(transcript_text, analysis_model, deepseek_key, local_mode=False, label="", cancel=None):
    """Использует LLM для глубокой коррекции ролей спикеров в транскрипции"""
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

    messages = [
        {"role": "system", "content": "Ты логический редактор. Твоя задача — распутать диалог, переставив теги ролей там, где это необходимо по смыслу. Ты не меняешь слова, но можешь разбивать склеенные абзацы."},
        {"role": "user", "content": prompt}
    ]
    try:
        corrected_text = llm_chat(messages, analysis_model, deepseek_key, local_mode,
                                  temperature=0.2, max_tokens=4000,
                                  label=f"коррекция ролей: {label}", cancel=cancel)

        if "👨‍💼 Менеджер:" in corrected_text and "👤 Клиент:" in corrected_text:
            return corrected_text
        else:
            print(f"⚠️ [коррекция ролей: {label}] в ответе нет обеих ролей — оставляю исходный текст")
            return transcript_text

    except Exception as e:
        print(f"⚠️ [коррекция ролей: {label}] не удалась, оставляю исходный текст: {e}")
        return transcript_text
