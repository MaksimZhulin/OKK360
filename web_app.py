import os
import sys
import json
import time
import traceback
from datetime import datetime

# Распознавание номенклатуры из звонка по каталогу (data/catalog.json)
from nomenclature import match_mentions, format_nomenclature, format_volumes

# Windows: консоль по умолчанию cp1251 и падает на эмодзи в print() — принудительно UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from auto_learning import save_to_knowledge_base, calculate_confidence, find_similar_calls, get_database_stats
from config import SPREADSHEET_ID, CREDENTIALS_FILE, SHEET_NAME, SHEET_TITLE
from audio import (
    WHISPERX_AVAILABLE, preprocess_audio, pad_audio_simple_silence,
    transcribe_with_yandex, transcribe_with_whisperx_diarization,
    unload_whisperx_from_vram, is_cuda_error,
)
from llm_analysis import (
    COST_CURRENCY, llm_chat, smart_text_correction, correct_speaker_roles,
    unload_ollama_model, ensure_ollama_running,
)
from run_log import RunLog

def flatten_analysis(obj):
    """ИИ иногда оборачивает поля в объекты-секции ('ОБЩАЯ ИНФОРМАЦИЯ': {...},
    'БИНАРНЫЕ КРИТЕРИИ': {...}) — по заголовкам из промпта. Тогда на верхнем уровне лежат
    не наши ключи (topic, establishing_contact...), а названия секций, и всё читается как
    «Не определено». Разворачиваем в ПЛОСКИЙ словарь по именам конечных ключей (они у нас
    уникальны). Списки/скаляры (например nomenclature_raw) остаются значениями как есть.
    Если ответ уже плоский — ничего не меняется."""
    flat = {}

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            else:
                flat[k] = v

    if isinstance(obj, dict):
        walk(obj)
        return flat
    return obj


# Критерии оценки и подсчёт баллов вынесены в scoring.py
from scoring import (
    NEED_CRITERIA, NEED_KEYS, OBJECTION_CRITERIA, OBJ_KEYS,
    DOZHIM_CRITERIA, DOZHIM_KEYS, CONTACT_CRITERIA, CONTACT_KEYS,
    NEXTSTEP_CRITERIA, NEXTSTEP_KEYS, SPEECH_CRITERIA, SPEECH_KEYS,
    block_score_1_5, block_max, grand_max, build_recommendation,
)


# Окно лога на экране прогона: сколько последних строк показывать
LOG_VIEW_LINES = 15
# Сколько файлов подряд должно упасть с ошибкой GPU (уже после перезагрузки моделей),
# чтобы считать видеокарту сбойной и не тратить время на остальные файлы Фазы 1
GPU_FAIL_LIMIT = 3


def fmt_duration(sec):
    """Секунды -> «1 ч 5 мин» / «3 мин 10 сек» / «12 сек»."""
    sec = int(round(sec or 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} ч {m} мин"
    if m:
        return f"{m} мин {s} сек"
    return f"{s} сек"


def build_analysis_messages(transcript_text, examples_text):
    """Главный промпт анализа звонка (ответ — плоский JSON)."""
    prompt = f"""Проанализируй транскрипцию звонка. Текст: {transcript_text[:10000]}
                {examples_text}

Выведи результат СТРОГО в формате JSON. Используй ключи, указанные ниже.

ОБЩАЯ ИНФОРМАЦИЯ:
1. "topic": Основная тема звонка (1-3 слова).
2. "call_type": Тип звонка (выбери строго одно: "Первичный" или "Повторный"). Ставь "Повторный", если клиент говорит "я вам скинул", "мы общались", "я по счету" или контекст явно указывает, что диалог уже ведется.
3. "technical_issue": Технический брак связи (1 = ДА, 0 = НЕТ). Ставь 1, если в тексте есть фразы "алло", "вас не слышно", "пропадает связь", много обрывков слов или звонок внезапно оборвался без прощания.
4. "client_request": Запрос клиента (1-2 предложения).
5. "solution": Предложенное решение или результат (1-2 предложения).
6. "urgency": Срочность (выбери одно: Низкая / Средняя / Высокая).
7. "client_mood": Настроение клиента (выбери одно: Нейтральное / Заинтересованное / Раздраженное / Довольное / Сомневающееся).
8. "manager_actions": Ключевые действия менеджера (1-2 предложения).
9. "recommendations": Краткое (1-2 предложения) ФАКТИЧЕСКОЕ наблюдение по сути разговора: что менеджер сделал хорошо и что конкретно упустил. БЕЗ слов "отлично/хорошо/плохо", без процентов и без общей оценки — только конкретика по содержанию. Итоговую оценку и зоны роста посчитает система отдельно.
9a. "nomenclature_raw": Массив ОБЪЕКТОВ — ВСЕ товары, которые обсуждались в звонке как интересующие клиента. Каждый объект: {{"item": "...", "size": "...", "grade": "...", "quantity": "...", "unit": "..."}}. КРИТИЧЕСКИ ВАЖНО правильно раскладывать по полям — не смешивай размер и марку.
Поле "item" (ТИП товара, БЕЗ размера и БЕЗ марки): материал + тип. Примеры: "круг стальной", "лист горячекатаный", "труба профильная", "труба бесшовная", "чугунная батарея", "квадрат стальной", "лист оцинкованный", "алюминиевый лист".
- НЕ клади в item числа-размеры и марки — для них отдельные поля size и grade.
- Если назван только код/модель (МС-140), тип товара всё равно допиши в item ("чугунная батарея"), а сам код — в grade.
Поле "size" (РАЗМЕР/габарит, что клиент назвал как размер) — только цифры/габариты, единицы можно опустить:
- Один размер -> "8". Составной -> "50х20" или "1200х2500". Толщина/диаметр — тоже сюда ("0.8", "65").
- НЕСКОЛЬКО размеров одного товара разделяй ";" : "70; 80; 75". Диапазон -> "25-40".
- Обозначения ДУ/DN оставляй как есть: "ДУ150".
Поле "grade" (МАРКА стали/сплава/модель изделия) — буквенно-цифровое обозначение: "18ХГТ", "Ст35", "А500С", "09Г2С", "25Б2", "МС500", "16S13NG", "9ГС", "40Х", "10Г2ФБЮ", "АМЦМ", "ЛС-12". Несколько -> через ";". Нет марки -> "".
⚠️ Различай size и grade: "18ХГТ" и "Ст35" и "25Б2" — это МАРКА (grade), а не размер. "70", "50х20", "0.8" — это РАЗМЕР (size). Пример: клиент «кругляк 70, марка 18ХГТ» -> item "круг стальной", size "70", grade "18ХГТ" (НЕ size "18").
Поле "quantity" (СКОЛЬКО заказать) — только число: устную речь в цифры ("тонн пять" -> "5", "полтонны" -> "0.5", "пара" -> "2"). Диапазон -> "5-10". Это КОЛИЧЕСТВО, а НЕ размер. Не названо -> "".
Поле "unit" (единица количества) — СТРОГО из списка: "т", "кг", "м", "м.п.", "шт", "лист", "пруток", "хлыст", "рулон", "м²", "упак". Транспорт ("фура", "газель") — впиши словом. Не названа -> "".
Пример: [{{"item": "круг стальной", "size": "70; 80; 75", "grade": "18ХГТ", "quantity": "", "unit": ""}}, {{"item": "лист оцинкованный", "size": "0.8", "grade": "Ст35", "quantity": "20", "unit": "лист"}}].
Только то, что реально есть в тексте — НЕ выдумывай товары. Если товары не обсуждались — верни пустой массив [].

БИНАРНЫЕ КРИТЕРИИ (Оценивай строго: 1 = ДА, 0 = НЕТ):
⚠️ ПРАВИЛО ДЛЯ ПОВТОРНЫХ ЗВОНКОВ: Если "call_type" == "Повторный", ты ОБЯЗАН ставить 1 (оправдано) в критериях "establishing_contact", "client_type" и "clarifying_questions", так как этот этап уже пройден ранее.
⚠️ ПРАВИЛО ДЛЯ БРАКА СВЯЗИ: Если "technical_issue" == 1, оценивай всё ОБЪЕКТИВНО по тому, что слышишь (ставь 0, если чего-то нет). Система сама исключит этот звонок из KPI, не завышай оценки искусственно!

10. "establishing_contact": Установление контакта. (Приветствие + Название компании + Имя). 
11. "client_type": Тип клиента (физ/юр лицо). 
12. "clarifying_questions": Выявление потребностей. (Минимум 2 вопроса о товаре).
13. "knowledge_quality": Компетентность. (1 = уверенные ответы, 0 = незнание/ошибки).
14. "software_proficiency": Работа с ПО. (1 = быстрый поиск, 0 = долгие неловкие паузы).
15. "politeness": Вежливость. (1 = тактично, 0 = грубо/сухо).

БЛОК "ПОТРЕБНОСТЬ" (Оценивай строго: 1 = ДА, 0 = НЕТ). Насколько глубоко менеджер выявил ситуацию клиента. Ставь 1 ТОЛЬКО если в разговоре это реально прозвучало:
18. "need_purpose": Узнал цель покупки клиента (например, стройка, ремонт, перепродажа, производство).
19. "need_project_details": Узнал детали проекта клиента (например, строительство коттеджного поселка, объём, что именно делают).
20. "need_geography": Узнал географию работ или расположение объекта (город, регион, адрес доставки).
21. "need_supplier_criteria": Узнал критерии выбора поставщика — что важно клиенту (цена, сроки, логистика, качество, наличие).
22. "need_interaction_terms": Узнал условия взаимодействия (оплата по счёту, отсрочка, форма оплаты, документы).
23. "need_purchase_frequency": Узнал частоту закупок (разовая покупка или регулярные поставки).
24. "need_competitors": Выявил конкурентов или конкурентные предложения (где ещё смотрел, какие цены называли).
25. "need_other_projects": Узнал о других проектах клиента (иные объекты, будущие потребности).

БЛОК "ВОЗРАЖЕНИЯ" (Оценивай строго: 1 = ДА, 0 = НЕТ). Как менеджер (МОП) отработал возражения и сомнения клиента.
"had_objections": Были ли у клиента возражения или сомнения в звонке? (1 = да, были; 0 = нет, клиент не возражал). Если 0 — блок не учитывается (Н/У).
⚠️ Отмечай 1 ТОЛЬКО за реально выполненные действия. Если возражений не было, пункты 26-32 будут 0 — это нормально, блок просто не зачтётся.
26. "obj_active_listening": Использовал активное слушание — принял сторону клиента, показал понимание ("понимаю вас", "согласен, это важно").
27. "obj_no_interrupt": Не перебивал клиента, дал полностью выразить мысль/возражение.
28. "obj_no_argue": Не спорил с клиентом, не шёл в конфликт (фразы "вы не правы", "это не так" = 0).
29. "obj_clarify_reason": Задавал уточняющие вопросы, чтобы выявить истинную причину возражения.
30. "obj_direct_answer": Прямо ответил на сомнения клиента, не ушёл от ответа.
31. "obj_arguments": Привёл аргументы и контраргументы в пользу предложения.
32. "obj_leading_questions": Задавал наводящие вопросы так, что клиент сам пришёл к закрытию своих сомнений.

БЛОК "ДОЖИМ" (Оценивай строго: 1 = ДА, 0 = НЕТ). Как менеджер дожимал клиента к сделке / следующему шагу:
33. "dozhim_concrete_solution": Предложил конкретное решение после закрытия возражений.
34. "dozhim_action_plan": Предложил план действий и вовлёк клиента (что и когда делаем дальше).
35. "dozhim_detailed_offer": Детализировал предложение исходя из потребностей клиента.
36. "dozhim_no_pressure": Не давил на клиента и не создавал стресс/негатив.
37. "dozhim_alternative": Предложил альтернативное решение (если основное не подошло).
38. "dozhim_better_terms": Предложил условия лучше озвученных ранее (скидка, бонус, доставка).
39. "dozhim_scarcity": Создал ограничение по времени или составу предложения (дефицит, "только сегодня").
40. "dozhim_upsell": Предложил дополнительные услуги или товары (допродажа).

БЛОК "КОНТАКТНЫЕ ДАННЫЕ / Кл-счёт" (Оценивай строго: 1 = ДА, 0 = НЕТ). Насколько полно менеджер собрал контактные данные клиента:
41. "contact_preferred_channel": Узнал предпочитаемый способ связи (звонок, WhatsApp, почта).
42. "contact_email": Узнал или уточнил контактную электронную почту.
43. "contact_phone": Узнал или уточнил контактный номер телефона.
44. "contact_other_person": Узнал иное контактное лицо для связи (ЛПР, снабженец, бухгалтер).
45. "contact_convenient_time": Узнал удобное время для связи с учётом часового пояса.
46. "contact_additional": Узнал дополнительные контактные данные (мессенджеры, доб. номер).

БЛОК "СЛЕДУЮЩИЙ ШАГ / ЗАВЕРШЕНИЕ" (Оценивай строго: 1 = ДА, 0 = НЕТ). Как менеджер закрыл звонок и зафиксировал следующий шаг:
47. "next_fixed_agreement": Зафиксировал договорённость или промежуточный результат разговора.
48. "next_time_set": Установил конкретные дату и время следующего контакта.
49. "next_own_action": Чётко обозначил своё следующее действие ("я отправлю счёт", "перезвоню в среду").
50. "next_result_details": Обозначил характеристики результата следующего шага (например, какие товары будут в счёте).
51. "next_benefits": Обозначил преимущества следующего шага для клиента.
52. "next_polite_close": Вежливо завершил диалог (попрощался, поблагодарил).

БЛОК "РЕЧЬ" (Оценивай строго: 1 = ДА, 0 = НЕТ). Качество речи менеджера:
53. "speech_literacy": Грамотность — логичное и последовательное выражение мысли, без сумбура.
54. "speech_empathy": Эмпатия — использовал фразы активного слушания ("понимаю вас", "согласен", "верно", "конечно" и подобные).

⚠️ СТРУКТУРА JSON: все ключи — на ОДНОМ верхнем уровне (плоский объект). НЕ группируй их в под-объекты по секциям (никаких "ОБЩАЯ ИНФОРМАЦИЯ": {{...}}). Заголовки секций выше — только для твоего удобства, в JSON их быть не должно.

Верни ТОЛЬКО JSON, без Markdown-разметки и без пояснений:"""

    return [
        {"role": "system", "content": "Ты — опытный аналитик колл-центра, специализирующийся на глубоком анализе транскрипций звонков. Твоя цель — предоставить всестороннюю, объективную и профессиональную оценку взаимодействия между клиентом и агентом, выявить ключевые паттерны, проблемы и предложить конкретные, действенные рекомендации. Отвечай строго в формате JSON."},
        {"role": "user", "content": prompt}
    ]


def parse_analysis_response(result_text):
    """Достаёт JSON из ответа ИИ и разворачивает секции в плоский словарь.
    Не удалось — пустой словарь (поля потом заполнятся «Не определено»/0)."""
    json_start = result_text.find('{')
    json_end = result_text.rfind('}')

    if json_start != -1 and json_end != -1:
        json_str = result_text[json_start:json_end + 1]
        try:
            # flatten: ИИ порой вкладывает поля в объекты-секции — разворачиваем
            analysis_result = flatten_analysis(json.loads(json_str))
        except json.JSONDecodeError:
            print(f"⚠️ Ошибка парсинга JSON от ИИ. Сырой текст: {result_text}")
            analysis_result = {}
    else:
        print(f"⚠️ ИИ не вернул фигурные скобки. Сырой текст: {result_text}")
        analysis_result = {}
    return analysis_result


def normalize_analysis(analysis_result):
    """Приводит ответ ИИ к рабочему виду (на месте): недостающие поля, бинарные 0/1,
    флаг возражений, рекомендация от системы, номенклатура и объёмы по каталогу."""
    required_fields = ["topic", "call_type", "technical_issue", "client_request", "solution", "urgency", "client_mood", "manager_actions", "recommendations",
                        "establishing_contact", "client_type", "clarifying_questions",
                        "knowledge_quality", "software_proficiency", "politeness"] + NEED_KEYS + OBJ_KEYS + DOZHIM_KEYS + CONTACT_KEYS + NEXTSTEP_KEYS + SPEECH_KEYS

    for field in required_fields:
        if field not in analysis_result:
            analysis_result[field] = "Не определено" if field in ["topic", "call_type", "client_request", "solution", "urgency", "client_mood", "manager_actions", "recommendations"] else 0

    binary_fields = ["establishing_contact", "client_type", "clarifying_questions",
                    "knowledge_quality", "software_proficiency", "politeness"]

    for field in binary_fields + NEED_KEYS + OBJ_KEYS + DOZHIM_KEYS + CONTACT_KEYS + NEXTSTEP_KEYS + SPEECH_KEYS:
        if field in analysis_result:
            try:
                val = analysis_result[field]
                if isinstance(val, int):
                    analysis_result[field] = 1 if val > 0 else 0
                elif isinstance(val, str):
                    analysis_result[field] = 1 if val.lower().strip() in ["1", "да", "yes", "true"] else 0
                else:
                    analysis_result[field] = 0
            except:
                analysis_result[field] = 0
        else:
            analysis_result[field] = 0

    # Флаг наличия возражений (по умолчанию считаем, что были — честный подсчёт)
    _hv = analysis_result.get("had_objections", 1)
    if isinstance(_hv, str):
        analysis_result["had_objections"] = 1 if _hv.lower().strip() in ["1", "да", "yes", "true"] else 0
    else:
        try:
            analysis_result["had_objections"] = 1 if int(_hv) > 0 else 0
        except Exception:
            analysis_result["had_objections"] = 1

    # Рекомендация строится в коде (вердикт по % + слабые зоны), а не ИИ
    analysis_result["recommendations"] = build_recommendation(analysis_result)

    # Номенклатура: сырые упоминания от ИИ -> каноничные позиции каталога.
    # nomenclature_raw = массив объектов {item, quantity, unit}; на строки
    # (старый формат) тоже реагируем — для обратной совместимости.
    try:
        _raw_noms = analysis_result.get("nomenclature_raw", []) or []
        if isinstance(_raw_noms, (str, dict)):
            _raw_noms = [_raw_noms]
        _mentions, _volumes = [], []
        for _r in _raw_noms:
            if isinstance(_r, dict):
                _mentions.append({
                    "item": str(_r.get("item", "")).strip(),
                    "size": str(_r.get("size", "")).strip(),
                    "grade": str(_r.get("grade", "")).strip(),
                })
                _volumes.append({
                    "quantity": str(_r.get("quantity", "")).strip(),
                    "unit": str(_r.get("unit", "")).strip(),
                })
            else:
                _mentions.append(str(_r).strip())
                _volumes.append({"quantity": "", "unit": ""})
        _nom_items = match_mentions(_mentions)
        analysis_result["nomenclature"] = _nom_items
        analysis_result["nomenclature_str"] = format_nomenclature(_nom_items)
        analysis_result["volume_str"] = format_volumes(_nom_items, _volumes)
    except Exception as _nom_err:
        print(f"⚠️ Матчинг номенклатуры: {_nom_err}")
        analysis_result["nomenclature"] = []
        analysis_result["nomenclature_str"] = ""
        analysis_result["volume_str"] = ""


# =============== НАСТРОЙКИ STREAMLIT ===============
st.set_page_config(page_title="Массовый анализ звонков", page_icon="🎤", layout="wide")
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'sheets_saved' not in st.session_state:
    st.session_state.sheets_saved = None  # итог последней выгрузки в Sheets (строки + ссылка)

st.title("🎤 МАССОВЫЙ АНАЛИЗ ТЕЛЕФОННЫХ ЗВОНКОВ")
st.markdown("---")

# =============== БОКОВАЯ ПАНЕЛЬ ===============
with st.sidebar:
    st.header("⚙️ Настройки")
    
    local_mode = st.checkbox("🖥️ Локальная модель (Ollama)", value=False, key="local_mode_checkbox")

    if local_mode:
        analysis_model = st.selectbox(
            "🧠 Локальная модель (Ollama)",
            ["qwen3:14b", "mistral-nemo"],
            index=0,
            key="local_model_select",
            help="Запускается локально на вашей GPU через Ollama (API-ключ не нужен). "
                 "Модель должна быть скачана: `ollama pull qwen3:14b`."
        )
        deepseek_key = ""
        st.info(f"🏠 Локальная обработка через Ollama: {analysis_model} (API-ключ не нужен)")
    else:
        analysis_model = st.selectbox(
            "🧠 Модель анализа",
            [
                "google/gemini-2.5-flash", 
                "deepseek/deepseek-chat", 
                "openai/gpt-4o", 
                "mistralai/mistral-nemo"
            ],
            index=0
        )
        deepseek_key = st.text_input("🔑 API ключ (для анализа)", type="password", help="Введите API ключ для DeepSeek/Gemini")
    transcription_method = st.radio(
        "🎤 Метод транскрибации",
        ["WhisperX + диаризация (локально)", "Yandex SpeechKit (API)"],
        index=0,
        key="transcription_method_select"
    )
    st.checkbox(
        "🧹 Предобработка звука (нормализация громкости)",
        value=True,
        key="preprocess_audio_checkbox",
        help="Безопасно: выравнивает громкость и режет суб-бас гул. НЕ вырезает речь. "
             "Работает для WhisperX. Сними галочку, чтобы сравнить до/после."
    )
    
    hf_token = ""
    yandex_api_key = ""
    aws_access_key_id = ""
    aws_secret_access_key = ""
    yandex_bucket = ""
    
    if "WhisperX" in transcription_method:
        st.info("🔒 Локальная обработка. Требуется HF Token.")
        hf_token = st.text_input("🔑 Hugging Face Token", type="password")
        if not hf_token:
            st.warning("⚠️ Введите HF Token для диаризации!")
        if not WHISPERX_AVAILABLE:
            st.error("❌ WhisperX не установлен!")
    else:
        st.info("🟡 Облако Яндекс (SpeechKit + Object Storage)")
        yandex_api_key = st.text_input(
            "🔑 API-ключ для SpeechKit (длинный)", 
            type="password", 
            help="Начинается с AQVN... (Берется в Сервисном аккаунте -> Создать новый ключ -> API-ключ)"
        )
        aws_access_key_id = st.text_input(
            "🔑 Идентификатор ключа (короткий)", 
            help="Берется в Сервисном аккаунте -> Создать новый ключ -> Статический ключ доступа"
        )
        aws_secret_access_key = st.text_input(
            "🔑 Секретный ключ (длинный)", 
            type="password", 
            help="Берется там же, выдается один раз вместе с Идентификатором"
        )
        yandex_bucket = st.text_input(
            "🪣 Имя бакета (Хранилище)", 
            value="", 
            help="Точное название, которое ты указал при создании бакета (например: my-okk-bucket)"
        )

    st.markdown("---")
    st.markdown("**📊 Статистика:**")
    st.markdown(f"📁 Загружено: **{len(st.session_state.uploaded_files)}**")
    st.markdown(f"✅ Успешно: **{len([r for r in st.session_state.processing_results if r['status'] == 'success'])}**")
    st.markdown(f"❌ Ошибок: **{len([r for r in st.session_state.processing_results if r['status'] == 'error'])}**")

# =============== ШАГ 1: ЗАГРУЗКА ===============
if st.session_state.current_step == 1:
    st.header("1. 📁 Загрузите аудиофайлы")
    uploaded_files = st.file_uploader("Перетащите файлы", type=['mp3', 'wav', 'm4a'], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        with st.expander(f"📋 Загружено: {len(uploaded_files)}", expanded=True):
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("🚀 Начать обработку", type="primary"):
            st.session_state.current_step = 2
            st.session_state.processing_results = []
            st.session_state.sheets_saved = None  # новый прогон — прошлая выгрузка не актуальна
            st.rerun()

# =============== ШАГ 2: ОБРАБОТКА ===============
elif st.session_state.current_step == 2:
    local_mode = st.session_state.get("local_mode_checkbox", False)
    st.header("2. 🔄 Обработка файлов")
    if not st.session_state.uploaded_files:
        st.warning("❌ Нет файлов")
        if st.button("← Назад"): st.session_state.current_step = 1; st.rerun()
    else:
        files = st.session_state.uploaded_files
        total_files = len(files)
        transcription_method = st.session_state.get("transcription_method_select", "WhisperX + диаризация (локально)")
        use_whisperx = "WhisperX" in transcription_method

        st.caption(f"{'🏠 Локальная' if local_mode else '☁️ Облачная'} модель анализа: **{analysis_model}** · "
                   f"🎤 {'WhisperX + диаризация' if use_whisperx else 'Yandex SpeechKit'} · файлов: **{total_files}**")
        # Экран прогона: что делается и с каким файлом, общий прогресс, счётчики.
        # Подробности (каждый шаг, ошибки с traceback, токены и цена) — в логе ниже.
        stage_box = st.empty()
        progress_bar = st.progress(0.0, text="Подготовка...")
        metrics_box = st.empty()
        notice_box = st.container()  # важные предупреждения (не прячем в лог)
        log_expander = st.expander("📜 Подробный лог прогона", expanded=False)

        run_log = RunLog()
        st.session_state.run_log_path = run_log.path
        st.session_state.run_notes = []
        log_expander.caption(f"Последние строки. Полный лог пишется в файл `{run_log.path}` — "
                             "скачать его можно на экране результатов.")
        log_view = log_expander.empty()

        batch_start = time.time()
        run = {"transcribed": 0, "analysed": 0, "errors": 0, "cost": 0.0,
               "phase": 1, "phase_name": "Транскрибация", "phase_total": total_files,
               "phase_done": 0, "phase_start": batch_start}

        def refresh_screen():
            """Перерисовывает прогресс, счётчики и хвост лога."""
            now = time.time()
            done, n = run["phase_done"], max(1, run["phase_total"])
            text = f"Фаза {run['phase']} из 2 · {run['phase_name']} · готово {done} из {run['phase_total']}"
            if done:
                left = (now - run["phase_start"]) / done * (run["phase_total"] - done)
                text += f" · до конца фазы ≈ {fmt_duration(left)}"
            progress_bar.progress(min(1.0, ((run["phase"] - 1) + done / n) / 2), text=text)
            cost_now = run["cost"] + st.session_state.get("_file_llm_cost", 0.0)
            with metrics_box.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("🎤 Транскрибировано", f"{run['transcribed']} / {total_files}")
                c2.metric("🧠 Проанализировано", f"{run['analysed']} / {total_files}")
                c3.metric("❌ Ошибок", run["errors"])
                c4.metric("⏱️ Прошло", fmt_duration(now - batch_start))
                c5.metric("💰 Стоимость ИИ", f"{cost_now:.2f} {COST_CURRENCY}")
            log_view.code("\n".join(run_log.tail().splitlines()[-LOG_VIEW_LINES:]), language=None)

        def show(idx, fname, op):
            """Текущая операция: крупно на экране + отметка в логе."""
            print(f"▶ [{idx}/{run['phase_total']}] {op}: {fname}")
            stage_box.markdown(f"**{op}** · файл {idx} из {run['phase_total']}  \n`{fname}`")
            refresh_screen()

        def start_phase(phase, name, n):
            run.update(phase=phase, phase_name=name, phase_total=n, phase_done=0, phase_start=time.time())
            print(f"===== ФАЗА {phase} из 2: {name} ({n} файлов) =====")

        def note(msg, level="warning"):
            """Предупреждение, которое должно быть видно сразу и на экране результатов."""
            print(msg)
            getattr(notice_box, level)(msg)
            st.session_state.run_notes.append((level, msg))

        def fail(fname, stage, err, elapsed, cost=0.0):
            """Фиксирует ошибку файла: коротко — в результаты, подробно — уже в логе."""
            lines = str(err).strip().splitlines()
            short = lines[0].lstrip("❌ ") if lines else type(err).__name__
            run["errors"] += 1
            print(f"❌ [{stage}] {fname}: {short}")
            st.session_state.processing_results.append({
                'filename': fname, 'status': 'error', 'stage': stage, 'error': short,
                'elapsed': elapsed, 'cost': cost,
            })

        with run_log:
            print(f"===== Прогон: {total_files} файлов · транскрибация: {transcription_method} · "
                  f"анализ: {analysis_model} ({'локально' if local_mode else 'облако'}) =====")
            # На 12 ГБ VRAM WhisperX large-v3 и Qwen-14B одновременно не помещаются, поэтому
            # обрабатываем в две фазы: сначала транскрибируем ВСЕ файлы (GPU занят только
            # WhisperX), затем выгружаем его и анализируем ВСЕ транскрипты (GPU занят только Qwen).
            if local_mode:
                stage_box.markdown("🔌 Проверяю локальный сервер Ollama...")
                if ensure_ollama_running():  # сам поднимет сервер без прокси, если он не запущен
                    unload_ollama_model(analysis_model)  # вдруг Qwen висит с прошлого прогона — освобождаем VRAM
                else:
                    note("⚠️ Не удалось поднять Ollama — локальный анализ (Фаза 2) может не сработать. "
                         "Проверь, что Ollama установлена.")

            transcribed = []  # сырые транскрипты переносим из Фазы 1 в Фазу 2
            gpu_fail_streak = 0
            gpu_broken = False

            # ============ ФАЗА 1: ТРАНСКРИБАЦИЯ (GPU = только WhisperX) ============
            start_phase(1, "Транскрибация", total_files)
            for i, uploaded_file in enumerate(files, 1):
                fname = uploaded_file.name
                file_start = time.time()
                if gpu_broken:
                    fail(fname, "Транскрибация", "Не обработан: видеокарта в сбойном состоянии — "
                         "перезапустите приложение и обработайте файл заново", 0.0)
                    run["phase_done"] += 1
                    continue

                # Временные файлы — удаляем в finally
                original_temp_path = None
                padded_temp_path = None
                cleaned_temp_path = None
                try:
                    show(i, fname, "📥 Подготовка аудио")
                    os.makedirs("recordings", exist_ok=True)
                    # Уникальное имя (спасет от ошибки Errno 13)
                    safe_name = f"{int(time.time())}_{fname}"
                    original_temp_path = os.path.join("recordings", safe_name)
                    with open(original_temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    time.sleep(0.5)

                    if use_whisperx:
                        if not hf_token: raise Exception("Нет HF Token")
                        if not WHISPERX_AVAILABLE: raise Exception("WhisperX не установлен")
                        # Предобработка + лечим обрезку начала (2 с тишины)
                        src_for_pad = original_temp_path
                        if st.session_state.get("preprocess_audio_checkbox", True):
                            cleaned = preprocess_audio(original_temp_path)
                            if cleaned != original_temp_path:
                                cleaned_temp_path = cleaned
                                src_for_pad = cleaned
                        padded_temp_path = pad_audio_simple_silence(src_for_pad)

                        show(i, fname, "🎤 Транскрибация и разделение на говорящих (WhisperX)")
                        transcript_text = transcribe_with_whisperx_diarization(padded_temp_path, hf_token)
                    else:
                        if not all([yandex_api_key, aws_access_key_id, aws_secret_access_key, yandex_bucket]):
                            raise Exception("Заполните все ключи Яндекса в боковой панели!")
                        show(i, fname, "🎤 Распознавание (Yandex SpeechKit)")
                        transcript_text = transcribe_with_yandex(
                            original_temp_path, yandex_api_key, aws_access_key_id,
                            aws_secret_access_key, yandex_bucket)

                    if "❌" in transcript_text:
                        raise Exception(transcript_text)

                    t_phase1 = time.time() - file_start
                    transcribed.append({'uploaded_file': uploaded_file, 'transcript': transcript_text,
                                        't_phase1': t_phase1})
                    run["transcribed"] += 1
                    gpu_fail_streak = 0
                    print(f"✅ [{i}/{total_files}] {fname}: транскрибация {fmt_duration(t_phase1)}, "
                          f"{len(transcript_text)} симв.")
                except Exception as e:
                    if "❌" not in str(e):  # у ошибок транскрибации traceback уже в логе
                        print(traceback.format_exc())
                    fail(fname, "Транскрибация", e, time.time() - file_start)
                    if use_whisperx and is_cuda_error(e):
                        gpu_fail_streak += 1
                        if gpu_fail_streak >= GPU_FAIL_LIMIT:
                            gpu_broken = True
                            note(f"🛑 {GPU_FAIL_LIMIT} файла подряд упали с ошибкой видеокарты даже после "
                                 "перезагрузки моделей — GPU в сбойном состоянии. Остальные файлы пропущены, "
                                 "уже расшифрованные будут проанализированы. Перезапустите приложение и "
                                 "загрузите пропущенные файлы заново.", level="error")
                finally:
                    # Временные аудио больше не нужны — Фаза 2 работает только с текстом
                    for _p in (original_temp_path, padded_temp_path, cleaned_temp_path):
                        try:
                            if _p and os.path.exists(_p):
                                os.remove(_p)
                        except Exception as cleanup_err:
                            print(f"⚠️ Не удалось удалить временный файл: {cleanup_err}")
                    run["phase_done"] += 1
            refresh_screen()

            # === СМЕНА МОДЕЛИ НА GPU: выгружаем WhisperX, освобождаем VRAM под Qwen ===
            stage_box.markdown("🔄 Освобождаю видеопамять перед анализом...")
            unload_whisperx_from_vram()
            phase1_time = time.time() - batch_start

            # ============ ФАЗА 2: КОРРЕКЦИЯ + АНАЛИЗ (GPU = только Qwen / облако) ============
            phase2_start = time.time()
            start_phase(2, "Анализ", len(transcribed))
            for j, item in enumerate(transcribed, 1):
                uploaded_file = item['uploaded_file']
                fname = uploaded_file.name
                transcript_text = item['transcript']
                file_start = time.time()
                st.session_state["_file_llm_cost"] = 0.0  # копим стоимость LLM за файл
                stage = "Коррекция текста"
                try:
                    show(j, fname, "✏️ Коррекция текста")
                    transcript_text = smart_text_correction(transcript_text, analysis_model, deepseek_key,
                                                            local_mode, label=fname)
                    stage = "Коррекция ролей"
                    show(j, fname, "🎭 Коррекция ролей говорящих")
                    transcript_text = correct_speaker_roles(transcript_text, analysis_model, deepseek_key,
                                                            local_mode, label=fname)

                    stage = "Анализ"
                    show(j, fname, "🧠 Анализ звонка")
                    similar_calls = find_similar_calls(transcript_text, top_k=3)
                    examples_text = ""
                    if similar_calls:
                        examples_text = "\n\nПРИМЕРЫ ПОХОЖИХ ЗВОНКОВ ИЗ БАЗЫ:\n"
                        for idx, call in enumerate(similar_calls, 1):
                            examples_text += f"\nПример {idx} (уверенность {call.get('confidence', 0):.0f}%):\n"
                            examples_text += f"Тема: {call['analysis'].get('topic', '—')}\n"
                            examples_text += f"Оценка: {json.dumps(call['analysis'], ensure_ascii=False)[:500]}\n"

                    result_text = llm_chat(build_analysis_messages(transcript_text, examples_text),
                                           analysis_model, deepseek_key, local_mode,
                                           temperature=0.3, max_tokens=2500, label=f"анализ: {fname}")
                    # ДИАГНОСТИКА: что реально вернул ИИ (длина + превью)
                    print(f"🔎 [анализ {fname}] ответ ИИ: {len(result_text)} симв. | "
                          f"превью: {result_text[:300]!r}")
                    analysis_result = parse_analysis_response(result_text)

                    confidence = calculate_confidence(analysis_result, transcript_text)
                    print(f"🧠 Уверенность модели: {confidence:.0f}%")
                    if confidence >= 95:
                        save_to_knowledge_base(transcript_text, analysis_result, confidence, auto_save=True, filename=fname)
                    else:
                        print(f"⚠️ Уверенность {confidence:.0f}% < 95% — не сохраняем автоматически")
                    stats = get_database_stats()
                    print(f"📚 База знаний: {stats['total']} звонков (средняя уверенность: {stats['avg_confidence']:.0f}%)")

                    normalize_analysis(analysis_result)

                    base_name = fname[:-4] if fname.lower().endswith(('.mp3', '.wav', '.m4a')) else fname
                    parts = base_name.split('-')
                    call_date = parts[1] if len(parts) > 1 else datetime.now().strftime("%d.%m.%Y")
                    formatted_name = ' '.join([p.strip().capitalize() for p in parts[3:] if p.strip()]) if len(parts) >= 4 else "Неизвестный"

                    # Время файла = транскрибация (Фаза 1) + анализ (Фаза 2)
                    t_phase2 = time.time() - file_start
                    file_cost = st.session_state.get("_file_llm_cost", 0.0)
                    st.session_state.processing_results.append({
                        'filename': fname, 'base_name': base_name, 'call_date': call_date,
                        'operator_name': formatted_name, 'transcript': transcript_text,
                        'analysis': analysis_result, 'status': 'success',
                        'cost': file_cost, 't_phase1': item['t_phase1'], 't_phase2': t_phase2,
                        'elapsed': item['t_phase1'] + t_phase2,
                    })
                    run["analysed"] += 1
                    print(f"✅ [{j}/{run['phase_total']}] {fname}: анализ {fmt_duration(t_phase2)}, "
                          f"{file_cost:.2f} {COST_CURRENCY}")
                except Exception as e:
                    print(traceback.format_exc())
                    fail(fname, stage, e, item['t_phase1'] + (time.time() - file_start),
                         cost=st.session_state.get("_file_llm_cost", 0.0))
                finally:
                    # Деньги за вызовы, сделанные до ошибки, тоже потрачены — учитываем
                    run["cost"] += st.session_state.get("_file_llm_cost", 0.0)
                    st.session_state["_file_llm_cost"] = 0.0
                    run["phase_done"] += 1

            phase2_time = time.time() - phase2_start
            total_elapsed = time.time() - batch_start
            print(f"===== Готово: успешно {run['analysed']} из {total_files}, ошибок {run['errors']}, "
                  f"время {fmt_duration(total_elapsed)} (транскрибация {fmt_duration(phase1_time)}, "
                  f"анализ {fmt_duration(phase2_time)}), стоимость ИИ {run['cost']:.2f} {COST_CURRENCY} =====")

        st.session_state.total_elapsed = total_elapsed
        st.session_state.phase_times = {"transcribe": phase1_time, "analyze": phase2_time}
        st.session_state.current_step = 3
        st.rerun()

elif st.session_state.current_step == 3:
    st.header("3. 📊 Результаты обработки")
    
    if os.path.exists("recordings") and os.listdir("recordings"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Удалить все файлы из папки recordings", type="secondary"):
                try:
                    files_count = 0
                    for filename in os.listdir("recordings"):
                        file_path = os.path.join("recordings", filename)
                        os.remove(file_path)
                        files_count += 1
                    st.success(f"✅ Удалено {files_count} файлов из папки recordings")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка удаления: {e}")
        
        with col2:
            with st.expander("📁 Файлы в папке recordings", expanded=False):
                files = os.listdir("recordings")
                st.write(f"Всего файлов: {len(files)}")
                for f in files:
                    size = os.path.getsize(os.path.join("recordings", f))
                    st.write(f"- {f} ({size:,} байт)")
    
    successful = [r for r in st.session_state.processing_results if r['status'] == 'success']
    failed = [r for r in st.session_state.processing_results if r['status'] == 'error']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Всего файлов", len(st.session_state.processing_results))
    with col2:
        st.metric("Успешно", len(successful))
    with col3:
        st.metric("Ошибок", len(failed))
    with col4:
        _tot = st.session_state.get("total_elapsed", 0)
        _n = max(1, len(st.session_state.processing_results))
        _ph = st.session_state.get("phase_times")
        st.metric("⏱️ Время обработки", fmt_duration(_tot), f"~{fmt_duration(_tot/_n)}/звонок",
                  delta_color="off")
    with col5:
        # Итог — по всем файлам: вызовы ИИ до ошибки тоже оплачены
        _cost_tot = sum(r.get('cost', 0.0) for r in st.session_state.processing_results)
        _cost_ok = sum(r.get('cost', 0.0) for r in successful)
        _cost_err = _cost_tot - _cost_ok
        st.metric("💰 Стоимость ИИ", f"{_cost_tot:.2f} {COST_CURRENCY}",
                  f"~{_cost_ok/max(1, len(successful)):.2f} {COST_CURRENCY}/звонок",
                  delta_color="off",
                  help=(f"Из них {_cost_err:.2f} {COST_CURRENCY} — вызовы по файлам, "
                        "анализ которых завершился ошибкой") if _cost_err > 0.005 else None)
    if _ph:
        st.caption(f"⏱️ Транскрибация: {fmt_duration(_ph['transcribe'])} · "
                   f"анализ: {fmt_duration(_ph['analyze'])}")

    for _level, _msg in st.session_state.get("run_notes", []):
        getattr(st, _level)(_msg)

    if successful:
        def safe_int(val, default=0):
            try:
                if isinstance(val, int):
                    return val
                elif isinstance(val, str):
                    return 1 if val.lower().strip() in ["1", "да", "yes"] else 0
                return int(val) if val else default
            except:
                return default
                
        with st.expander("✅ Успешно обработанные файлы", expanded=True):
            for result in successful:
                analysis = result['analysis']
                binary_fields = ["establishing_contact", "client_type", "clarifying_questions",
                                "knowledge_quality", "software_proficiency", "politeness"]
                
                total_score = sum(safe_int(analysis.get(k, 0)) for k in binary_fields)
                need_score = block_score_1_5(analysis, NEED_KEYS)
                obj_score = block_score_1_5(analysis, OBJ_KEYS)
                dozhim_score = block_score_1_5(analysis, DOZHIM_KEYS)
                contact_score = block_score_1_5(analysis, CONTACT_KEYS)
                nextstep_score = block_score_1_5(analysis, NEXTSTEP_KEYS)
                speech_score = block_score_1_5(analysis, SPEECH_KEYS)
                had_obj = safe_int(analysis.get("had_objections", 1)) == 1
                is_tech_issue = safe_int(analysis.get("technical_issue", 0)) == 1
                score_display = "Н/О (Брак связи)" if is_tech_issue else f"{total_score}/6"
                need_display = "Н/О" if is_tech_issue else f"{need_score}/5"
                obj_display = "Н/О" if is_tech_issue else ("Н/У" if not had_obj else f"{obj_score}/5")
                dozhim_display = "Н/О" if is_tech_issue else f"{dozhim_score}/5"
                contact_display = "Н/О" if is_tech_issue else f"{contact_score}/5"
                nextstep_display = "Н/О" if is_tech_issue else f"{nextstep_score}/5"
                speech_display = "Н/О" if is_tech_issue else f"{speech_score}/{block_max(SPEECH_KEYS)}"
                grand = total_score + need_score + dozhim_score + contact_score + nextstep_score + speech_score + (obj_score if had_obj else 0)
                grand_display = "Н/О (Брак связи)" if is_tech_issue else f"{grand}/{grand_max(had_obj)}"
                
                call_type = analysis.get('call_type', 'Первичный')
                type_badge = "🔄 Повторный" if call_type == "Повторный" else "🆕 Первичный"
                
                st.write(f"📁 **{result['filename']}**")
                st.write(f"  📅 Дата: {result['call_date']}")
                st.write(f"  👤 Оператор: {result['operator_name']}")
                st.write(f"  📞 Тип звонка: {type_badge}")
                st.write(f"  🎯 Тема: {result['analysis'].get('topic', '—')}")
                _nom_str = result['analysis'].get('nomenclature_str', '')
                _vol_str = result['analysis'].get('volume_str', '')
                st.write(f"  📦 Номенклатура: {_nom_str if _nom_str else '—'}")
                st.write(f"  📊 Объём: {_vol_str if _vol_str else '—'}")
                st.write(f"  🏆 Общий балл: {grand_display}")
                st.write(f"  ⭐ Базовый: {score_display}")
                st.write(f"  🔎 Потребность: {need_display}")
                st.write(f"  🛡️ Возражения: {obj_display}")
                st.write(f"  🎯 Дожим: {dozhim_display}")
                st.write(f"  📇 Кл/счёт (контакты): {contact_display}")
                st.write(f"  ➡️ Следующий шаг: {nextstep_display}")
                st.write(f"  🗣️ Речь: {speech_display}")
                if 'elapsed' in result:
                    st.write(f"  ⏱️ Время: {fmt_duration(result['elapsed'])} "
                             f"(транскрибация {fmt_duration(result.get('t_phase1'))}, "
                             f"анализ {fmt_duration(result.get('t_phase2'))})")
                _fcost = result.get('cost', 0.0)
                if _fcost:
                    st.write(f"  💰 Стоимость ИИ: {_fcost:.2f} {COST_CURRENCY}")
                st.write("---")

    if failed:
        with st.expander(f"❌ Файлы с ошибками ({len(failed)})", expanded=False):
            st.caption("Подробности по каждой ошибке — в логе прогона ниже.")
            for result in failed:
                st.write(f"📁 {result['filename']}")
                _stage = result.get('stage')
                st.error(f"{_stage}: {result['error']}" if _stage else f"Ошибка: {result['error']}")

    _log_path = st.session_state.get("run_log_path")
    if _log_path and os.path.exists(_log_path):
        with open(_log_path, encoding="utf-8") as _lf:
            _log_text = _lf.read()
        with st.expander("📜 Лог прогона", expanded=False):
            st.code(_log_text, language=None, height=400)
        st.download_button("⬇️ Скачать лог", data=_log_text.encode("utf-8"),
                           file_name=os.path.basename(_log_path), mime="text/plain",
                           on_click="ignore")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Обработать ещё", type="secondary"):
            st.session_state.current_step = 1
            st.session_state.sheets_saved = None
            st.rerun()

    with col2:
        if st.button("📋 Просмотреть детали", type="secondary"):
            st.session_state.current_step = 4
            st.rerun()

    with col3:
        # После успешной выгрузки кнопка гаснет — повторный клик задвоил бы строки в таблице
        if successful and st.button("💾 Сохранить в Google Sheets", type="primary",
                                    disabled=bool(st.session_state.sheets_saved)):
            with st.spinner("Сохраняю в таблицу..."):
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        CREDENTIALS_FILE,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    service = build('sheets', 'v4', credentials=credentials)

                    # Последняя занятая строка — по нескольким столбцам, а не только по A:
                    # если в последних строках A пустой, а остальное заполнено, запись по
                    # одному A легла бы поверх данных. Столбец D (транскрипт до 99к символов)
                    # не читаем — он тяжёлый, а строка без D всё равно видна по соседним.
                    # API отдаёт строки до последней непустой, поэтому len() = её номер.
                    check = service.spreadsheets().values().batchGet(
                        spreadsheetId=SPREADSHEET_ID,
                        ranges=[f"{SHEET_NAME}!A:C", f"{SHEET_NAME}!E:AA"]
                    ).execute()
                    last_row = max((len(vr.get('values', [])) for vr in check.get('valueRanges', [])),
                                   default=0)
                    first_empty_row = max(last_row + 1, 2)  # строка 1 — шапка
                    
                    all_rows_data = []
                    now = datetime.now()
                    upload_date = f"{now.day}.{now.month}.{now.year}"
                    
                    for i, result in enumerate(successful):
                        analysis = result['analysis']
                        
                        establishing_contact = analysis.get("establishing_contact", 0)
                        client_type = analysis.get("client_type", 0)
                        clarifying_questions = analysis.get("clarifying_questions", 0)
                        knowledge_quality = analysis.get("knowledge_quality", 0)
                        software_proficiency = analysis.get("software_proficiency", 0)
                        politeness = analysis.get("politeness", 0)

                        total_score = int(establishing_contact) + int(client_type) + int(clarifying_questions) + int(knowledge_quality) + int(software_proficiency) + int(politeness)
                        
                        is_tech_issue = str(analysis.get("technical_issue", "0")).strip() == "1"

                        # Блоки 1-5 по числу действий; Возражения = Н/У, если возражений не было
                        had_obj = int(analysis.get("had_objections", 1) or 0) == 1
                        need_block = "Брак связи" if is_tech_issue else block_score_1_5(analysis, NEED_KEYS)
                        if is_tech_issue:
                            obj_block = "Брак связи"
                        elif not had_obj:
                            obj_block = "Н/У"
                        else:
                            obj_block = block_score_1_5(analysis, OBJ_KEYS)
                        dozhim_block = "Брак связи" if is_tech_issue else block_score_1_5(analysis, DOZHIM_KEYS)
                        contact_block = "Брак связи" if is_tech_issue else block_score_1_5(analysis, CONTACT_KEYS)
                        nextstep_block = "Брак связи" if is_tech_issue else block_score_1_5(analysis, NEXTSTEP_KEYS)
                        speech_block = "Брак связи" if is_tech_issue else block_score_1_5(analysis, SPEECH_KEYS)

                        # Итоговый балл = базовые (0-6) + все блоки. Максимум 6+5+5+5+5+5+2 = 33
                        if is_tech_issue:
                            grand_total = "Брак связи"
                        else:
                            grand_total = (total_score + need_block + dozhim_block + contact_block
                                           + nextstep_block + speech_block
                                           + (obj_block if isinstance(obj_block, int) else 0))

                        row_data = [
                            result['call_date'],
                            result['operator_name'],
                            result['base_name'],
                            result['transcript'][:99000],
                            upload_date,
                            analysis.get("topic", ""),
                            analysis.get("client_request", ""),
                            analysis.get("nomenclature_str", ""),
                            analysis.get("volume_str", ""),
                            analysis.get("solution", ""),
                            analysis.get("urgency", ""),
                            analysis.get("client_mood", ""),
                            analysis.get("manager_actions", ""),
                            establishing_contact,
                            client_type,
                            clarifying_questions,
                            knowledge_quality,
                            software_proficiency,
                            politeness,
                            need_block,
                            obj_block,
                            dozhim_block,
                            contact_block,
                            nextstep_block,
                            speech_block,
                            grand_total,
                            analysis.get("recommendations", "")
                        ]
                        all_rows_data.append(row_data)
                    
                    start_row = first_empty_row
                    end_row = first_empty_row + len(successful) - 1
                    range_to_write = f"{SHEET_NAME}!A{start_row}:AA{end_row}"

                    # Гарантируем, что в листе достаточно строк (иначе update упадёт
                    # с "exceeds grid limits"). При нехватке — докидываем строки с запасом.
                    sheet_id, current_rows = None, 0
                    try:
                        meta = service.spreadsheets().get(
                            spreadsheetId=SPREADSHEET_ID,
                            fields="sheets(properties(sheetId,title,gridProperties/rowCount))"
                        ).execute()
                        for sh in meta.get('sheets', []):
                            if sh['properties']['title'] == SHEET_TITLE:
                                sheet_id = sh['properties']['sheetId']
                                current_rows = sh['properties']['gridProperties']['rowCount']
                                break
                        if sheet_id is not None and end_row > current_rows:
                            service.spreadsheets().batchUpdate(
                                spreadsheetId=SPREADSHEET_ID,
                                body={"requests": [{"appendDimension": {
                                    "sheetId": sheet_id, "dimension": "ROWS",
                                    "length": end_row - current_rows + 500
                                }}]}
                            ).execute()
                    except Exception as _grid_err:
                        print(f"⚠️ Не удалось расширить сетку листа: {_grid_err}")

                    body = {'values': all_rows_data}
                    
                    service.spreadsheets().values().update(
                        spreadsheetId=SPREADSHEET_ID,
                        range=range_to_write,
                        valueInputOption='RAW',
                        body=body
                    ).execute()

                    # Итог кладём в session_state и перерисовываем страницу: сообщение и
                    # кнопка «Начать новую сессию» живут ВНЕ обработчика этой кнопки.
                    # (Раньше вложенная кнопка не срабатывала никогда: её клик перезапускает
                    # скрипт, внешняя кнопка в новом прогоне уже False, и до неё не доходит.)
                    st.session_state.sheets_saved = {
                        "count": len(successful), "start_row": start_row, "end_row": end_row,
                        "gid": sheet_id if sheet_id is not None else 0,
                        "fresh": True,  # шарики показать один раз
                    }
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Ошибка сохранения: {str(e)}")

    saved = st.session_state.sheets_saved
    if saved:
        st.success(f"✅ Сохранено {saved['count']} записей в строки {saved['start_row']}-{saved['end_row']}!")
        if saved.get("fresh"):
            st.balloons()
            saved["fresh"] = False
        st.markdown(f"""
        **📎 Ссылка на таблицу:**
        https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={saved['gid']}&range=A{saved['start_row']}
        """)
        if st.button("🔄 Начать новую сессию"):
            st.session_state.current_step = 1
            st.session_state.uploaded_files = []
            st.session_state.processing_results = []
            st.session_state.sheets_saved = None
            st.rerun()

elif st.session_state.current_step == 4:
    st.header("4. 📋 Детальный просмотр")
    successful = [r for r in st.session_state.processing_results if r['status'] == 'success']
    if not successful:
        st.warning("Нет результатов")
    else:
        file_options = [f"{r['filename']} ({r['analysis'].get('topic', '—')})" for r in successful]
        selected_file = st.selectbox("Выберите файл", file_options)
        result = successful[file_options.index(selected_file)]
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Инфо")
            st.write(f"**Файл:** {result['filename']}")
            st.write(f"**Дата:** {result['call_date']}")
            st.write(f"**Оператор:** {result['operator_name']}")
            call_type = result['analysis'].get('call_type', 'Первичный')
            st.write(f"**Тип звонка:** {'🔄 ' if call_type == 'Повторный' else '🆕 '}{call_type}")
        with col2:
            st.subheader("🎯 Анализ")
            st.write(f"**Тема:** {result['analysis'].get('topic', '—')}")
            st.write(f"**Настроение:** {result['analysis'].get('client_mood', '—')}")
            st.info(f"💡 **Рекомендация ИИ:** {result['analysis'].get('recommendations', 'Нет рекомендаций')}")

        with st.expander("📝 Транскрипция"):
            st.text_area("Текст", result['transcript'], height=300)
        
        with st.expander("⭐ Оценка качества"):
            binary_criteria = [
                ("Установление контакта", "establishing_contact"),
                ("Определение физ/юр лица", "client_type"),
                ("Уточняющие вопросы", "clarifying_questions"),
                ("Качество консультации", "knowledge_quality"),
                ("Работа в программах", "software_proficiency"),
                ("Вежливость общения", "politeness"),
            ]
            total_score = 0
            for label, key in binary_criteria:
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                total_score += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            
            is_tech_issue = str(result['analysis'].get("technical_issue", "0")).strip() == "1"

            if is_tech_issue:
                st.markdown("### 🏆 Итоговая оценка: **Н/О (Брак связи)**")
                st.warning("⚠️ Оценка не учитывается в статистике менеджера из-за технических проблем со связью.")
            else:
                st.markdown(f"### 🏆 Итоговая оценка: **{total_score}/6**")
                st.progress(total_score / 6)

                if total_score >= 4:
                    st.success("Отличное качество! 🎉")
                elif total_score >= 2:
                    st.warning("Средний результат.")
                else:
                    st.error("Требуется обучение.")

            st.markdown("---")
            st.markdown("#### 🔎 Блок «Потребность» (выявление ситуации клиента)")
            need_count = 0
            for key, label in NEED_CRITERIA:  # в scoring.py кортежи (ключ, подпись)
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                need_count += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            need_score = max(1, min(need_count, 5))
            if is_tech_issue:
                st.markdown("### 📊 Потребность: **Н/О (Брак связи)**")
            else:
                st.markdown(f"### 📊 Потребность: **{need_score}/5**  _(действий: {need_count})_")
                st.progress(need_score / 5)

            st.markdown("---")
            st.markdown("#### 🛡️ Блок «Возражения» (отработка возражений клиента)")
            obj_count = 0
            for key, label in OBJECTION_CRITERIA:
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                obj_count += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            obj_score = max(1, min(obj_count, 5))
            had_obj = int(result['analysis'].get("had_objections", 1) or 0) == 1
            if is_tech_issue:
                st.markdown("### 📊 Возражения: **Н/О (Брак связи)**")
            elif not had_obj:
                st.markdown("### 📊 Возражения: **Н/У** _(возражений не было)_")
            else:
                st.markdown(f"### 📊 Возражения: **{obj_score}/5**  _(действий: {obj_count})_")
                st.progress(obj_score / 5)

            st.markdown("---")
            st.markdown("#### 🎯 Блок «Дожим» (закрытие сделки)")
            dozhim_count = 0
            for key, label in DOZHIM_CRITERIA:
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                dozhim_count += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            dozhim_score = max(1, min(dozhim_count, 5))
            if is_tech_issue:
                st.markdown("### 📊 Дожим: **Н/О (Брак связи)**")
            else:
                st.markdown(f"### 📊 Дожим: **{dozhim_score}/5**  _(действий: {dozhim_count})_")
                st.progress(dozhim_score / 5)

            st.markdown("---")
            st.markdown("#### 📇 Блок «Кл/счёт, контактные данные»")
            contact_count = 0
            for key, label in CONTACT_CRITERIA:
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                contact_count += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            contact_score = max(1, min(contact_count, 5))
            if is_tech_issue:
                st.markdown("### 📊 Кл/счёт: **Н/О (Брак связи)**")
            else:
                st.markdown(f"### 📊 Кл/счёт: **{contact_score}/5**  _(действий: {contact_count})_")
                st.progress(contact_score / 5)

            st.markdown("---")
            st.markdown("#### ➡️ Блок «Следующий шаг / завершение»")
            nextstep_count = 0
            for key, label in NEXTSTEP_CRITERIA:
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                nextstep_count += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            nextstep_score = max(1, min(nextstep_count, 5))
            if is_tech_issue:
                st.markdown("### 📊 Следующий шаг: **Н/О (Брак связи)**")
            else:
                st.markdown(f"### 📊 Следующий шаг: **{nextstep_score}/5**  _(действий: {nextstep_count})_")
                st.progress(nextstep_score / 5)

            st.markdown("---")
            st.markdown("#### 🗣️ Блок «Речь»")
            speech_count = 0
            for key, label in SPEECH_CRITERIA:
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                speech_count += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            speech_score = max(1, min(speech_count, 5))
            speech_max = block_max(SPEECH_KEYS)  # критериев 2 -> максимум 2, а не 5
            if is_tech_issue:
                st.markdown("### 📊 Речь: **Н/О (Брак связи)**")
            else:
                st.markdown(f"### 📊 Речь: **{speech_score}/{speech_max}**  _(действий: {speech_count})_")
                st.progress(speech_score / speech_max)

            st.markdown("---")
            if is_tech_issue:
                st.markdown("## 🏆 Общий балл: **Н/О (Брак связи)**")
            else:
                grand = total_score + need_score + dozhim_score + contact_score + nextstep_score + speech_score + (obj_score if had_obj else 0)
                st.markdown(f"## 🏆 Общий балл: **{grand} / {grand_max(had_obj)}**")

        if st.button("← Назад"): st.session_state.current_step = 3; st.rerun()
