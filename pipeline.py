# -*- coding: utf-8 -*-
"""
Пайплайн обработки пачки звонков: транскрибация -> коррекция -> анализ -> оценка.
Работает в фоновом потоке (Job), независимо от скрипта Streamlit: страница только
показывает прогресс и отправляет команды пауза / продолжить / стоп.
"""
import os
import sys
import json
import time
import types
import threading
import traceback
from datetime import datetime

import timing_stats
from nomenclature import match_mentions, format_nomenclature, format_volumes
from auto_learning import save_to_knowledge_base, calculate_confidence, find_similar_calls, get_database_stats
from audio import (
    WHISPERX_AVAILABLE, preprocess_audio, pad_audio_simple_silence,
    transcribe_with_yandex, transcribe_with_whisperx_diarization,
    unload_whisperx_from_vram, is_cuda_error, audio_duration, warmup_whisperx,
)
from llm_analysis import (
    COST_CURRENCY, llm_chat, LLMCancelled, smart_text_correction, correct_speaker_roles,
    unload_ollama_model, ensure_ollama_running, reset_file_cost, get_file_cost,
    check_model_access,
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
    NEED_KEYS, OBJ_KEYS, DOZHIM_KEYS, CONTACT_KEYS, NEXTSTEP_KEYS, SPEECH_KEYS,
    build_recommendation,
)


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


# =====================================================================
#                      ФОНОВЫЙ ПРОГОН ОБРАБОТКИ
# =====================================================================
# Обработка идёт в отдельном потоке, а не в скрипте Streamlit. Раньше любой клик
# в интерфейсе (или обновление страницы) перезапускал скрипт и обрывал прогон
# посреди операции. Теперь страница только показывает состояние и шлёт команды:
# пауза / продолжить / стоп. Команды выполняются в безопасных точках МЕЖДУ
# операциями (checkpoint) — начатая операция всегда доводится до конца.

TEMP_DIR = os.path.join("temp", "processing")  # временные аудио прогона


class Cancelled(BaseException):
    """Пользователь нажал «Стоп». BaseException — чтобы обработчики ошибок файла
    (except Exception) не приняли остановку за ошибку и не продолжили прогон."""


class PreflightFailed(Exception):
    """Проверка перед стартом не прошла (ключ, доступ к модели, шлюз) — прогон
    не начинаем, чтобы не тратить время на транскрибацию впустую."""


class Job:
    """Один прогон пачки файлов. В процессе одновременно живёт только один
    (видеокарта одна) — см. start_job()/get_job()."""

    def __init__(self, files, cfg):
        self.files = list(files)   # объекты с .name и .getvalue()
        self.cfg = dict(cfg)
        self.total = len(self.files)
        self.log = RunLog()
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.started = time.time()
        self.finished = None
        self.outcome = None         # done | cancelled | failed
        self.error = None
        self.claimed = False        # результаты уже забрала страница

        self.paused_total = 0.0     # сколько секунд простояли на паузе
        self.pause_started = None   # не None — стоим на паузе прямо сейчас

        self.phase = 1
        self.phase_name = "Транскрибация"
        self.phase_total = self.total
        self.phase_done = 0
        self.phase_start = self.started
        self.stage = "Подготовка..."
        self.stage_file = ""
        self.stage_idx = 0

        self.transcribed_n = 0
        self.analysed_n = 0
        self.errors_n = 0
        self.cost_total = 0.0
        self.results = []
        self.notes = []             # [(level, текст)] — важное, показываем на экране
        self.phase_times = {}

        # Прогноз времени: длительность каждого аудио + модели скорости по замерам
        # этого компьютера для выбранных настроек (timing_stats).
        self.durations = [audio_duration(f.getvalue(), f.name)[0] for f in self.files]
        self._t_rate = timing_stats.Rate(timing_stats.transcribe_key(self.cfg))
        self._a_rate = timing_stats.Rate(timing_stats.analyze_key(self.cfg))
        self._load_left = (timing_stats.Rate("load|whisperx").predict(0)
                           if "WhisperX" in self.cfg["transcription_method"] else 0.0)
        self._p1 = []               # [(прогноз, факт)] по готовым файлам фазы 1
        self._p2 = []               # то же для фазы 2
        self._queue2 = []           # длительности файлов, ждущих анализа
        self._file_start = None     # clock() начала текущего файла
        self._file_pred = 0.0       # прогноз текущего файла

        self._thread = threading.Thread(target=self._run, name="okk-job", daemon=True)

    # ---------------- управление (вызывает страница) ----------------
    def pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def stop(self):
        self._stop.set()
        self._pause.clear()  # чтобы поток вышел из ожидания паузы

    @property
    def status(self):
        """running | pausing | paused | stopping | done | cancelled | failed"""
        if self.outcome:
            return self.outcome
        if self._stop.is_set():
            return "stopping"
        if self._pause.is_set():
            # пауза запрошена; «paused» — когда поток уже встал в безопасной точке
            return "paused" if self.pause_started is not None else "pausing"
        return "running"

    @property
    def active(self):
        return self.outcome is None

    def clock(self):
        """Время без пауз — для замеров длительности файлов и фаз."""
        paused = self.paused_total
        if self.pause_started is not None:
            paused += time.time() - self.pause_started
        return time.time() - paused

    def elapsed(self):
        end = self.finished if self.finished else time.time()
        paused = self.paused_total
        if self.pause_started is not None:
            paused += time.time() - self.pause_started
        return end - self.started - paused

    @staticmethod
    def _speed(pairs):
        """Во сколько раз этот прогон идёт медленнее/быстрее прогноза (по 2+ файлам).
        Ограничиваем 0.1–10×: защита от абсурда, но и стартовые значения, ошибшиеся
        в разы (новые настройки без замеров), быстро подстраиваются под факт."""
        if len(pairs) < 2:
            return 1.0
        pred = sum(p for p, _ in pairs)
        fact = sum(f for _, f in pairs)
        return min(10.0, max(0.1, fact / pred)) if pred > 0 else 1.0

    def eta(self):
        """Сколько секунд осталось до конца ВСЕГО прогона: оставшиеся файлы × модель
        скорости (по замерам) × поправка на фактическую скорость этого прогона."""
        if not self.active:
            return 0.0
        done = self.phase_done
        in_file = self._file_start is not None
        if self.phase == 1:
            k = self._speed(self._p1)
            # текущий файл — остаток его прогноза; дальше — файлы, которые ещё не начаты
            cur_left = max(0.0, self._file_pred * k - (self.clock() - self._file_start)) if in_file else 0.0
            pending = self.durations[done + 1:] if in_file else self.durations[done:]
            left1 = self._load_left + cur_left + sum(self._t_rate.predict(d) for d in pending) * k
            # в анализ пойдут уже расшифрованные + текущий + ещё не начатые
            left2 = sum(self._a_rate.predict(d) for d in self._queue2 + self.durations[done:])
            return left1 + left2
        k = self._speed(self._p2)
        cur_left = max(0.0, self._file_pred * k - (self.clock() - self._file_start)) if in_file else 0.0
        pending = self._queue2[done + 1:] if in_file else self._queue2[done:]
        return cur_left + sum(self._a_rate.predict(d) for d in pending) * k

    def cost_now(self):
        return self.cost_total + (get_file_cost() if self.active else 0.0)

    # ---------------- внутреннее ----------------
    def checkpoint(self):
        """Безопасная точка между операциями: здесь срабатывают «Стоп» и «Пауза»."""
        if self._stop.is_set():
            raise Cancelled()
        if self._pause.is_set():
            self.pause_started = time.time()
            print("⏸️ Пауза — текущая операция завершена, жду «Продолжить»")
            while self._pause.is_set() and not self._stop.is_set():
                time.sleep(0.3)
            dt = time.time() - self.pause_started
            self.paused_total += dt
            self.pause_started = None
            if self._stop.is_set():
                raise Cancelled()
            print(f"▶️ Продолжаем (пауза длилась {fmt_duration(dt)})")

    def _start_phase(self, phase, name, n):
        self.phase, self.phase_name, self.phase_total = phase, name, n
        self.phase_done = 0
        self.phase_start = self.clock()
        print(f"===== ФАЗА {phase} из 2: {name} ({n} файлов) =====")

    def _stage(self, idx, fname, op):
        """Перед каждой операцией: точка паузы/стопа + отметка для экрана и лога."""
        self.checkpoint()
        self.stage, self.stage_file, self.stage_idx = op, fname, idx
        print(f"▶ [{idx}/{self.phase_total}] {op}: {fname}")

    def _note(self, msg, level="warning"):
        print(msg)
        self.notes.append((level, msg))

    def _fail(self, fname, stage, err, elapsed, cost=0.0):
        lines = str(err).strip().splitlines()
        short = lines[0].lstrip("❌ ") if lines else type(err).__name__
        self.errors_n += 1
        print(f"❌ [{stage}] {fname}: {short}")
        self.results.append({'filename': fname, 'status': 'error', 'stage': stage,
                             'error': short, 'elapsed': elapsed, 'cost': cost})

    def _run(self):
        with self.log:
            try:
                self._process()
                self.outcome = "done"
            except PreflightFailed as e:
                self.error = str(e)
                self._note(f"🛑 Прогон не начат: {e}", level="error")
                self.outcome = "failed"
            except Cancelled:
                print("⏹️ Прогон остановлен пользователем — результаты сброшены")
                self._release_gpu()
                self.outcome = "cancelled"
            except Exception as e:
                print(traceback.format_exc())
                self.error = f"{type(e).__name__}: {e}"
                self._note(f"🛑 Прогон прерван внутренней ошибкой: {self.error}. "
                           "Ниже — то, что успело обработаться; подробности в логе.", level="error")
                self._release_gpu()
                self.outcome = "failed"
            finally:
                self.finished = time.time()

    def _release_gpu(self):
        try:
            unload_whisperx_from_vram()
            if self.cfg["local_mode"]:
                unload_ollama_model(self.cfg["analysis_model"])
        except Exception as e:
            print(f"⚠️ Освобождение GPU: {e}")

    def _process(self):
        cfg = self.cfg
        local_mode = cfg["local_mode"]
        analysis_model = cfg["analysis_model"]
        api_key = cfg["api_key"]
        use_whisperx = "WhisperX" in cfg["transcription_method"]
        files, total_files = self.files, self.total

        print(f"===== Прогон: {total_files} файлов · транскрибация: {cfg['transcription_method']} · "
              f"анализ: {analysis_model} ({'локально' if local_mode else 'облако'}) =====")
        print(f"Аудио всего: {fmt_duration(sum(self.durations))} · прогноз на старте: "
              f"{fmt_duration(self.eta())} (замеров: транскрибация {self._t_rate.n}, анализ {self._a_rate.n})")
        # На 12 ГБ VRAM WhisperX large-v3 и Qwen-14B одновременно не помещаются, поэтому
        # обрабатываем в две фазы: сначала транскрибируем ВСЕ файлы (GPU занят только
        # WhisperX), затем выгружаем его и анализируем ВСЕ транскрипты (GPU занят только Qwen).
        if local_mode:
            self.stage = "🔌 Проверяю локальный сервер Ollama..."
            if ensure_ollama_running():  # сам поднимет сервер без прокси, если он не запущен
                unload_ollama_model(analysis_model)  # вдруг Qwen висит с прошлого прогона — освобождаем VRAM
            else:
                self._note("⚠️ Не удалось поднять Ollama — локальный анализ (Фаза 2) может не сработать. "
                           "Проверь, что Ollama установлена.")

        # Облачная модель: проверяем ключ и доступ к модели ДО транскрибации
        if not local_mode:
            self.stage, self.stage_file = "🔑 Проверяю доступ к модели анализа...", ""
            problem = check_model_access(analysis_model, api_key)
            reset_file_cost()
            if problem:
                raise PreflightFailed(problem)
            print(f"✅ Модель {analysis_model} доступна")

        transcribed = []  # сырые транскрипты переносим из Фазы 1 в Фазу 2
        gpu_fail_streak = 0
        gpu_broken = False
        os.makedirs(TEMP_DIR, exist_ok=True)

        # ============ ФАЗА 1: ТРАНСКРИБАЦИЯ (GPU = только WhisperX) ============
        self._start_phase(1, "Транскрибация", total_files)
        # Модели WhisperX грузим отдельным шагом: загрузка меряется сама по себе,
        # а время первого файла не раздувается (точнее и прогноз, и замеры).
        if use_whisperx and cfg["hf_token"] and WHISPERX_AVAILABLE:
            self.checkpoint()
            self.stage, self.stage_file = "⏳ Загрузка моделей распознавания (WhisperX)...", ""
            t0 = self.clock()
            try:
                warmup_whisperx(cfg["hf_token"])
                t_load = self.clock() - t0
                print(f"⏱️ Модели WhisperX готовы за {fmt_duration(t_load)}")
                if t_load > 3:  # модели действительно грузились, а не были уже в памяти
                    timing_stats.record("load|whisperx", 1.0, t_load)
            except Exception as e:
                print(f"⚠️ Предзагрузка WhisperX не удалась (повторится на первом файле): {e}")
        self._load_left = 0.0
        for i, uploaded_file in enumerate(files, 1):
            fname = uploaded_file.name
            file_start = self.clock()
            if gpu_broken:
                self._fail(fname, "Транскрибация", "Не обработан: видеокарта в сбойном состоянии — "
                           "перезапустите приложение и обработайте файл заново", 0.0)
                self.phase_done += 1
                continue

            dur = self.durations[i - 1]
            self._file_start, self._file_pred = file_start, self._t_rate.predict(dur)

            # Временные файлы — удаляем в finally (и при «Стопе» тоже)
            original_temp_path = None
            padded_temp_path = None
            cleaned_temp_path = None
            try:
                self._stage(i, fname, "📥 Подготовка аудио")
                # Уникальное имя (спасет от ошибки Errno 13)
                safe_name = f"{int(time.time())}_{fname}"
                original_temp_path = os.path.join(TEMP_DIR, safe_name)
                with open(original_temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                time.sleep(0.5)

                if use_whisperx:
                    if not cfg["hf_token"]: raise Exception("Нет HF Token")
                    if not WHISPERX_AVAILABLE: raise Exception("WhisperX не установлен")
                    # Предобработка + лечим обрезку начала (2 с тишины)
                    src_for_pad = original_temp_path
                    if cfg["preprocess"]:
                        cleaned = preprocess_audio(original_temp_path)
                        if cleaned != original_temp_path:
                            cleaned_temp_path = cleaned
                            src_for_pad = cleaned
                    padded_temp_path = pad_audio_simple_silence(src_for_pad)

                    self._stage(i, fname, "🎤 Транскрибация и разделение на говорящих (WhisperX)")
                    transcript_text = transcribe_with_whisperx_diarization(padded_temp_path, cfg["hf_token"])
                else:
                    y = cfg["yandex"]
                    if not all([y["api_key"], y["access_key_id"], y["secret_access_key"], y["bucket"]]):
                        raise Exception("Заполните все ключи Яндекса в боковой панели!")
                    self._stage(i, fname, "🎤 Распознавание (Yandex SpeechKit)")
                    transcript_text = transcribe_with_yandex(
                        original_temp_path, y["api_key"], y["access_key_id"],
                        y["secret_access_key"], y["bucket"])

                if "❌" in transcript_text:
                    raise Exception(transcript_text)

                t_phase1 = self.clock() - file_start
                transcribed.append({'uploaded_file': uploaded_file, 'transcript': transcript_text,
                                    't_phase1': t_phase1, 'duration': dur})
                self._queue2.append(dur)
                self._p1.append((self._file_pred, t_phase1))
                timing_stats.record(timing_stats.transcribe_key(cfg), dur, t_phase1)
                self.transcribed_n += 1
                gpu_fail_streak = 0
                print(f"✅ [{i}/{total_files}] {fname}: транскрибация {fmt_duration(t_phase1)} "
                      f"(аудио {fmt_duration(dur)}, прогноз был {fmt_duration(self._file_pred)}), "
                      f"{len(transcript_text)} симв.")
            except Exception as e:
                if "❌" not in str(e):  # у ошибок транскрибации traceback уже в логе
                    print(traceback.format_exc())
                self._fail(fname, "Транскрибация", e, self.clock() - file_start)
                if use_whisperx and is_cuda_error(e):
                    gpu_fail_streak += 1
                    if gpu_fail_streak >= GPU_FAIL_LIMIT:
                        gpu_broken = True
                        self._note(f"🛑 {GPU_FAIL_LIMIT} файла подряд упали с ошибкой видеокарты даже после "
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
                self._file_start = None
                self.phase_done += 1

        # === СМЕНА МОДЕЛИ НА GPU: выгружаем WhisperX, освобождаем VRAM под Qwen ===
        self.checkpoint()
        self.stage, self.stage_file = "🔄 Освобождаю видеопамять перед анализом...", ""
        unload_whisperx_from_vram()
        phase1_time = self.clock() - self.started

        # ============ ФАЗА 2: КОРРЕКЦИЯ + АНАЛИЗ (GPU = только Qwen / облако) ============
        phase2_start = self.clock()
        self._queue2 = [it['duration'] for it in transcribed]
        self._start_phase(2, "Анализ", len(transcribed))
        for j, item in enumerate(transcribed, 1):
            uploaded_file = item['uploaded_file']
            fname = uploaded_file.name
            transcript_text = item['transcript']
            file_start = self.clock()
            dur = item['duration']
            self._file_start, self._file_pred = file_start, self._a_rate.predict(dur)
            reset_file_cost()  # копим стоимость LLM за файл
            stage = "Коррекция текста"
            try:
                self._stage(j, fname, "✏️ Коррекция текста")
                transcript_text = smart_text_correction(transcript_text, analysis_model, api_key,
                                                        local_mode, label=fname, cancel=self._stop)
                stage = "Коррекция ролей"
                self._stage(j, fname, "🎭 Коррекция ролей говорящих")
                transcript_text = correct_speaker_roles(transcript_text, analysis_model, api_key,
                                                        local_mode, label=fname, cancel=self._stop)

                stage = "Анализ"
                self._stage(j, fname, "🧠 Анализ звонка")
                similar_calls = find_similar_calls(transcript_text, top_k=3)
                examples_text = ""
                if similar_calls:
                    examples_text = "\n\nПРИМЕРЫ ПОХОЖИХ ЗВОНКОВ ИЗ БАЗЫ:\n"
                    for idx, call in enumerate(similar_calls, 1):
                        examples_text += f"\nПример {idx} (уверенность {call.get('confidence', 0):.0f}%):\n"
                        examples_text += f"Тема: {call['analysis'].get('topic', '—')}\n"
                        examples_text += f"Оценка: {json.dumps(call['analysis'], ensure_ascii=False)[:500]}\n"

                result_text = llm_chat(build_analysis_messages(transcript_text, examples_text),
                                       analysis_model, api_key, local_mode,
                                       temperature=0.3, max_tokens=2500, label=f"анализ: {fname}",
                                       cancel=self._stop)
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

                # Время файла = транскрибация (Фаза 1) + анализ (Фаза 2), без пауз
                t_phase2 = self.clock() - file_start
                file_cost = get_file_cost()
                self.results.append({
                    'filename': fname, 'base_name': base_name, 'call_date': call_date,
                    'operator_name': formatted_name, 'transcript': transcript_text,
                    'analysis': analysis_result, 'status': 'success',
                    'cost': file_cost, 't_phase1': item['t_phase1'], 't_phase2': t_phase2,
                    'elapsed': item['t_phase1'] + t_phase2,
                })
                self.analysed_n += 1
                self._p2.append((self._file_pred, t_phase2))
                timing_stats.record(timing_stats.analyze_key(cfg), dur, t_phase2)
                print(f"✅ [{j}/{self.phase_total}] {fname}: анализ {fmt_duration(t_phase2)} "
                      f"(прогноз был {fmt_duration(self._file_pred)}), {file_cost:.2f} {COST_CURRENCY}")
            except LLMCancelled:
                raise Cancelled()
            except Exception as e:
                print(traceback.format_exc())
                self._fail(fname, stage, e, item['t_phase1'] + (self.clock() - file_start),
                           cost=get_file_cost())
            finally:
                # Деньги за вызовы, сделанные до ошибки/остановки, тоже потрачены — учитываем
                self.cost_total += get_file_cost()
                reset_file_cost()
                self._file_start = None
                self.phase_done += 1

        phase2_time = self.clock() - phase2_start
        self.phase_times = {"transcribe": phase1_time, "analyze": phase2_time}
        print(f"===== Готово: успешно {self.analysed_n} из {total_files}, ошибок {self.errors_n}, "
              f"время {fmt_duration(self.elapsed())} (транскрибация {fmt_duration(phase1_time)}, "
              f"анализ {fmt_duration(phase2_time)}), стоимость ИИ {self.cost_total:.2f} {COST_CURRENCY} =====")


# Реестр прогона — в отдельном «модуле» без файла: Streamlit при изменении кода
# перезагружает pipeline.py, и обычная глобальная переменная обнулилась бы, а фоновый
# поток остался бы без хозяина (и следующий старт запустил бы второй прогон на GPU).
_REG = sys.modules.setdefault("_okk_job_registry", types.ModuleType("_okk_job_registry"))
if not hasattr(_REG, "lock"):
    _REG.lock = threading.Lock()
    _REG.job = None


def get_job():
    """Текущий (или последний) прогон процесса — переживает перезагрузку страницы."""
    return _REG.job


def start_job(files, cfg):
    """Запускает новый прогон. Если предыдущий ещё идёт — возвращает его (второй
    параллельный прогон на одной видеокарте только уронил бы оба)."""
    with _REG.lock:
        if _REG.job is not None and _REG.job.active:
            return _REG.job
        _REG.job = Job(files, cfg)
        _REG.job._thread.start()
        return _REG.job
