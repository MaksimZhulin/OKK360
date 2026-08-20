import os
import sys
import json
import time
from datetime import datetime

# Распознавание номенклатуры из звонка по каталогу (data/catalog.json)
from nomenclature import match_mentions, format_nomenclature

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
from audio import (
    WHISPERX_AVAILABLE, preprocess_audio, pad_audio_simple_silence,
    transcribe_with_yandex, transcribe_with_whisperx_diarization,
)
from llm_analysis import (
    COST_CURRENCY, add_llm_cost, smart_text_correction, correct_speaker_roles,
)

# Критерии оценки и подсчёт баллов вынесены в scoring.py
from scoring import (
    NEED_CRITERIA, NEED_KEYS, OBJECTION_CRITERIA, OBJ_KEYS,
    DOZHIM_CRITERIA, DOZHIM_KEYS, CONTACT_CRITERIA, CONTACT_KEYS,
    NEXTSTEP_CRITERIA, NEXTSTEP_KEYS, SPEECH_CRITERIA, SPEECH_KEYS,
    block_score_1_5, build_recommendation,
)


# =============== НАСТРОЙКИ STREAMLIT ===============
st.set_page_config(page_title="Массовый анализ звонков", page_icon="🎤", layout="wide")
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

st.title("🎤 МАССОВЫЙ АНАЛИЗ ТЕЛЕФОННЫХ ЗВОНКОВ")
st.markdown("---")

# =============== БОКОВАЯ ПАНЕЛЬ ===============
with st.sidebar:
    st.header("⚙️ Настройки")
    
    local_mode = st.checkbox("🖥️ Локальная модель (Ollama Mistral NeMo)", value=False, key="local_mode_checkbox")
    
    if local_mode:
        analysis_model = "mistral-nemo"
        deepseek_key = ""  
        st.info("🏠 Локальная обработка через Ollama (не требует API ключа)")
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
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Очистить", type="secondary"):
                st.session_state.uploaded_files = []
                st.rerun()
        with col2:
            if st.button("🚀 Начать обработку", type="primary"):
                st.session_state.current_step = 2
                st.session_state.processing_results = []
                st.rerun()

# =============== ШАГ 2: ОБРАБОТКА ===============
elif st.session_state.current_step == 2:
    local_mode = st.session_state.get("local_mode_checkbox", False)
    st.header("2. 🔄 Обработка файлов")
    if not st.session_state.uploaded_files:
        st.warning("❌ Нет файлов")
        if st.button("← Назад"): st.session_state.current_step = 1; st.rerun()
    else:
        total_files = len(st.session_state.uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        from openai import OpenAI
        
        if local_mode:
            client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            st.write("🏠 Используем локальную модель: mistral-nemo")
        else:
            client = OpenAI(api_key=deepseek_key, base_url="https://litellm.tokengate.ru/v1")
            st.write(f"☁️ Используем облачную модель: {analysis_model}")

        batch_start = time.time()
        for i, uploaded_file in enumerate(st.session_state.uploaded_files):
            file_start = time.time()
            elapsed_so_far = time.time() - batch_start
            status_text.text(f"⏱️ Обработка {i+1}/{total_files}: {uploaded_file.name} · прошло {elapsed_so_far:.0f} сек")
            
            # --- ВАЖНО: Запоминаем пути, чтобы удалить их в конце ---
            original_temp_path = None
            padded_temp_path = None
            cleaned_temp_path = None

            try:
                st.session_state["_file_llm_cost"] = 0.0  # копим стоимость LLM за файл
                os.makedirs("recordings", exist_ok=True)
                
                # Уникальное имя (спасет от ошибки Errno 13)
                safe_name = f"{int(time.time())}_{uploaded_file.name}"
                original_temp_path = os.path.join("recordings", safe_name)
                
                with open(original_temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                time.sleep(0.5)
                
                temp_filename = original_temp_path
                # --- ПРЕДОБРАБОТКА + ЛЕЧИМ ОБРЕЗКУ НАЧАЛА (только WhisperX) ---
                if "WhisperX" in transcription_method:
                    src_for_pad = original_temp_path
                    if st.session_state.get("preprocess_audio_checkbox", True):
                        cleaned = preprocess_audio(original_temp_path)
                        if cleaned != original_temp_path:
                            cleaned_temp_path = cleaned  # временный, удалим в finally
                            src_for_pad = cleaned
                    temp_filename = pad_audio_simple_silence(src_for_pad)
                    padded_temp_path = temp_filename # Запоминаем созданный дубль
                # ----------------------------
                
                st.write("🎤 Транскрибация...")
                transcript_text = ""
                
                transcription_method = st.session_state.get("transcription_method_select", "WhisperX + диаризация (локально)")
                
                if "WhisperX" in transcription_method:
                    if not hf_token: raise Exception("Нет HF Token")
                    if not WHISPERX_AVAILABLE: raise Exception("WhisperX не установлен")
                    
                    st.write("🔧 Используем WhisperX + PyAnnote (диаризация)")
                    transcript_text = transcribe_with_whisperx_diarization(temp_filename, hf_token)
                    
                    if "❌" in transcript_text:
                        raise Exception(transcript_text)
                    
                    st.write("🔍 Коррекция текста...")
                    transcript_text = smart_text_correction(transcript_text, analysis_model, deepseek_key, local_mode)
                    st.write(f"✅ Коррекция завершена! Символов: {len(transcript_text)}")
                    
                    st.write("🔍 Коррекция ролей спикеров...")
                    transcript_text = correct_speaker_roles(transcript_text, analysis_model, deepseek_key, local_mode)
                    st.write(f"✅ Коррекция ролей завершена!")

                else:
                    if not all([yandex_api_key, aws_access_key_id, aws_secret_access_key, yandex_bucket]):
                        raise Exception("Заполните все ключи Яндекса в боковой панели!")
                    
                    st.write("⚡ Используем Yandex SpeechKit")
                    transcript_text = transcribe_with_yandex(
                        temp_filename, 
                        yandex_api_key, 
                        aws_access_key_id, 
                        aws_secret_access_key, 
                        yandex_bucket
                    )
                
                    if "❌" in transcript_text:
                        raise Exception(transcript_text)
                
                    st.write("🔍 Коррекция терминов (Сальметро -> СтальМетУрал)...")
                    transcript_text = smart_text_correction(transcript_text, analysis_model, deepseek_key, local_mode)
                    
                    st.write("🔍 Коррекция ролей спикеров (LLM)...")
                    transcript_text = correct_speaker_roles(transcript_text, analysis_model, deepseek_key, local_mode)
                    st.write(f"✅ Коррекция завершена! Символов: {len(transcript_text)}")

                similar_calls = find_similar_calls(transcript_text, top_k=3)
                
                examples_text = ""
                if similar_calls:
                    examples_text = "\n\nПРИМЕРЫ ПОХОЖИХ ЗВОНКОВ ИЗ БАЗЫ:\n"
                    for idx, call in enumerate(similar_calls, 1):
                        examples_text += f"\nПример {idx} (уверенность {call.get('confidence', 0):.0f}%):\n"
                        examples_text += f"Тема: {call['analysis'].get('topic', '—')}\n"
                        examples_text += f"Оценка: {json.dumps(call['analysis'], ensure_ascii=False)[:500]}\n"

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
9a. "nomenclature_raw": Массив строк — ВСЕ товары/номенклатура, которые упоминались в звонке (что интересует клиента или обсуждается в заказе). Выпиши КАЖДУЮ упомянутую позицию ОТДЕЛЬНЫМ элементом массива. Пиши позицию дословно как звучит, вместе с размером/маркой если названы. Пример: ["двутавр 20", "арматура 8 мм А500С", "лист 3мм оцинкованный"]. ВАЖНО: если назван только код/модель товара (например "МС-140", "МС500", "А500С"), обязательно добавь тип товара из контекста разговора (например "чугунная батарея МС-140", "радиатор МС500", "арматура А500С"). Не выдумывай то, чего нет в тексте. Если товары не обсуждались — верни пустой массив [].

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

Верни ТОЛЬКО JSON, без Markdown-разметки и без пояснений:"""

                response = client.chat.completions.create(model=analysis_model, messages=[
                    {"role": "system", "content": "Ты — опытный аналитик колл-центра, специализирующийся на глубоком анализе транскрипций звонков. Твоя цель — предоставить всестороннюю, объективную и профессиональную оценку взаимодействия между клиентом и агентом, выявить ключевые паттерны, проблемы и предложить конкретные, действенные рекомендации. Отвечай строго в формате JSON."},
                    {"role": "user", "content": prompt}
                ], temperature=0.3, max_tokens=2500)

                if not local_mode:
                    add_llm_cost(analysis_model, response)
                result_text = response.choices[0].message.content.strip()
                json_start = result_text.find('{')
                json_end = result_text.rfind('}')
                
                if json_start != -1 and json_end != -1:
                    json_str = result_text[json_start:json_end + 1]
                    try:
                        analysis_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        print(f"⚠️ Ошибка парсинга JSON от ИИ. Сырой текст: {result_text}")
                        analysis_result = {}
                else:
                    print(f"⚠️ ИИ не вернул фигурные скобки. Сырой текст: {result_text}")
                    analysis_result = {}
                
                confidence = calculate_confidence(analysis_result, transcript_text)
                st.write(f"🧠 Уверенность модели: {confidence:.0f}%")
                
                if confidence >= 95:
                    save_to_knowledge_base(transcript_text, analysis_result, confidence, auto_save=True, filename=uploaded_file.name)
                else:
                    st.write(f"⚠️ Уверенность {confidence:.0f}% < 95% — не сохраняем автоматически")

                stats = get_database_stats()
                st.info(f"📚 База знаний: {stats['total']} звонков (средняя уверенность: {stats['avg_confidence']:.0f}%)")
                
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

                # Номенклатура: сырые упоминания от ИИ -> каноничные позиции каталога
                try:
                    _raw_noms = analysis_result.get("nomenclature_raw", []) or []
                    if isinstance(_raw_noms, str):
                        _raw_noms = [_raw_noms]
                    _nom_items = match_mentions(_raw_noms)
                    analysis_result["nomenclature"] = _nom_items
                    analysis_result["nomenclature_str"] = format_nomenclature(_nom_items)
                except Exception as _nom_err:
                    print(f"⚠️ Матчинг номенклатуры: {_nom_err}")
                    analysis_result["nomenclature"] = []
                    analysis_result["nomenclature_str"] = ""

                filename = uploaded_file.name
                base_name = filename[:-4] if filename.lower().endswith(('.mp3', '.wav', '.m4a')) else filename
                parts = base_name.split('-')
                call_date = parts[1] if len(parts) > 1 else datetime.now().strftime("%d.%m.%Y")
                formatted_name = ' '.join([p.strip().capitalize() for p in parts[3:] if p.strip()]) if len(parts) >= 4 else "Неизвестный"
                
                st.session_state.processing_results.append({
                    'filename': filename, 'base_name': base_name, 'call_date': call_date,
                    'operator_name': formatted_name, 'transcript': transcript_text,
                    'analysis': analysis_result, 'status': 'success',
                    'cost': st.session_state.get("_file_llm_cost", 0.0)
                })
            except Exception as e:
                st.session_state.processing_results.append({'filename': uploaded_file.name, 'status': 'error', 'error': str(e)})
            finally:
                # === ОЧИСТКА МУСОРА ПОСЛЕ ОБРАБОТКИ ===
                # Удаляем и оригинал, и файл с тишиной, чтобы они не копились
                try:
                    if original_temp_path and os.path.exists(original_temp_path):
                        os.remove(original_temp_path)
                    if padded_temp_path and os.path.exists(padded_temp_path):
                        os.remove(padded_temp_path)
                    if cleaned_temp_path and os.path.exists(cleaned_temp_path):
                        os.remove(cleaned_temp_path)
                except Exception as cleanup_err:
                    print(f"⚠️ Не удалось удалить временный файл: {cleanup_err}")

            file_elapsed = time.time() - file_start
            if st.session_state.processing_results:
                st.session_state.processing_results[-1]['elapsed'] = file_elapsed
            total_elapsed = time.time() - batch_start
            status_text.text(f"✅ {i+1}/{total_files} · {file_elapsed:.0f} сек на файл · всего {total_elapsed:.0f} сек")
            progress_bar.progress((i + 1) / total_files)
            time.sleep(1)

        st.session_state.total_elapsed = time.time() - batch_start
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
        _m, _s = divmod(int(_tot), 60)
        _tot_str = f"{_m} мин {_s} сек" if _m else f"{_s} сек"
        st.metric("⏱️ Время обработки", _tot_str, f"~{_tot/_n:.0f} сек/звонок")
    with col5:
        _cost_tot = sum(r.get('cost', 0.0) for r in successful)
        _cn = max(1, len(successful))
        st.metric("💰 Стоимость ИИ", f"{_cost_tot:.2f} {COST_CURRENCY}",
                  f"~{_cost_tot/_cn:.2f} {COST_CURRENCY}/звонок")
    
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
                speech_display = "Н/О" if is_tech_issue else f"{speech_score}/5"
                grand = total_score + need_score + dozhim_score + contact_score + nextstep_score + speech_score + (obj_score if had_obj else 0)
                grand_display = "Н/О (Брак связи)" if is_tech_issue else f"{grand}/33"
                
                call_type = analysis.get('call_type', 'Первичный')
                type_badge = "🔄 Повторный" if call_type == "Повторный" else "🆕 Первичный"
                
                st.write(f"📁 **{result['filename']}**")
                st.write(f"  📅 Дата: {result['call_date']}")
                st.write(f"  👤 Оператор: {result['operator_name']}")
                st.write(f"  📞 Тип звонка: {type_badge}")
                st.write(f"  🎯 Тема: {result['analysis'].get('topic', '—')}")
                _nom_str = result['analysis'].get('nomenclature_str', '')
                st.write(f"  📦 Номенклатура: {_nom_str if _nom_str else '—'}")
                st.write(f"  🏆 Общий балл: {grand_display}")
                st.write(f"  ⭐ Базовый: {score_display}")
                st.write(f"  🔎 Потребность: {need_display}")
                st.write(f"  🛡️ Возражения: {obj_display}")
                st.write(f"  🎯 Дожим: {dozhim_display}")
                st.write(f"  📇 Кл/счёт (контакты): {contact_display}")
                st.write(f"  ➡️ Следующий шаг: {nextstep_display}")
                st.write(f"  🗣️ Речь: {speech_display}")
                _fcost = result.get('cost', 0.0)
                if _fcost:
                    st.write(f"  💰 Стоимость ИИ: {_fcost:.2f} {COST_CURRENCY}")
                st.write("---")
    
    if failed:
        with st.expander("❌ Файлы с ошибками", expanded=False):
            for result in failed:
                st.write(f"📁 {result['filename']}")
                st.error(f"Ошибка: {result['error']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Обработать ещё", type="secondary"):
            st.session_state.current_step = 1
            st.rerun()
    
    with col2:
        if st.button("📋 Просмотреть детали", type="secondary"):
            st.session_state.current_step = 4
            st.rerun()
    
    with col3:
        if successful and st.button("💾 Сохранить в Google Sheets", type="primary"):
            with st.spinner("Сохраняю в таблицу..."):
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        'credentials.json',
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    service = build('sheets', 'v4', credentials=credentials)
                    
                    SPREADSHEET_ID = "1Oe-dKF_0oPhCdlwcj6jeco7BSIBi37jPuO3rSG4C930"
                    
                    range_to_check = "'Выгрузка из проекта'!A:A"
                    result_sheets = service.spreadsheets().values().get(
                        spreadsheetId=SPREADSHEET_ID,
                        range=range_to_check
                    ).execute()
                    
                    values = result_sheets.get('values', [])
                    first_empty_row = len(values) + 1
                    if first_empty_row == 1:
                        first_empty_row = 2
                    
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
                            analysis.get("nomenclature_str", ""),
                            analysis.get("recommendations", "")
                        ]
                        all_rows_data.append(row_data)
                    
                    start_row = first_empty_row
                    end_row = first_empty_row + len(successful) - 1
                    range_to_write = f"'Выгрузка из проекта'!A{start_row}:Z{end_row}"

                    # Гарантируем, что в листе достаточно строк (иначе update упадёт
                    # с "exceeds grid limits"). При нехватке — докидываем строки с запасом.
                    try:
                        meta = service.spreadsheets().get(
                            spreadsheetId=SPREADSHEET_ID,
                            fields="sheets(properties(sheetId,title,gridProperties/rowCount))"
                        ).execute()
                        sheet_id, current_rows = None, 0
                        for sh in meta.get('sheets', []):
                            if sh['properties']['title'] == 'Выгрузка из проекта':
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
                    
                    result_update = service.spreadsheets().values().update(
                        spreadsheetId=SPREADSHEET_ID,
                        range=range_to_write,
                        valueInputOption='RAW',
                        body=body
                    ).execute()
                    
                    st.success(f"✅ Сохранено {len(successful)} записей в строки {start_row}-{end_row}!")
                    st.balloons()
                    
                    st.markdown(f"""
                    **📎 Ссылка на таблицу:**
                    https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid=0&range=A{start_row}
                    """)
                    
                    if st.button("🔄 Начать новую сессию"):
                        st.session_state.current_step = 1
                        st.session_state.uploaded_files = []
                        st.session_state.processing_results = []
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Ошибка сохранения: {str(e)}")

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
            for label, key in NEED_CRITERIA:
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
            for label, key in OBJECTION_CRITERIA:
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
            for label, key in DOZHIM_CRITERIA:
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
            for label, key in CONTACT_CRITERIA:
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
            for label, key in NEXTSTEP_CRITERIA:
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
            for label, key in SPEECH_CRITERIA:
                value = result['analysis'].get(key, 0)
                score = 1 if value == 1 else 0
                speech_count += score
                icon = "✅" if score == 1 else "❌"
                st.markdown(f"{icon} **{label}:** {'Да' if score == 1 else 'Нет'}")
            speech_score = max(1, min(speech_count, 5))
            if is_tech_issue:
                st.markdown("### 📊 Речь: **Н/О (Брак связи)**")
            else:
                st.markdown(f"### 📊 Речь: **{speech_score}/5**  _(действий: {speech_count})_")
                st.progress(speech_score / 5)

            st.markdown("---")
            if is_tech_issue:
                st.markdown("## 🏆 Общий балл: **Н/О (Брак связи)**")
            else:
                grand = total_score + need_score + dozhim_score + contact_score + nextstep_score + speech_score + (obj_score if had_obj else 0)
                st.markdown(f"## 🏆 Общий балл: **{grand} / 33**")

        if st.button("← Назад"): st.session_state.current_step = 3; st.rerun()
