import os
import sys
import time
from datetime import datetime, timedelta

# Windows: консоль по умолчанию cp1251 и падает на эмодзи в print() — принудительно UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from config import SPREADSHEET_ID, CREDENTIALS_FILE, SHEET_NAME, SHEET_TITLE
import timing_stats
from audio import WHISPERX_AVAILABLE, audio_duration
from llm_analysis import COST_CURRENCY, api_key_problem
# Сама обработка (транскрибация, анализ, пауза/стоп) — в pipeline.py, в фоновом потоке
from pipeline import fmt_duration, start_job, get_job, TEMP_DIR
# Критерии для детального просмотра и подсчёта баллов
from scoring import (
    NEED_CRITERIA, NEED_KEYS, OBJECTION_CRITERIA, OBJ_KEYS,
    DOZHIM_CRITERIA, DOZHIM_KEYS, CONTACT_CRITERIA, CONTACT_KEYS,
    NEXTSTEP_CRITERIA, NEXTSTEP_KEYS, SPEECH_CRITERIA, SPEECH_KEYS,
    block_score_1_5, block_max, grand_max,
)


# Окно лога на экране прогона: сколько последних строк показывать
LOG_VIEW_LINES = 15


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


SHEET_CELL_LIMIT = 50000  # предел символов в ячейке Google Sheets


def sheet_cell_text(text):
    """Режет текст под ячейку Sheets. Раньше резали до 99 000 — длинный разговор
    превышал предел в 50 000, и API отклонял ВСЮ выгрузку пачки."""
    text = text or ""
    if len(text) <= SHEET_CELL_LIMIT:
        return text
    suffix = "\n… [транскрипт обрезан: не помещается в ячейку]"
    return text[:SHEET_CELL_LIMIT - len(suffix)] + suffix


def claim_job(job):
    """Забирает итоги закончившегося прогона в сессию (ровно один раз)."""
    job.claimed = True
    st.session_state.confirm_stop = False
    st.session_state.run_log_path = job.log.path
    if job.outcome == "cancelled":
        st.session_state.processing_results = []
        st.session_state.uploaded_files = []
        st.session_state.current_step = 1
        st.session_state.flash = "⏹️ Анализ остановлен и сброшен. Загрузите файлы и запустите обработку заново."
    else:  # done или failed (во втором случае — то, что успело обработаться)
        st.session_state.processing_results = list(job.results)
        st.session_state.total_elapsed = job.elapsed()
        st.session_state.phase_times = job.phase_times or None
        st.session_state.run_notes = list(job.notes)
        st.session_state.current_step = 3


@st.dialog("Сбросить текущий анализ?", dismissible=False)
def confirm_stop_dialog(job):
    st.write("Вы уверены, что хотите сбросить текущий анализ? "
             "Результаты уже обработанных файлов не сохранятся.")
    st.caption("Если сейчас выполняется операция, она сначала доработает до конца — "
               "это может занять до нескольких минут.")
    c1, c2 = st.columns(2)
    if c1.button("Да, сбросить", type="primary", width="stretch"):
        job.stop()
        st.session_state.confirm_stop = False
        st.rerun()
    if c2.button("Нет, продолжить", width="stretch"):
        st.session_state.confirm_stop = False
        st.rerun()


# Прогон живёт в фоне и переживает обновление страницы: если он идёт (или закончился,
# а итоги ещё никто не забрал) — сразу показываем экран обработки.
_job = get_job()
if _job is not None and not _job.claimed and st.session_state.current_step != 2:
    st.session_state.current_step = 2

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
                # названия — с префиксом поставщика, как в примерах API AgentPlatform
                "google/gemini-2.5-flash",
                "google/gemini-2.5-flash-lite",
                "deepseek/deepseek-chat",
                "openai/gpt-4o",
            ],
            index=0
        )
        deepseek_key = st.text_input("🔑 API ключ (для анализа)", type="password",
                                     help="Ключ AgentPlatform: Доступ к моделям → Ключи")
        if deepseek_key and api_key_problem(deepseek_key):
            st.warning(api_key_problem(deepseek_key))
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

def current_run_cfg():
    """Настройки прогона из боковой панели — для старта и для прогноза времени."""
    return {
        "local_mode": st.session_state.get("local_mode_checkbox", False),
        "analysis_model": analysis_model,
        "api_key": deepseek_key,
        "transcription_method": st.session_state.get("transcription_method_select",
                                                     "WhisperX + диаризация (локально)"),
        "hf_token": hf_token,
        "preprocess": st.session_state.get("preprocess_audio_checkbox", True),
        "yandex": {"api_key": yandex_api_key, "access_key_id": aws_access_key_id,
                   "secret_access_key": aws_secret_access_key, "bucket": yandex_bucket},
    }


def show_time_estimate(durations, approx_n, cfg):
    """Прогноз времени всего прогона до старта — пересчитывается при смене настроек."""
    est = timing_stats.estimate(durations, cfg)
    finish = datetime.now() + timedelta(seconds=est["total"])
    day = "" if finish.date() == datetime.now().date() else f" {finish:%d.%m}"
    whisperx = "WhisperX" in cfg["transcription_method"]
    parts = []
    if est["load"]:
        parts.append(f"загрузка {fmt_duration(est['load'])}")
    parts.append(f"транскрибация {fmt_duration(est['transcribe'])}")
    parts.append(f"анализ {fmt_duration(est['analyze'])}")
    missing = [name for name, n in (("транскрибации", est["n_transcribe"]),
                                    ("анализа", est["n_analyze"])) if not n]
    # Подробности — во всплывающей подсказке, чтобы блок оставался в две строки
    details = [f"Транскрибация: {'WhisperX' if whisperx else 'Yandex SpeechKit'}. "
               f"Анализ: {cfg['analysis_model']} ({'локально' if cfg['local_mode'] else 'облако'}).",
               f"Замеров на этом компьютере с этими настройками: транскрибация — {est['n_transcribe']}, "
               f"анализ — {est['n_analyze']}. Во время прогона оценка уточняется по фактической скорости."]
    if missing:
        details.append(f"Для {' и '.join(missing)} замеров ещё нет — взяты стартовые значения; "
                       "после первого прогона с этими настройками оценка станет точной.")
    if approx_n:
        details.append(f"У {approx_n} файлов длительность определена по размеру — для них оценка грубее.")
    with st.container(border=True):
        st.markdown(f"⏱️ **Обработка ≈ {fmt_duration(est['total'])}** · окончание ≈ {finish:%H:%M}{day} · "
                    f"{len(durations)} файлов, {fmt_duration(sum(durations))} аудио")
        st.caption(" · ".join(parts) + (" · ⚠️ приблизительно" if missing or approx_n else ""),
                   help="\n\n".join(details))


# =============== ШАГ 1: ЗАГРУЗКА ===============
if st.session_state.current_step == 1:
    st.header("1. 📁 Загрузите аудиофайлы")
    if st.session_state.get("flash"):
        st.info(st.session_state.pop("flash"))
    estimate_box = st.container()  # прогноз времени — над загрузчиком, заполняется ниже
    uploaded_files = st.file_uploader("Перетащите файлы", type=['mp3', 'wav', 'm4a'], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        # Длительность аудио — основа прогноза времени (считаем один раз на файл)
        _dur_cache = st.session_state.setdefault("audio_durations", {})
        durations, approx_n = [], 0
        for f in uploaded_files:
            key = (f.name, f.size)
            if key not in _dur_cache:
                _dur_cache[key] = audio_duration(f.getvalue(), f.name)
            d, exact = _dur_cache[key]
            durations.append(d)
            approx_n += not exact
        with st.expander(f"📋 Загружено: {len(uploaded_files)}", expanded=True):
            for i, (file, d) in enumerate(zip(uploaded_files, durations)):
                st.write(f"{i+1}. {file.name} — {fmt_duration(d)} ({file.size / 1024:.1f} KB)")
        with estimate_box:
            show_time_estimate(durations, approx_n, current_run_cfg())

        if st.button("🚀 Начать обработку", type="primary"):
            st.session_state.current_step = 2
            st.session_state.processing_results = []
            st.session_state.sheets_saved = None  # новый прогон — прошлая выгрузка не актуальна
            st.rerun()

# =============== ШАГ 2: ОБРАБОТКА ===============
elif st.session_state.current_step == 2:
    st.header("2. 🔄 Обработка файлов")
    job = get_job()
    if job is None or job.claimed:
        # Новый прогон. Настройки снимаем с боковой панели один раз — дальше поток
        # работает с ними, даже если в панели что-то поменяют.
        if not st.session_state.uploaded_files:
            st.warning("❌ Нет файлов")
            if st.button("← Назад"):
                st.session_state.current_step = 1
                st.rerun()
            st.stop()
        job = start_job(st.session_state.uploaded_files, current_run_cfg())

    if not job.active:
        claim_job(job)  # прогон закончился — забираем результаты и уходим со страницы
        st.rerun()

    cfg = job.cfg
    st.caption(f"{'🏠 Локальная' if cfg['local_mode'] else '☁️ Облачная'} модель анализа: "
               f"**{cfg['analysis_model']}** · 🎤 "
               f"{'WhisperX + диаризация' if 'WhisperX' in cfg['transcription_method'] else 'Yandex SpeechKit'}"
               f" · файлов: **{job.total}**")

    # Подтверждение «Стопа» — модальное окно поверх экрана. Пока оно открыто,
    # автообновление экрана выключено (иначе перерисовка могла бы закрыть окно).
    if st.session_state.get("confirm_stop"):
        confirm_stop_dialog(job)

    # Живая часть экрана перерисовывается раз в секунду сама, без перезапуска страницы.
    @st.fragment(run_every=None if st.session_state.get("confirm_stop") else 1.0)
    def job_view():
        status = job.status
        if not job.active:
            st.rerun(scope="app")

        # --- кнопки управления ---
        b1, b2, b3 = st.columns([1, 1, 4])
        with b1:
            if status == "running":
                st.button("⏸️ Пауза", key="job_pause", on_click=job.pause, width="stretch",
                          help="Текущая операция доработает до конца, затем прогон встанет на паузу")
            elif status in ("pausing", "paused"):
                st.button("▶️ Продолжить", key="job_resume", on_click=job.resume, type="primary",
                          width="stretch")
            else:
                st.button("⏸️ Пауза", key="job_pause_off", disabled=True, width="stretch")
        with b2:
            if st.button("⏹️ Стоп", key="job_stop", disabled=(status == "stopping"),
                         width="stretch", help="Сбросить весь текущий анализ"):
                st.session_state.confirm_stop = True
                st.rerun(scope="app")

        # --- что происходит сейчас ---
        if status == "stopping":
            st.warning("⏹️ Останавливаю: дожидаюсь завершения текущей операции, затем анализ будет сброшен...")
        elif status == "paused":
            st.info(f"⏸️ **На паузе** ({fmt_duration(time.time() - (job.pause_started or time.time()))}). "
                    "Нажмите «Продолжить», чтобы возобновить.")
        elif status == "pausing":
            st.info("⏳ Пауза поставлена — дожидаюсь завершения текущей операции...")
        if job.stage_file:
            st.markdown(f"**{job.stage}** · файл {job.stage_idx} из {job.phase_total}  \n`{job.stage_file}`")
        else:
            st.markdown(f"**{job.stage}**")

        n = max(1, job.phase_total)
        text = f"Фаза {job.phase} из 2 · {job.phase_name} · готово {job.phase_done} из {job.phase_total}"
        st.progress(min(1.0, ((job.phase - 1) + job.phase_done / n) / 2), text=text)

        eta = job.eta()
        finish = datetime.now() + timedelta(seconds=eta)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🎤 Транскрибировано", f"{job.transcribed_n} / {job.total}")
        c2.metric("🧠 Проанализировано", f"{job.analysed_n} / {job.total}")
        c3.metric("❌ Ошибок", job.errors_n)
        c4.metric("⏱️ Прошло", fmt_duration(job.elapsed()),
                  f"пауза {fmt_duration(job.paused_total)}" if job.paused_total >= 1 else None,
                  delta_color="off")
        c5.metric("⏳ Осталось ≈", fmt_duration(eta),
                  "на паузе" if status == "paused" else f"окончание ≈ {finish:%H:%M}",
                  delta_color="off",
                  help="Прогноз по замерам на этом компьютере для выбранных настроек и длительности "
                       "оставшихся файлов; уточняется по фактической скорости этого прогона.")
        c6.metric("💰 Стоимость ИИ", f"{job.cost_now():.2f} {COST_CURRENCY}")

        for level, msg in job.notes:
            getattr(st, level)(msg)

        with st.expander("📜 Подробный лог прогона", expanded=False):
            st.caption(f"Последние строки. Полный лог пишется в файл `{job.log.path}` — "
                       "скачать его можно на экране результатов.")
            st.code("\n".join(job.log.tail().splitlines()[-LOG_VIEW_LINES:]), language=None)

    job_view()


elif st.session_state.current_step == 3:
    st.header("3. 📊 Результаты обработки")
    
    # Временные аудио прогона удаляются сами; остаются только после аварийного выхода.
    # (Раньше временной была папка recordings — там же лежит архив записей для
    # process_calls.py, и кнопка удаляла его целиком.)
    if os.path.isdir(TEMP_DIR) and os.listdir(TEMP_DIR):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Удалить временные файлы обработки", type="secondary"):
                try:
                    files_count = 0
                    for filename in os.listdir(TEMP_DIR):
                        os.remove(os.path.join(TEMP_DIR, filename))
                        files_count += 1
                    st.success(f"✅ Удалено временных файлов: {files_count}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка удаления: {e}")

        with col2:
            with st.expander(f"📁 Временные файлы ({TEMP_DIR})", expanded=False):
                files = os.listdir(TEMP_DIR)
                st.write(f"Всего файлов: {len(files)}")
                for f in files:
                    size = os.path.getsize(os.path.join(TEMP_DIR, f))
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
                            sheet_cell_text(result['transcript']),
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
