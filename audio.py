# -*- coding: utf-8 -*-
"""
Аудио-пайплайн: транскрибация (WhisperX + диаризация / Yandex SpeechKit),
предобработка звука, коррекция ролей спикеров. Вынесено из web_app.py.
"""
import os
import re
import shutil
import glob as _glob


if shutil.which("ffmpeg") is None:
    _ff_globs = []
    _local = os.environ.get("LOCALAPPDATA", "")
    if _local:
        _ff_globs.append(os.path.join(_local, "Microsoft", "WinGet", "Packages",
                                       "Gyan.FFmpeg*", "**", "bin", "ffmpeg.exe"))
    for _pat in _ff_globs:
        _hits = _glob.glob(_pat, recursive=True)
        if _hits:
            os.environ["PATH"] = os.path.dirname(_hits[0]) + os.pathsep + os.environ.get("PATH", "")
            print(f"ffmpeg найден и добавлен в PATH: {_hits[0]}")
            break
    else:
        print("⚠️ ffmpeg не найден — локальная транскрибация (WhisperX) работать не будет")

import streamlit as st
import torch

try:
    import whisperx
    WHISPERX_AVAILABLE = True
    print("✅ WhisperX доступен")
except ImportError as e:
    WHISPERX_AVAILABLE = False
    print(f"⚠️ WhisperX не доступен: {e}")

WHISPER_DOMAIN_PROMPT = (
    "Разговор менеджера компании СтальМетУрал (СМУ) с клиентом о металлопрокате. "
    "Термины: арматура А500С, А240, А400, двутавр, балка, швеллер, уголок, "
    "лист горячекатаный, лист оцинкованный, круг, квадрат, полоса, шестигранник, "
    "труба профильная, труба ВГП, труба бесшовная, труба ПНД, профнастил, "
    "проволока, катанка, сетка кладочная, оцинковка, нержавейка, "
    "толщина, диаметр, миллиметр, тонна, ГОСТ, счёт, отсрочка, доставка, самовывоз."
)


def preprocess_audio(audio_path):
    """МИНИМАЛЬНАЯ и БЕЗОПАСНАЯ предобработка звука: только громкостная нормализация
    и срез суб-баса. НИКОГДА не вырезает тишину/речь (никакого VAD/шумодава),
    поэтому фрагменты речи не могут потеряться. Возвращает путь к очищенному wav
    или исходный путь при любой ошибке."""
    import subprocess
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("⚠️ ffmpeg не найден — пропускаю предобработку")
        return audio_path

    out_path = audio_path + ".clean.wav"
    # highpass=80 — режет только гул ниже человеческого голоса;
    # loudnorm — выравнивает громкость (тихого менеджера станет слышно).
    filters = "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11"
    cmd = [ffmpeg, "-y", "-i", audio_path, "-af", filters,
           "-ar", "16000", "-ac", "1", out_path]
    try:
        st.write("🧹 Предобработка звука (нормализация громкости)...")
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path
    except Exception as e:
        print(f"⚠️ Предобработка не удалась, беру оригинал: {e}")
        return audio_path

def pad_audio_simple_silence(audio_path, pad_seconds=2.0):
    """
    Добавляет тишину в начало файла, сдвигая первое слово из "мёртвой зоны"
    начала записи (где VAD/Whisper часто срезает приветствие менеджера).
    Не ломает тайминги диаризации. По умолчанию 2 секунды.
    """
    import soundfile as sf
    import numpy as np

    st.write(f"⏳ [Хак] Добавляем {pad_seconds:g} сек тишины для защиты первого слова...")

    try:
        data, samplerate = sf.read(audio_path)

        if len(data.shape) > 1:
            channels = data.shape[1]
            silence = np.zeros((int(samplerate * pad_seconds), channels), dtype=data.dtype)
        else:
            silence = np.zeros(int(samplerate * pad_seconds), dtype=data.dtype)
            
        padded_data = np.concatenate((silence, data))
        
        padded_path = audio_path + ".silence.wav"
        sf.write(padded_path, padded_data, samplerate)
        
        return padded_path
    except Exception as e:
        print(f"⚠️ Ошибка паддинга: {e}")
        return audio_path

def filter_duplicate_lines(transcript_text):
    """Убирает повторяющиеся подряд строки"""
    lines = transcript_text.split('\n')
    filtered_lines = []
    prev_line = None
    
    for line in lines:
        line = line.strip()
        if line and line != prev_line:
            filtered_lines.append(line)
            prev_line = line
            
    return '\n'.join(filtered_lines)

def transcribe_with_yandex(audio_path, yandex_api_key, aws_access_key_id, aws_secret_access_key, bucket_name):
    """Асинхронная транскрибация через Yandex SpeechKit v2 с диаризацией."""
    import boto3
    import time
    import os
    import requests
    
    start_time = time.time()
    file_name = os.path.basename(audio_path)
    object_name = f"audio_records/{file_name}"
    
    st.write("☁️ [Яндекс] Загрузка файла в облако...")
    
    try:
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        s3.upload_file(audio_path, bucket_name, object_name)
        filelink = f"https://storage.yandexcloud.net/{bucket_name}/{object_name}"
    except Exception as e:
        return f"❌ Ошибка загрузки в Яндекс.Облако: {str(e)}"

    st.write("🚀 [Яндекс] Распознавание и диаризация...")
    
    POST = "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize"
    
    body = {
        "config": {
            "specification": {
                "languageCode": "ru-RU",
                "model": "general",
                "profanityFilter": False,
                "literature_text": True,
                "audioEncoding": "MP3" if audio_path.lower().endswith('.mp3') else "LINEAR16_PCM",
                "diarizationEnabled": True
            }
        },
        "audio": {
            "uri": filelink
        }
    }

    header = {'Authorization': f'Api-Key {yandex_api_key}'}
    
    try:
        req = requests.post(POST, headers=header, json=body)
        req.raise_for_status()
        data = req.json()
        task_id = data.get('id')
    except Exception as e:
        return f"❌ Ошибка SpeechKit API: {req.text if 'req' in locals() else str(e)}"

    while True:
        time.sleep(5)
        GET = f"https://operation.api.cloud.yandex.net/operations/{task_id}"
        req = requests.get(GET, headers=header)
        req.raise_for_status()
        data = req.json()

        if data.get('done'):
            break

    if 'response' in data and 'chunks' in data['response']:
        st.write("✅ [Яндекс] Текст получен. Сборка диалога...")
        
        segments = []
        for chunk in data['response']['chunks']:
            channel = chunk['alternatives'][0].get('channelTag', '1')
            text = chunk['alternatives'][0].get('text', '')
            
            if text:
                segments.append({
                    "speaker": f"SPEAKER_0{channel}",
                    "text": text
                })
                
        speaker_manager, speaker_client = identify_speaker_roles(segments)
        
        labeled_segments = []
        for segment in segments:
            spk = segment["speaker"]
            txt = segment["text"]
            labeled_segments.append(f"{spk}: {txt}")

        full_text = "\n".join(labeled_segments)
        full_text = filter_duplicate_lines(full_text)
        
        try:
            s3.delete_object(Bucket=bucket_name, Key=object_name)
        except:
            pass
            
        elapsed = time.time() - start_time
        st.write(f"✅ Яндекс завершил работу за {elapsed:.1f} сек!")
        return full_text
    else:
        return f"❌ Ошибка: Яндекс не вернул текст. Ответ: {data}"

@st.cache_resource(show_spinner="Загрузка WhisperX в память...")
def load_whisperx_model(model_name, device, compute_type):
    """Загружает и кеширует основную модель WhisperX.
    initial_prompt (доменный словарь) биасит распознавание к нашим терминам."""
    print(f"🔧 [Cache] Загрузка WhisperX модели {model_name} на {device}...")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language="ru",
        asr_options={"initial_prompt": WHISPER_DOMAIN_PROMPT}
    )
    return model

@st.cache_resource(show_spinner="Загрузка модели выравнивания...")
def load_align_model(language_code, device):
    """Загружает и кеширует модель выравнивания (alignment)"""
    print(f"📍 [Cache] Загрузка модели выравнивания ({language_code})...")
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    return model_a, metadata

def build_turns_from_words(result):
    """Собирает реплики из ПОСЛОВНОЙ разметки спикеров (точные границы смены спикера).
    Это убирает слипание фраз, когда один сегмент Whisper охватывает двух людей.
    Возвращает список {speaker, text} или None — тогда вызывающий код откатывается
    на старую посегментную нарезку (текущий уровень не деградирует)."""
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            txt = w.get("word", "")
            if txt and txt.strip():
                words.append({"speaker": w.get("speaker"), "word": txt})
    if not words:
        return None

    # Заполняем слова без спикера ближайшим известным (вперёд, затем назад)
    last = None
    for w in words:
        if w["speaker"]:
            last = w["speaker"]
        elif last:
            w["speaker"] = last
    nxt = None
    for w in reversed(words):
        if w["speaker"]:
            nxt = w["speaker"]
        elif nxt:
            w["speaker"] = nxt
    if any(not w["speaker"] for w in words):
        return None  # разметки спикеров нет вовсе — откат

    # Сглаживание дребезга: короткие "островки" (1-2 слова) чужого спикера,
    # окружённые с ОБЕИХ сторон одним и тем же спикером, переносим к окружению.
    runs = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j]["speaker"] == words[i]["speaker"]:
            j += 1
        runs.append([words[i]["speaker"], i, j])
        i = j
    MAX_ISLAND = 2
    for k in range(1, len(runs) - 1):
        spk, s, e = runs[k]
        if (e - s) <= MAX_ISLAND and runs[k - 1][0] == runs[k + 1][0] and runs[k - 1][0] != spk:
            for idx in range(s, e):
                words[idx]["speaker"] = runs[k - 1][0]

    # Группируем подряд идущие слова одного спикера в реплики
    turns = []
    cur_spk = words[0]["speaker"]
    buf = [words[0]["word"]]
    for w in words[1:]:
        if w["speaker"] == cur_spk:
            buf.append(w["word"])
        else:
            text = " ".join(s.strip() for s in buf if s.strip())
            if text:
                turns.append({"speaker": cur_spk, "text": text})
            cur_spk = w["speaker"]
            buf = [w["word"]]
    text = " ".join(s.strip() for s in buf if s.strip())
    if text:
        turns.append({"speaker": cur_spk, "text": text})
    return turns or None

def transcribe_with_whisperx_diarization(audio_path, hf_token, model_name="large-v3", min_speakers=2, max_speakers=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"🔧 Устройство: {device} | 📦 Модель: {model_name}")
    
    try:
        print("⏳ Получение модели WhisperX из кеша...")
        model = load_whisperx_model(model_name, device, compute_type)
        
        print("🎤 Загрузка аудио...")
        audio = whisperx.load_audio(audio_path)
        
        print("📝 Транскрибация...")
        result = model.transcribe(
            audio, 
            batch_size=14 if device == "cuda" else 1,
            language="ru"
        )
      
        print("📍 Выравнивание по словам...")
        model_a, metadata = load_align_model("ru", device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
        
        print("👥 Диаризация спикеров...")
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        
        diarize_segments = diarize_model(
            audio, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers
        )
        
        print("🔗 Привязка спикеров...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Посегментная нарезка (пословная давала слипание коротких реплик — откатили)
        segments = []
        for segment in result["segments"]:
            speaker = segment.get("speaker", "SPEAKER_00")
            text = segment.get("text", "").strip()
            if text:
                segments.append({
                    "speaker": speaker,
                    "text": text,
                    "start": segment["start"],
                    "end": segment["end"]
                })
        
        speaker_manager, speaker_client = identify_speaker_roles(segments)
        print(f"🎭 Роль определена: Менеджер={speaker_manager}, Клиент={speaker_client}")
        
        labeled_segments = []
        speakers_stats = {}
        
        for segment in segments:
            speaker = segment.get("speaker", "SPEAKER_00")
            text = segment.get("text", "").strip()
            
            if text:
                speakers_stats[speaker] = speakers_stats.get(speaker, 0) + 1
                
                if speaker == speaker_manager:
                    labeled_segments.append(f"👨‍💼 Менеджер: {text}")
                elif speaker == speaker_client:
                    labeled_segments.append(f"👤 Клиент: {text}")
                else:
                    labeled_segments.append(f"{speaker}: {text}")
        
        full_text = "\n".join(labeled_segments)
        full_text = filter_duplicate_lines(full_text)
        
        print(f"✅ Готово! Спикеров: {len(speakers_stats)}")
        for spk, count in speakers_stats.items():
            role = "Менеджер" if spk == speaker_manager else "Клиент" if spk == speaker_client else "Неизвестно"
            print(f"   {spk} ({role}): {count} реплик")
            
        return full_text
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ ОШИБКА WhisperX: {str(e)}\n{error_details}"

def identify_speaker_roles(segments):
    def normalize_for_analysis(text):
        text = text.lower()
        corrections = {
            'копиер лица': 'физ лицо', 'продекоруйте': 'продиктуйте',
            'протикните': 'продиктуйте', 'сальметро': 'стальметурал',
            'сталин металл': 'стальметурал', 'стермит урал': 'стальметурал',
            'дальмед урал': 'стальметурал', 'тимотров': 'стальметурал',
            'альметрол': 'стальметурал', 'пмд': 'пнд', 'стартиковая': 'пластиковая',
            'собака': '@', 'точка ру': '.ру', 'м-608': 'm608',
            'ликвидитам': 'реквизиты',
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text

    def analyze_intent(text):
        text = normalize_for_analysis(text)
        client_patterns = [r'мне нужно', r'нам нужно', r'хочу', r'интересует',
                          r'есть у вас', r'а у вас есть', r'подскажите',
                          r'сколько стоит', r'какая цена', r'доставка есть',
                          r'могу забрать', r'как заказать', r'купить']
        manager_patterns = [r'посмотрю', r'уточню', r'проверю', r'сейчас',
                           r'доставкой можем', r'оформим', r'выставлю',
                           r'стоимость', r'цена за', r'по наличию', r'будет']
        c_score = sum(1 for p in client_patterns if re.search(p, text))
        m_score = sum(1 for p in manager_patterns if re.search(p, text))
        if c_score > m_score: return "client"
        elif m_score > c_score: return "manager"
        return "neutral"
        
    manager_keywords = {
        'добрый день': 2, 'здравствуйте': 2, 'чем могу помочь': 5, 'слушаю': 3,
        'стальметурал': 10, 'стальмет': 10, 'сму': 10, 'компания': 4,
        'сколько вам нужно': 5, 'в каком городе': 5, 'вам какие': 2,
        'сейчас посмотрю': 5, 'по наличию': 5, 'под заказ': 3,
        'стоимость': 3, 'цена': 2, 'со склада': 3,
        'доставкой': 4, 'по предоплате': 5, 'на карту': 2,
        'продиктуйте': 5, 'запишите': 3, 'карту предприятия': 5,
        'инн': 5, 'кпп': 5, 'заявку': 3, 'на whatsapp': 2, 'на почту': 2,
        'вы как организация': 5, 'или частное лицо': 5, 'физлицо или юрлицо': 5,
        'выставлю счет': 5, 'отправлю предложение': 5
    }

    client_keywords = {
        'а у вас есть': 5, 'мне нужно': 5, 'нам нужно': 5, 'хочу': 4,
        'интересует': 4, 'подскажите': 3, 'не подскажете': 3,
        'сколько стоит': 5, 'какая цена': 5, 'доставка есть': 4,
        'меня зовут': 3, 'я физлицо': 5, 'от юридического': 5,
        'а из дерева нету': 4, 'мне вот такая не надо': 4
    }

    company_keywords = ['стальметурал', 'стальмет', 'сму', 'стальмедурал']
    speaker_scores = {}
    speaker_intents = {}
    
    for segment in segments:
        speaker = segment.get("speaker", "SPEAKER_00")
        text = segment.get("text", "").strip()
        normalized = normalize_for_analysis(text)
        if speaker not in speaker_scores:
            speaker_scores[speaker] = {"manager": 0, "client": 0}
            speaker_intents[speaker] = {"manager": 0, "client": 0}
        
        if any(c in normalized for c in company_keywords):
            if 'добрый день' in normalized or 'здравствуйте' in normalized:
                speaker_scores[speaker]["manager"] += 10
        
        for kw, w in manager_keywords.items():
            if kw in normalized:
                speaker_scores[speaker]["manager"] += w
        for kw, w in client_keywords.items():
            if kw in normalized:
                speaker_scores[speaker]["client"] += w
        
        intent = analyze_intent(text)
        if intent == "manager":
            speaker_intents[speaker]["manager"] += 2
        elif intent == "client":
            speaker_intents[speaker]["client"] += 2
            
    for spk in speaker_scores:
        speaker_scores[spk]["manager"] += speaker_intents[spk]["manager"]
        speaker_scores[spk]["client"] += speaker_intents[spk]["client"]
    
    question_patterns = [r'\?', r'сколько', r'как', r'какой', r'какая', r'какие', r'где', r'когда', r'почему', r'зачем']
    for i in range(len(segments) - 1):
        current_speaker = segments[i].get("speaker")
        next_speaker = segments[i+1].get("speaker")
        current_text = normalize_for_analysis(segments[i].get("text", ""))
        
        if any(re.search(p, current_text) for p in question_patterns):
            if current_speaker in speaker_scores:
                if speaker_scores[current_speaker]["manager"] > speaker_scores[current_speaker]["client"]:
                    speaker_scores[next_speaker]["client"] += 1
                elif speaker_scores[current_speaker]["client"] > speaker_scores[current_speaker]["manager"]:
                    speaker_scores[next_speaker]["manager"] += 1
                    
    speakers = list(speaker_scores.keys())
    if len(speakers) >= 2:
        m_diff = abs(speaker_scores[speakers[0]]["manager"] - speaker_scores[speakers[1]]["manager"])
        max_m = max(speaker_scores[speakers[0]]["manager"], speaker_scores[speakers[1]]["manager"])
        if max_m > 0 and m_diff < max_m * 0.25:
            for seg in segments[:3]:
                txt = normalize_for_analysis(seg.get("text", "").strip())
                if 'добрый день' in txt and any(c in txt for c in company_keywords):
                    spk = seg.get("speaker")
                    return spk, (speakers[1] if spk == speakers[0] else speakers[0])
        if speaker_scores[speakers[0]]["manager"] > speaker_scores[speakers[1]]["manager"]:
            return speakers[0], speakers[1]
        else:
            return speakers[1], speakers[0]
    elif len(speakers) == 1:
        return speakers[0], None
    else:
        return "SPEAKER_00", "SPEAKER_01"

def normalize_text(text):
    corrections = {
        'копиер лица': 'физ лица',
        'продекоруйте': 'продиктуйте',
        'сальметро': 'стальметурал',
        'сталин металл': 'стальметурал',
        'собака': '@',
        'точка ру': '.ру',
        'м-608': 'm608',
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text
