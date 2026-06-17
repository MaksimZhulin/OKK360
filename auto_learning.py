import json
import os
import hashlib
from datetime import datetime

DATABASE_FILE = "knowledge_base.json"

def get_call_hash(transcript):
    """Создаёт уникальный хэш звонка"""
    return hashlib.md5(transcript[:1000].encode()).hexdigest()

def calculate_confidence(analysis_result, transcript=""):
    """
    Вычисляет уверенность модели на основе адекватности оценки, 
    качества диаризации и отсутствия логических галлюцинаций.
    """
    confidence = 0

    # === ФАКТОР 1: Заполненность полей (Макс 15 баллов) ===
    # Оцениваем, что ИИ не выдал пустые строки или слишком короткие отписки
    required_fields = ["topic", "client_request", "solution", "urgency", "client_mood", "manager_actions"]
    filled = sum(1 for f in required_fields if analysis_result.get(f) and len(str(analysis_result.get(f)).strip()) > 5)
    confidence += (filled / len(required_fields)) * 15

    # === ФАКТОР 2: Валидность и реалистичность чек-листа (Макс 20 баллов) ===
    binary_fields = ["establishing_contact", "client_type", "clarifying_questions",
                     "knowledge_quality", "contact_exchange", "software_proficiency", "call_completion"]
    
    # Проверяем, что оценки строго 0 или 1 (без сломанных данных)
    valid_binaries = sum(1 for f in binary_fields if analysis_result.get(f) in [0, 1, "0", "1"])
    confidence += (valid_binaries / len(binary_fields)) * 10
    
    # Логика паттернов реальных звонков (до 10 баллов)
    call_comp = int(analysis_result.get("call_completion", 0))
    exch_cont = int(analysis_result.get("contact_exchange", 0))
    cl_quest = int(analysis_result.get("clarifying_questions", 0))
    
    # 1. "Успешная сделка": Выявил потребности, взял контакты, закрыл сделку
    if call_comp == 1 and exch_cont == 1 and cl_quest == 1:
        confidence += 10
    # 2. "Отказ / Быстрый ответ": Сделку не закрыл, контакты не взял (логичный исход короткого звонка)
    elif call_comp == 0 and exch_cont == 0:
        confidence += 10
    # 3. "Звонок в работе": Контакты взял, но сделку в этом звонке не закрыл (классика В2В)
    elif exch_cont == 1 and call_comp == 0:
        confidence += 10
    else:
        # Нелогичные комбинации (например, звонок "успешно завершен", но контактов нет) баллов не получают
        confidence += 0

    # === ФАКТОР 3: Качество диаризации и транскрипции (Макс 30 баллов) ===
    # КРИТИЧЕСКИЙ ПУНКТ: Проверяем, не перепутал ли ИИ роли и есть ли диалог
    if transcript:
        transcript_lower = transcript.lower()
        manager_count = transcript_lower.count("менеджер:")
        client_count = transcript_lower.count("клиент:")
        
        # Если присутствуют оба спикера
        if manager_count > 0 and client_count > 0:
            confidence += 15
            
            # Проверяем баланс (защита от того, что 90% текста "съел" один спикер из-за ошибки Whisper)
            total_lines = manager_count + client_count
            if total_lines >= 4: # Должен быть нормальный диалог (от 4 реплик)
                ratio = min(manager_count, client_count) / total_lines
                if ratio >= 0.15: # Здоровый пинг-понг (никто не молчит весь звонок)
                    confidence += 15
                else:
                    confidence += 5 # Сильный перекос
        else:
            return 0 # КРИТИЧЕСКАЯ ОШИБКА: Нет диалога или роли не размечены. Сразу бракуем звонок!
    else:
        confidence += 10 # Заглушка, если транскрипт не передали

    # === ФАКТОР 4: Защита от галлюцинаций ИИ (Макс 20 баллов) ===
    # Проверка на то, что текстовое описание совпадает с бинарными цифрами
    solution_text = str(analysis_result.get("solution", "")).lower()
    topic_text = str(analysis_result.get("topic", "")).lower()
    
    if topic_text and "ошибка" not in topic_text and "не определено" not in topic_text:
        confidence += 10
        
    # Ищем жесткие противоречия:
    # ИИ поставил "успешное завершение" = 1, а тексте пишет про отказ
    if call_comp == 1 and any(word in solution_text for word in ["отказ", "нет в наличии", "не устроил", "дорого", "отказался"]):
        confidence -= 20 # Жесткий штраф за галлюцинацию
    # ИИ поставил "завершение" = 0, а в тексте счет или договор
    elif call_comp == 0 and any(word in solution_text for word in ["выставил счет", "оплатил", "отправил кп", "договорились"]):
        confidence -= 20 # Жесткий штраф за галлюцинацию
    else:
        confidence += 10

    # === ФАКТОР 5: Эмоциональная консистентность (Макс 15 баллов) ===
    urgency = analysis_result.get("urgency")
    mood = analysis_result.get("client_mood")
    
    if urgency == "Высокая" and mood in ["Раздраженное", "Сомневающееся"]:
        confidence += 15
    elif urgency == "Низкая" and mood in ["Нейтральное", "Довольное"]:
        confidence += 15
    elif mood in ["Заинтересованное", "Довольное", "Нейтральное", "Сомневающееся", "Раздраженное"]:
        confidence += 10 # Если значения просто из допустимого списка

    return int(max(0, min(confidence, 100))) # Удерживаем в рамках 0-100

def save_to_knowledge_base(transcript, analysis, confidence, auto_save=True, filename="unknown"):
    """Сохраняет звонок в базу знаний"""
    if auto_save and confidence < 85:
        print(f"⚠️ Уверенность {confidence}% < 85% — не сохраняем автоматически")
        return False
    
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            database = json.load(f)
    else:
        database = {"calls": []}
    
    # Теперь хэш не нужен, ищем прямо по имени файла
    for existing in database.get("calls", []):
        if existing.get("filename") == filename:
            print(f"ℹ️ Звонок '{filename}' уже есть в базе. Обновляем его.")
            # Если звонок найден, просто обновляем его данные (перезаписываем старый тест новым)
            existing["transcript"] = transcript[:3000]
            existing["analysis"] = analysis
            existing["confidence"] = confidence
            existing["timestamp"] = datetime.now().isoformat()
            
            with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
                json.dump(database, f, ensure_ascii=False, indent=2)
            return True
    
    # Если такого файла еще нет, добавляем как новый
    database["calls"].append({
        "id": hashlib.md5(filename.encode()).hexdigest(), # ID делаем из имени файла
        "filename": filename, # Сохраняем имя файла
        "transcript": transcript[:3000],
        "analysis": analysis,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "auto_saved": auto_save
    })
    
    with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Сохранено в базу знаний! Уверенность: {confidence}%")
    return True




def find_similar_calls(transcript, top_k=3):
    """Ищет похожие звонки в базе"""
    if not os.path.exists(DATABASE_FILE):
        return []
    
    with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    if not database.get("calls"):
        return []
    
    # Простой поиск по ключевым словам из транскрипции
    transcript_lower = transcript.lower()
    scored_calls = []
    
    for call in database["calls"]:
        score = 0
        call_transcript = call.get("transcript", "").lower()
        
        # Поиск общих фраз
        common_words = set(transcript_lower.split()) & set(call_transcript.split())
        score = len(common_words)
        
        # Бонус за высокую уверенность в прошлом анализе
        score += call.get("confidence", 50) / 10
        
        # Бонус за недавние звонки
        score += 5
        
        scored_calls.append((score, call))
    
    # Сортируем и возвращаем топ
    scored_calls.sort(reverse=True, key=lambda x: x[0])
    return [call for _, call in scored_calls[:top_k]]

def get_database_stats():
    """Статистика базы знаний"""
    if not os.path.exists(DATABASE_FILE):
        return {"total": 0, "auto_saved": 0, "avg_confidence": 0}
    
    with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    calls = database.get("calls", [])
    return {
        "total": len(calls),
        "auto_saved": sum(1 for c in calls if c.get("auto_saved")),
        "avg_confidence": sum(c.get("confidence", 0) for c in calls) / len(calls) if calls else 0
    }
