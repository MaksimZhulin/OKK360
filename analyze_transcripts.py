import os
import json
import time
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai

# ============== НАСТРОЙКИ ==============
# 1. DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "google/gemini-2.5-flash"

# 2. Google Таблица
CREDENTIALS_FILE = 'credentials.json'
SPREADSHEET_ID = "1Oe-dKF_0oPhCdlwcj6jeco7BSIBi37jPuO3rSG4C930"
SHEET_NAME = "'Выгрузка из проекта'"

# 3. Настройки анализа
MAX_TOKENS = 4000
# =======================================

def setup_deepseek():
    """Настраивает клиент DeepSeek"""
    client = openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://litellm.tokengate.ru/v1"
    )
    return client

def analyze_call_transcript(client, transcript_text):
    """Анализирует текст звонка с помощью ИИ"""
    
    prompt = f"""Проанализируй транскрипцию звонка и выдели ключевую информацию.

Текст звонка:
{transcript_text[:MAX_TOKENS]}

Проанализируй и верни ответ в формате JSON с такими полями:
1. "topic": основная тема звонка (1-3 слова)
2. "client_request": что хотел клиент (1-2 предложения)
3. "solution": что было предложено/сделано (1-2 предложения)
4. "urgency": срочность (Низкая/Средняя/Высокая)
5. "client_mood": настроение клиента (Нейтральное/Заинтересованное/Раздраженное/Довольное/Сомневающееся)
6. "manager_actions": действия менеджера (1-2 предложения)
7. "quality_score": оценка качества работы от 1 до 5 (только цифра)

Верни ТОЛЬКО JSON, без пояснений:"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "Ты помощник для анализа транскрипций звонков. Отвечай строго в формате JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content
        
        # Очищаем ответ от возможных лишних символов
        result_text = result_text.strip()
        
        # Ищем JSON в ответе
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            result_text = result_text[start_idx:end_idx]
        
        # Парсим JSON
        analysis_result = json.loads(result_text)
        
        # Проверяем обязательные поля
        required_fields = ["topic", "client_request", "solution", "urgency", 
                          "client_mood", "manager_actions", "quality_score"]
        
        for field in required_fields:
            if field not in analysis_result:
                analysis_result[field] = "Не определено"
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        return get_default_analysis()
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return get_default_analysis()

def get_default_analysis():
    """Возвращает анализ по умолчанию при ошибке"""
    return {
        "topic": "Не удалось определить",
        "client_request": "Не удалось определить",
        "solution": "Не удалось определить",
        "urgency": "Средняя",
        "client_mood": "Нейтральное",
        "manager_actions": "Не удалось определить",
        "quality_score": 3
    }

def get_google_sheets_service():
    """Подключается к Google Sheets"""
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=credentials)
    return service

def get_unanalyzed_rows(service):
    """Находит строки для анализа - проверяет только поле F"""
    try:
        # Читаем 3 колонки: D, E, F
        # D - текст звонка
        # E - может быть что-то еще (пропускаем)
        # F - поле "topic" (первое поле анализа)
        range_name = f"{SHEET_NAME}!D2:F1000"
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            print("📭 В таблице нет данных")
            return []
        
        unanalyzed_rows = []
        
        for i, row in enumerate(values):
            table_row_number = i + 2  # +2 потому что начинаем со строки 2
            
            # ПРОВЕРКА 1: Есть ли текст в колонке D?
            if len(row) == 0:  # Пустая строка
                continue
            
            if not row[0] or not str(row[0]).strip():  # Нет текста в колонке D
                continue
            
            # ПРОВЕРКА 2: Заполнено ли поле F (topic)?
            if len(row) < 3:  # Есть только D и E, F нет
                # F пустое - нужно анализировать
                unanalyzed_rows.append(table_row_number)
                continue
            
            field_f = row[2]  # Это колонка F (индекс 2)
            
            if not field_f or not str(field_f).strip():  # F пустое
                unanalyzed_rows.append(table_row_number)
            elif field_f.strip() == "Не удалось определить":  # Была ошибка анализа
                unanalyzed_rows.append(table_row_number)
        
        return unanalyzed_rows
        
    except Exception as e:
        print(f"❌ Ошибка поиска строк для анализа: {e}")
        return []

def get_transcript_text(service, row_number):
    """Получает текст транскрипции из указанной строки"""
    try:
        range_name = f"{SHEET_NAME}!D{row_number}"
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name
        ).execute()
        
        values = result.get('values', [])
        
        if not values or not values[0]:
            return ""
        
        return values[0][0]
        
    except Exception as e:
        print(f"❌ Ошибка получения текста из строки {row_number}: {e}")
        return ""

def update_analysis_results(service, row_number, analysis_result):
    """Записывает результаты анализа в таблицу"""
    
    # Подготавливаем данные для записи
    row_data = [
        analysis_result["topic"],           # F
        analysis_result["client_request"],  # G
        analysis_result["solution"],        # H
        analysis_result["urgency"],         # I
        analysis_result["client_mood"],     # J
        analysis_result["manager_actions"], # K
        analysis_result["quality_score"]    # L
    ]
    
    range_name = f"{SHEET_NAME}!F{row_number}:L{row_number}"
    
    body = {
        'values': [row_data]
    }
    
    try:
        result = service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute()
        
        print(f"✅ Анализ записан в строку {row_number}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка записи анализа в строку {row_number}: {e}")
        return False

def main():
    """Основная функция анализатора"""
    print("="*60)
    print("АНАЛИЗАТОР ТРАНСКРИПЦИЙ С ИИ (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print("="*60)
    
    print("🔧 Настраиваю подключения...")
    
    try:
        deepseek_client = setup_deepseek()
        print("✅ DeepSeek API подключен")
        
        sheets_service = get_google_sheets_service()
        print("✅ Google Sheets подключен")
        
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return
    
    print("\n🔍 Ищу транскрипции для анализа...")
    unanalyzed_rows = get_unanalyzed_rows(sheets_service)
    
    if not unanalyzed_rows:
        print("📭 Все транскрипции уже проанализированы!")
        return
    
    print(f"📊 Найдено транскрипций для анализа: {len(unanalyzed_rows)}")
    
    # Выводим номера строк для отладки
    if len(unanalyzed_rows) > 0:
        print(f"📝 Строки для анализа: {unanalyzed_rows}")
    
    total_analyzed = 0
    
    for i, row_num in enumerate(unanalyzed_rows, 1):
        print(f"\n{'='*40}")
        print(f"АНАЛИЗ {i}/{len(unanalyzed_rows)}: Строка {row_num}")
        print('='*40)
        
        try:
            transcript_text = get_transcript_text(sheets_service, row_num)
            
            if not transcript_text or len(transcript_text.strip()) < 10:
                print("⚠️ Текст слишком короткий, пропускаем")
                continue
            
            print(f"📄 Символов в тексте: {len(transcript_text)}")
            print("🤖 Анализирую с помощью ИИ...")
            
            # Проверяем баланс DeepSeek
            try:
                analysis_result = analyze_call_transcript(deepseek_client, transcript_text)
            except Exception as e:
                if "402" in str(e) or "Insufficient Balance" in str(e):
                    print("❌ НЕТ БАЛАНСА НА DEEPSEEK API!")
                    print("   Пополните баланс на platform.deepseek.com")
                    print("   Пока использую значения по умолчанию")
                    analysis_result = get_default_analysis()
                else:
                    raise e
            
            print(f"\n📋 РЕЗУЛЬТАТЫ АНАЛИЗА:")
            print(f"   • Тема: {analysis_result['topic']}")
            print(f"   • Оценка: {analysis_result['quality_score']}/5")
            
            if update_analysis_results(sheets_service, row_num, analysis_result):
                total_analyzed += 1
                print(f"✅ Успешно проанализировано")
            
            # Пауза между запросами
            if i < len(unanalyzed_rows):
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Ошибка при анализе строки {row_num}: {e}")
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("="*60)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ: Проанализировано {total_analyzed} транскрипций")
    
    print(f"\n🔗 Ссылка на таблицу:")
    print(f"   https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit")

if __name__ == "__main__":
    main()
    input("\nНажмите Enter для выхода...")