import os
import re
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import glob

# ============== НАСТРОЙКИ ==============
# 1. Пути к файлам
CREDENTIALS_FILE = 'credentials.json'
TRANSCRIPTS_FOLDER = 'transcripts'
PROCESSED_FOLDER = 'processed'
LOG_FILE = 'upload_log.txt'

# 2. ID вашей Google Таблицы
SPREADSHEET_ID = "1Oe-dKF_0oPhCdlwcj6jeco7BSIBi37jPuO3rSG4C930"

# 3. Настройки таблицы
SHEET_NAME = "'Выгрузка из проекта'"  # Название листа в таблице (в кавычках — есть пробелы)
# =======================================

def setup_folders():
    """Создает необходимые папки"""
    for folder in [TRANSCRIPTS_FOLDER, PROCESSED_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Создана папка: {folder}")

def parse_filename(filename):
    """Извлекает информацию из имени файла"""
    base_name = os.path.splitext(filename)[0]
    
    # Удаляем расширение .mp3.txt если есть
    if base_name.endswith('.mp3'):
        base_name = base_name[:-4]
    
    # Пытаемся найти дату в имени файла
    date_patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',      # 2023-12-15
        r'(\d{2})\.(\d{2})\.(\d{4})',    # 15.12.2023
        r'(\d{4})(\d{2})(\d{2})',        # 20231215
    ]
    
    file_date = None
    for pattern in date_patterns:
        match = re.search(pattern, base_name)
        if match:
            try:
                if '-' in pattern:
                    year, month, day = map(int, match.groups())
                elif '.' in pattern:
                    day, month, year = map(int, match.groups())
                else:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))
                
                file_date = f"{day:02d}.{month:02d}.{year}"
                break
            except:
                continue
    
    # Если дата не найдена, используем дату изменения файла
    if not file_date:
        file_path = os.path.join(TRANSCRIPTS_FOLDER, filename)
        mod_time = os.path.getmtime(file_path)
        file_date = datetime.fromtimestamp(mod_time).strftime("%d.%m.%Y")
    
    # Извлекаем имя клиента из названия файла
    client_name = "Не указан"
    
    # Паттерны для поиска ФИО (русские имена с тире или подчеркиванием)
    name_patterns = [
        r'([А-Я][а-я]+-[А-Я][а-я]+-[А-Я][а-я]+)',  # Иванов-Иван-Иванович
        r'([А-Я][а-я]+_[А-Я][а-я]+_[А-Я][а-я]+)',  # Иванов_Иван_Иванович
        r'([А-Я][а-я]+\s[А-Я]\.\s?[А-Я]\.)',       # Иванов И. И.
        r'([А-Я][а-я]+\s[А-Я][а-я]+\s[А-Я][а-я]+)', # Иванов Иван Иванович
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, base_name)
        if match:
            client_name = match.group(1).replace('_', ' ').replace('-', ' ')
            break
    
    return {
        'filename': filename,
        'date': file_date,
        'client_name': client_name,
        'clean_filename': base_name
    }

def read_transcript_file(filepath):
    """Читает содержимое текстового файла с разными кодировками"""
    encodings = ['utf-8', 'cp1251', 'cp866', 'iso-8859-1', 'windows-1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read().strip()
            return content[:5000]  # Ограничиваем длину
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue
    
    # Если ни одна кодировка не подошла, пробуем binary mode
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        return content.decode('utf-8', errors='replace').strip()[:5000]
    except Exception as e:
        print(f"❌ Не удалось прочитать файл: {e}")
        return ""

def get_next_empty_row(service, spreadsheet_id, sheet_name):
    """Находит следующую пустую строку в таблице"""
    try:
        sheets = service.spreadsheets()
        
        # Читаем столбец A полностью
        range_name = f"{sheet_name}!A:A"
        result = sheets.values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            majorDimension="COLUMNS"
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            return 1  # Таблица полностью пустая
        
        column_a = values[0] if values else []
        
        # Ищем первую пустую ячейку
        for i, cell in enumerate(column_a):
            if not cell or str(cell).strip() == '':
                return i + 1  # +1 потому что в Sheets нумерация с 1
        
        # Если все строки заполнены, возвращаем следующую
        return len(column_a) + 1
        
    except Exception as e:
        print(f"   ⚠️ Не удалось найти пустую строку: {e}")
        return 2

def upload_to_sheets(transcript_data, next_row):
    """Загружает данные в Google Таблицу"""
    
    print("\n" + "="*60)
    print("ПОДКЛЮЧЕНИЕ К GOOGLE ТАБЛИЦЕ")
    print("="*60)
    
    try:
        # 1. Аутентификация
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        service = build('sheets', 'v4', credentials=credentials)
        
        print(f"✅ Подключено как: {credentials.service_account_email}")
        print(f"📝 Будет записано в строку: {next_row}")
        
        # 2. Подготавливаем данные для записи (5 колонок вместо 6)
        row_data = [
            transcript_data['date'],           # Колонка A: Дата звонка
            transcript_data['client_name'],    # Колонка B: Клиент
            transcript_data['filename'],       # Колонка C: Имя файла
            transcript_data['content'],        # Колонка D: Текст транскрипции
            datetime.now().strftime("%d.%m.%Y %H:%M"),  # Колонка E: Дата загрузки
            # Убрана колонка F (статус) - можно добавить при необходимости
        ]
        
        # 3. Записываем данные (A-E колонки)
        range_name = f"{SHEET_NAME}!A{next_row}:E{next_row}"
        
        body = {
            'values': [row_data]
        }
        
        result = service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute()
        
        print(f"✅ Данные успешно записаны в строку {next_row}!")
        print(f"   Обновлено ячеек: {result.get('updatedCells', 5)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке в таблицу: {e}")
        return False

def move_to_processed(filename):
    """Перемещает обработанный файл в папку processed"""
    try:
        src = os.path.join(TRANSCRIPTS_FOLDER, filename)
        dst = os.path.join(PROCESSED_FOLDER, filename)
        
        # Если файл уже существует в processed, добавляем timestamp
        if os.path.exists(dst):
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
            dst = os.path.join(PROCESSED_FOLDER, f"{name}{timestamp}{ext}")
        
        os.rename(src, dst)
        print(f"📦 Файл перемещен в: {PROCESSED_FOLDER}")
        return True
    except Exception as e:
        print(f"⚠️ Не удалось переместить файл: {e}")
        return False

def log_upload(filename, success=True):
    """Записывает результат в лог-файл"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "УСПЕХ" if success else "ОШИБКА"
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} | {filename} | {status}\n")
    except:
        pass

def create_header_if_needed(service):
    """Создает заголовки если таблица пустая"""
    try:
        # Проверяем первую строку
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SHEET_NAME}!A1:E1"
        ).execute()
        
        values = result.get('values', [])
        
        # Если таблица пустая или нет заголовков
        if not values or len(values[0]) < 5:
            headers = [
                ["Дата звонка", "Клиент", "Файл", "Транскрипция", "Дата загрузки"]
            ]
            
            body = {
                'values': headers
            }
            
            service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{SHEET_NAME}!A1:E1",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            print("📋 Созданы заголовки таблицы")
            return True
            
    except Exception as e:
        print(f"⚠️ Не удалось проверить/создать заголовки: {e}")
    
    return False

def main():
    """Основная функция"""
    print("="*60)
    print("ЗАГРУЗЧИК ТРАНСКРИПЦИЙ В GOOGLE ТАБЛИЦУ")
    print("="*60)
    
    # 1. Настраиваем папки
    setup_folders()
    
    # 2. Ищем текстовые файлы
    transcript_files = glob.glob(os.path.join(TRANSCRIPTS_FOLDER, '*.txt'))
    
    if not transcript_files:
        print(f"\n📭 В папке '{TRANSCRIPTS_FOLDER}' нет текстовых файлов (*.txt)")
        print("   Положите файлы транскрипций в эту папку и запустите скрипт снова.")
        return
    
    print(f"\n📁 Найдено файлов: {len(transcript_files)}")
    
    # 3. Инициализируем сервис
    try:
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        service = build('sheets', 'v4', credentials=credentials)
        print("✅ Сервис Google Sheets инициализирован")
    except Exception as e:
        print(f"❌ Не удалось инициализировать сервис: {e}")
        return
    
    # 4. Создаем заголовки если нужно
    create_header_if_needed(service)
    
    # 5. Получаем стартовую строку (после заголовков)
    start_row = get_next_empty_row(service, SPREADSHEET_ID, SHEET_NAME)
    if start_row < 2:
        start_row = 2  # Всегда начинаем со 2 строки (после заголовков)
    
    print(f"📊 Начинаем запись с строки: {start_row}")
    
    # 6. Обрабатываем каждый файл
    current_row = start_row
    success_count = 0
    
    for i, filepath in enumerate(transcript_files, 1):
        filename = os.path.basename(filepath)
        
        print(f"\n{'='*40}")
        print(f"ФАЙЛ {i}/{len(transcript_files)}: {filename}")
        print('='*40)
        
        try:
            # 6.1. Извлекаем информацию из имени файла
            file_info = parse_filename(filename)
            print(f"📅 Дата звонка: {file_info['date']}")
            print(f"👤 Клиент: {file_info['client_name']}")
            print(f"📄 Файл: {file_info['filename']}")
            
            # 6.2. Читаем содержимое файла
            content = read_transcript_file(filepath)
            if not content:
                print("⚠️ Файл пустой, пропускаем")
                log_upload(filename, success=False)
                continue
            
            print(f"📝 Символов в тексте: {len(content)}")
            
            # 6.3. Добавляем текст к данным
            file_info['content'] = content
            
            # 6.4. Загружаем в Google Таблицу
            if upload_to_sheets(file_info, current_row):
                # 6.5. Перемещаем в processed
                if move_to_processed(filename):
                    log_upload(filename, success=True)
                    print(f"✅ Файл '{filename}' успешно обработан в строке {current_row}!")
                    success_count += 1
                    current_row += 1
                else:
                    log_upload(filename, success=False)
            else:
                log_upload(filename, success=False)
                
        except Exception as e:
            print(f"❌ Критическая ошибка при обработке файла: {e}")
            log_upload(filename, success=False)
    
    print("\n" + "="*60)
    print("ОБРАБОТКА ЗАВЕРШЕНА!")
    print("="*60)
    
    # Статистика
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Обработано файлов: {len(transcript_files)}")
    print(f"   Успешно загружено: {success_count}")
    print(f"   Записи с {start_row} по {current_row - 1} строку")
    
    print(f"\n📁 Файлы логов: {LOG_FILE}")
    print(f"📁 Обработанные файлы: {PROCESSED_FOLDER}")
    
    print(f"\n🔗 Ссылка на таблицу:")
    print(f"   https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit")
    
    print(f"\n📋 СТРУКТУРА ТАБЛИЦЫ:")
    print(f"   A: Дата звонка")
    print(f"   B: Клиент")
    print(f"   C: Имя файла")
    print(f"   D: Текст транскрипции")
    print(f"   E: Дата загрузки")

if __name__ == "__main__":
    main()
    input("\nНажмите Enter для выхода...")