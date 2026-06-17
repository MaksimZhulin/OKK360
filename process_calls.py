import whisper
import os

print("=" * 40)
print("ОБРАБОТКА ЗАПИСЕЙ WHISPER")
print("=" * 40)

print("\n1. Загружаю модель...")
model = whisper.load_model("small")
print("   Модель загружена")

print("\n2. Проверяю записи...")

# Проверяем папку recordings
if not os.path.exists("recordings"):
    print("   ОШИБКА: Папки 'recordings' нет!")
    print("   Создайте папку 'recordings' и положите туда MP3 файлы")
    input("\nНажмите Enter...")
    exit()

# Ищем файлы
files = []
for f in os.listdir("recordings"):
    if f.lower().endswith(".mp3") or f.lower().endswith(".wav"):
        files.append(f)

if len(files) == 0:
    print("   Нет MP3/WAV файлов")
    print("   Положите записи в папку 'recordings'")
    input("\nНажмите Enter...")
    exit()

print(f"   Найдено файлов: {len(files)}")

print("\n3. Создаю папку для результатов...")
os.makedirs("transcripts", exist_ok=True)

print("\n4. Обрабатываю записи...")
for i, f in enumerate(files, 1):
    print(f"\n   [{i}/{len(files)}] Файл: {f}")
    
    try:
        # Транскрибируем
        result = model.transcribe(f"recordings/{f}", language="ru")
        
        # Сохраняем
        with open(f"transcripts/{f}.txt", "w", encoding="utf-8") as file:
            file.write(result["text"])
        
        print(f"   ✓ Сохранено: transcripts/{f}.txt")
        print(f"   Текст: {result['text'][:80]}...")
        
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

print("\n" + "=" * 40)
print("ГОТОВО! Результаты в папке 'transcripts'")
print("=" * 40)

input("\nНажмите Enter для выхода...")