# -*- coding: utf-8 -*-
"""
CLI: пакетная транскрибация записей из папки recordings/ в transcripts/*.txt —
тем же пайплайном, что и веб-приложение (Фаза 1): предобработка ffmpeg -> +2 с тишины
-> WhisperX large-v3 -> выравнивание -> диаризация pyannote -> роли Менеджер/Клиент.

Дальше транскрипты заливает в Google Sheets upload_transcripts.py.

Запуск:
    venv\\Scripts\\python process_calls.py --hf-token hf_xxx
    (или задать переменную окружения HF_TOKEN и запускать без аргументов)

Ключи:
    --input DIR        папка с записями (по умолчанию recordings)
    --output DIR       куда класть .txt (по умолчанию transcripts)
    --no-preprocess    без нормализации громкости (как снятая галка в веб-приложении)
"""
import argparse
import os
import sys
import time

# Windows: консоль по умолчанию cp1251 и падает на эмодзи в print() (их печатает и audio.py
# уже при импорте) — переключаем вывод в UTF-8 ДО импорта пайплайна.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

AUDIO_EXTS = (".mp3", ".wav", ".m4a")
# Временные файлы пайплайна (создаются рядом с исходником) — не путать с записями,
# если прошлый запуск упал и не успел их удалить.
TEMP_SUFFIXES = (".clean.wav", ".silence.wav")


def find_recordings(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if low.endswith(AUDIO_EXTS) and not low.endswith(TEMP_SUFFIXES):
            if os.path.isfile(os.path.join(folder, name)):
                files.append(name)
    return files


def transcribe_file(path, hf_token, preprocess):
    """Транскрибирует один файл. Возвращает текст с ролями; при ошибке — исключение."""
    from audio import preprocess_audio, pad_audio_simple_silence, transcribe_with_whisperx_diarization

    temp_files = []
    try:
        src = path
        if preprocess:
            cleaned = preprocess_audio(path)
            if cleaned != path:
                temp_files.append(cleaned)
                src = cleaned
        padded = pad_audio_simple_silence(src)
        if padded != src:
            temp_files.append(padded)

        text = transcribe_with_whisperx_diarization(padded, hf_token)
        if text.startswith("❌"):  # audio.py возвращает ошибку строкой
            raise RuntimeError(text)
        return text
    finally:
        for tmp in temp_files:
            try:
                os.remove(tmp)
            except OSError as e:
                print(f"   ⚠️ Не удалось удалить временный файл {tmp}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Пакетная транскрибация звонков (WhisperX + диаризация)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="Hugging Face Token для диаризации (или переменная окружения HF_TOKEN)")
    parser.add_argument("--input", default="recordings", help="папка с записями")
    parser.add_argument("--output", default="transcripts", help="папка для .txt")
    parser.add_argument("--no-preprocess", action="store_true", help="без нормализации громкости")
    args = parser.parse_args()

    print("=" * 50)
    print("ПАКЕТНАЯ ТРАНСКРИБАЦИЯ (WhisperX + диаризация)")
    print("=" * 50)

    if not args.hf_token:
        print("❌ Нет Hugging Face Token: передайте --hf-token или задайте HF_TOKEN")
        return 2
    if not os.path.isdir(args.input):
        print(f"❌ Папки '{args.input}' нет — положите туда записи MP3/WAV/M4A")
        return 2

    files = find_recordings(args.input)
    if not files:
        print(f"❌ В '{args.input}' нет файлов MP3/WAV/M4A")
        return 2
    print(f"📁 Найдено файлов: {len(files)}")

    from audio import WHISPERX_AVAILABLE
    if not WHISPERX_AVAILABLE:
        print("❌ WhisperX не установлен — venv\\Scripts\\pip install -r requirements.txt")
        return 2

    os.makedirs(args.output, exist_ok=True)
    ok, failed = 0, 0
    batch_start = time.time()

    for i, name in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {name}")
        t0 = time.time()
        try:
            text = transcribe_file(os.path.join(args.input, name), args.hf_token,
                                   preprocess=not args.no_preprocess)
            # Имя "<файл>.mp3.txt" — такой формат разбирает upload_transcripts.py
            out_path = os.path.join(args.output, f"{name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            ok += 1
            print(f"   ✅ Сохранено: {out_path} ({time.time() - t0:.0f} сек)")
            print(f"   Текст: {text[:80]}...")
        except Exception as e:
            failed += 1
            print(f"   ❌ Ошибка: {str(e)[:500]}")

    print("\n" + "=" * 50)
    print(f"ГОТОВО за {time.time() - batch_start:.0f} сек: успешно {ok}, ошибок {failed}. "
          f"Результаты в папке '{args.output}'")
    print("=" * 50)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
