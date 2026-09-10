@echo off
chcp 65001 >nul
REM ============================================================
REM  Запуск Ollama БЕЗ системного прокси.
REM  Зачем: при активном HTTP(S)_PROXY Ollama 0.34 зависает на
REM  загрузке модели (сетевой вызов висит на прокси) -> 503 ->
REM  локальный анализ возвращает пустой результат.
REM  Этот скрипт поднимает локальный сервер Ollama с отключённым
REM  прокси и облаком. Запускать ПЕРЕД локальным режимом в приложении.
REM  Окно держать открытым (закрыть = остановить Ollama).
REM ============================================================
echo Останавливаю запущенный Ollama (трей/сервер)...
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM "ollama.exe" >nul 2>&1
timeout /t 2 >nul

set HTTP_PROXY=
set HTTPS_PROXY=
set NO_PROXY=*
set OLLAMA_NO_CLOUD=true
set OLLAMA_HOST=127.0.0.1:11434

echo Запускаю Ollama без прокси на 127.0.0.1:11434 ...
echo (оставьте это окно открытым, пока пользуетесь локальной моделью)
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
