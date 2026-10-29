# -*- coding: utf-8 -*-
"""
Лог прогона обработки. Всё, что пайплайн печатает (print модулей audio/llm_analysis,
предупреждения библиотек в stderr), на время прогона дублируется в файл
logs/run_<дата-время>.log с отметкой времени на каждой строке. Интерфейс показывает
только этапы, а подробности — в свёрнутом окне лога и в скачиваемом файле.
"""
import os
import sys
import threading
from collections import deque
from datetime import datetime

LOG_DIR = "logs"


class RunLog:
    def __init__(self, started=None, view_lines=400):
        started = started or datetime.now()
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, f"run_{started:%Y-%m-%d_%H-%M-%S}.log")
        self._file = open(self.path, "a", encoding="utf-8")
        self._tail = deque(maxlen=view_lines)  # хвост для окна в интерфейсе
        self._lock = threading.Lock()
        self._saved = None

    def add(self, line):
        """Записывает одну строку (в файл сразу, чтобы лог пережил падение приложения)."""
        line = f"[{datetime.now():%H:%M:%S}] {line.rstrip()}"
        with self._lock:
            self._tail.append(line)
            if not self._file.closed:
                self._file.write(line + "\n")
                self._file.flush()

    def tail(self):
        with self._lock:
            return "\n".join(self._tail)

    def __enter__(self):
        """Перехватывает stdout/stderr на время прогона (в консоль вывод тоже идёт)."""
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self)
        sys.stderr = _Tee(sys.stderr, self)
        return self

    def __exit__(self, exc_type, exc, tb):
        for tee in (sys.stdout, sys.stderr):
            if isinstance(tee, _Tee):
                tee.flush_pending()
        sys.stdout, sys.stderr = self._saved
        with self._lock:
            self._file.close()
        return False


class _Tee:
    """Поток-двойник: пишет в исходную консоль и построчно в RunLog.
    Возврат каретки (\\r) — как в терминале: строка перезаписывается, поэтому
    полосы прогресса tqdm попадают в лог одной итоговой строкой, а не сотнями."""

    def __init__(self, orig, log):
        self._orig = orig
        self._log = log
        self._buf = ""

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rsplit("\r", 1)[-1]
            if line.strip():
                self._log.add(line)
        if "\r" in self._buf:
            self._buf = self._buf.rsplit("\r", 1)[-1]
        return len(s)

    def flush_pending(self):
        if self._buf.strip():
            self._log.add(self._buf)
        self._buf = ""

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._orig, name)
