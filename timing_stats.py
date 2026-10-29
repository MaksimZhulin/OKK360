# -*- coding: utf-8 -*-
"""
Замеры времени обработки и прогноз длительности прогона.

После каждого файла в logs/timing_stats.json пишется замер: сколько секунд заняла
транскрибация / анализ при такой-то длительности аудио и таких-то настройках.
Прогноз строится по замерам ИМЕННО этого компьютера и ИМЕННО этих настроек:
время файла ≈ a + b × длительность (a — постоянная часть на файл, b — секунд обработки
на секунду аудио). Пока замеров нет — стартовые значения из прошлых бенчмарков,
и это честно помечается как грубая оценка.
"""
import os
import json
import statistics
import threading

STATS_FILE = os.path.join("logs", "timing_stats.json")
MAX_SAMPLES = 300      # на каждую комбинацию настроек храним последние N замеров
MIN_FOR_FIT = 5        # с этого числа замеров — своя линейная модель вместо стартовой

# Стартовые значения (секунды): a — на файл, b — на секунду аудио. Нужны только для
# настроек, по которым своих замеров ещё нет (например, новая облачная модель).
# WhisperX: прогоны 2026-10-29 на RTX 4070 — 0.10–0.12 с на секунду аудио, загрузка ~15–20 с.
# Облако: gemini-2.5-flash через AgentPlatform, 2026-10-29 — 12–29 с на звонок (3 вызова).
# Qwen3-14B локально: ~17 мин на звонок средней длины (бенчмарк 2026-10-25).
# Яндекс — без замеров, ориентировочно.
DEFAULTS = {
    "transcribe|whisperx": (0.0, 0.12),
    "transcribe|yandex": (15.0, 0.30),
    "analyze|cloud": (8.0, 0.034),
    "analyze|local": (420.0, 1.5),
    "load|whisperx": (18.0, 0.0),     # загрузка моделей WhisperX в начале прогона
}

_lock = threading.Lock()


def transcribe_key(cfg):
    method = "whisperx" if "WhisperX" in cfg["transcription_method"] else "yandex"
    pre = int(bool(cfg.get("preprocess", True))) if method == "whisperx" else 0
    return f"transcribe|{method}|pre={pre}"


def analyze_key(cfg):
    return f"analyze|{'local' if cfg['local_mode'] else 'cloud'}|{cfg['analysis_model']}"


def _default_for(key):
    # transcribe|whisperx|pre=1 -> transcribe|whisperx; analyze|cloud|gemini -> analyze|cloud
    return DEFAULTS["|".join(key.split("|")[:2])]


def _load():
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record(key, duration, seconds):
    """Добавляет замер (длительность аудио, секунды обработки). Аудио короче секунды
    (битый файл, не определилась длительность) не пишем — такой замер ломал бы модель."""
    if not duration or duration < 1 or seconds is None or seconds <= 0:
        return
    with _lock:
        data = _load()
        samples = data.setdefault(key, [])
        samples.append([round(duration, 1), round(seconds, 1)])
        del samples[:-MAX_SAMPLES]
        os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
        tmp = STATS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, STATS_FILE)


class Rate:
    """Модель времени одного этапа: seconds ≈ a + b × duration."""

    def __init__(self, key, data=None):
        data = _load() if data is None else data
        self.key = key
        a0, b0 = _default_for(key)
        self.a, self.b = a0, b0
        try:
            samples = [(float(d), float(s)) for d, s in data.get(key, []) if d >= 1 and s > 0]
        except (TypeError, ValueError):
            samples = []  # испорченный файл замеров — работаем на стартовых значениях
        self.n = len(samples)
        if self.n >= MIN_FOR_FIT:
            self.a, self.b = _fit(samples, a0, b0)
        elif self.n:
            # мало замеров — масштабируем стартовую модель под фактическую скорость
            # (отношение сумм: точно попадает в суммарное время пачки)
            k = sum(s for _, s in samples) / sum(a0 + b0 * d for d, _ in samples)
            self.a, self.b = a0 * k, b0 * k

    def predict(self, duration):
        return self.a + self.b * (duration or 0.0)

    @property
    def measured(self):
        return self.n > 0


def _fit(samples, a0, b0):
    """МНК по последним замерам. Если прямая не получается (все файлы одной длины,
    или разброс скорости дал отрицательный коэффициент — так бывает, когда первые
    файлы прогона идут медленнее остальных), берём «секунд на секунду аудио» как
    отношение сумм: такая модель точно попадает в суммарное время пачки."""
    xs = [d for d, _ in samples]
    ys = [s for _, s in samples]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    if vx > 0:
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
        a = my - b * mx
        if a >= 0 and b >= 0:
            return a, b
    if b0 == 0:  # этап без зависимости от длины (загрузка моделей) — среднее время
        return my, 0.0
    return 0.0, sum(ys) / sum(xs)


def estimate(durations, cfg):
    """Прогноз всего прогона до старта. durations — секунды аудио по файлам.
    Возвращает словарь секунд + сколько замеров легло в основу."""
    data = _load()
    t_rate = Rate(transcribe_key(cfg), data)
    a_rate = Rate(analyze_key(cfg), data)
    load = 0.0
    if "WhisperX" in cfg["transcription_method"]:
        load = Rate("load|whisperx", data).predict(0)
    transcribe = sum(t_rate.predict(d) for d in durations)
    analyze = sum(a_rate.predict(d) for d in durations)
    return {
        "load": load, "transcribe": transcribe, "analyze": analyze,
        "total": load + transcribe + analyze,
        "n_transcribe": t_rate.n, "n_analyze": a_rate.n,
    }
