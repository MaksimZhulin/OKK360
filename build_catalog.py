# -*- coding: utf-8 -*-
"""
Строит канонический каталог номенклатуры из CSV-структуры.

Вход:  CSV с колонками Категория1..Категория5 (иерархия сверху вниз).
Выход: data/catalog.json — плоский список узлов каталога:
    {
      "id":    "cat_00042",              # стабильный id
      "l1":    "Арматурный прокат",       # верхняя категория (для группировки)
      "level": 3,                          # глубина (сколько уровней заполнено)
      "name":  "Арматура А500С",           # самое глубокое непустое значение
      "path":  ["Арматурный прокат", "Стальная арматура", "Арматура А500С"],
      "norm":  "арматура а500с",           # нормализованное имя (для матчинга)
      "tokens":["арматура","а500с"]        # токены нормализованного имени
    }

Запуск:  py build_catalog.py  [путь_к_csv]
По умолчанию берёт "Структура - Лист1.csv" из папки Downloads текущего пользователя.
"""
import csv
import json
import os
import re
import sys

DEFAULT_CSV = os.path.join(
    os.path.expanduser("~"), "Downloads", "Структура - Лист1.csv"
)
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog.json")


def normalize(text: str) -> str:
    """Приводит строку к виду для матчинга: нижний регистр, ё->е,
    буквы/цифры остаются, всё остальное -> пробел, пробелы схлопываются."""
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    # оставляем русские/латинские буквы и цифры, остальное в пробел
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(norm_text: str):
    """Токены нормализованной строки без пустышек и однобуквенных предлогов."""
    stop = {"и", "в", "с", "на", "для", "из", "по", "от", "до"}
    return [t for t in norm_text.split() if t and t not in stop]


def build(csv_path: str):
    entries = []
    seen_paths = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # пропускаем строку заголовка
        for row in reader:
            # чистим ячейки
            cells = [(c or "").strip() for c in row]
            # путь = все непустые ячейки подряд слева направо
            path = [c for c in cells if c]
            if not path:
                continue
            path_key = " / ".join(path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            name = path[-1]
            norm = normalize(name)
            entries.append({
                "l1": path[0],
                "level": len(path),
                "name": name,
                "path": path,
                "norm": norm,
                "tokens": tokenize(norm),
            })

    # стабильные id по порядку
    for i, e in enumerate(entries):
        e["id"] = f"cat_{i:05d}"

    # переносим id в начало для читаемости
    entries = [{"id": e["id"], **{k: v for k, v in e.items() if k != "id"}} for e in entries]
    return entries


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv_path):
        print(f"❌ CSV не найден: {csv_path}")
        sys.exit(1)

    print(f"📥 Читаю каталог: {csv_path}")
    entries = build(csv_path)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # статистика
    by_level = {}
    l1 = set()
    for e in entries:
        by_level[e["level"]] = by_level.get(e["level"], 0) + 1
        l1.add(e["l1"])

    print(f"✅ Каталог собран: {len(entries)} позиций → {OUT_PATH}")
    print(f"   Верхних категорий: {len(l1)}")
    for lvl in sorted(by_level):
        print(f"   Уровень {lvl}: {by_level[lvl]} позиций")


if __name__ == "__main__":
    main()
