# -*- coding: utf-8 -*-
"""
Строит теговый индекс номенклатуры из CSV с тегами (34к позиций).

Вход:  CSV с колонками:
    Категория2, Категория3, Категория4, Наименование фильтра,
    Значение фильтра, Тег, H1
    H1 (колонка G) — финальное имя тега с размером/маркой ("Арматура 8 мм").

Выход: data/tags.json — плоский список тегов:
    {
      "tag":   "Арматура 8 мм",
      "norm":  "арматура 8 мм",
      "ctokens": ["арматура"],          # смысловые слова (без чисел)
      "nums":  ["8"],                    # числовые/размерные токены из имени
      "size":  "8",                      # чистый размер из фильтра D/E (если есть)
      "cat2":  "Арматура",               # категория из файла тегов
      "l1":    "Арматурный прокат"        # верхняя категория каталога (единый язык)
    }

Запуск:  py build_tags.py  [путь_к_csv]
"""
import csv
import json
import os
import re
import sys
from difflib import SequenceMatcher

_BASE = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(_BASE, "data", "catalog.json")
CAT2MAP_PATH = os.path.join(_BASE, "data", "cat2_to_l1.json")
OUT_PATH = os.path.join(_BASE, "data", "tags.json")
DEFAULT_CSV = os.path.join(
    os.path.expanduser("~"), "Downloads",
    "Копия подфильтровые_и_теговые_ссылки - Теги.csv"
)

DIMENSION_HINTS = ("диаметр", "размер", "толщин", "длин", "ширин", "сечени", "ду", "проход")
STOP = {"и", "в", "с", "на", "для", "из", "по", "от", "до", "мм", "м", "см", "шт", "тн", "т", "кг"}


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(norm_text: str):
    out = []
    for t in norm_text.split():
        if t in STOP or t.isdigit():
            continue
        if len(t) >= 3 or re.search(r"[a-zа-я]\d|\d[a-zа-я]", t):
            out.append(t)
    return out


def num_tokens(norm_text: str):
    """Числовые/размерные токены: '8', '20', '40x20x2', '100х50х3'->'100x50x3'."""
    out = []
    for m in re.finditer(r"\d+(?:\s*[xх*]\s*\d+)+|\d+", norm_text):
        out.append(re.sub(r"\s*", "", m.group(0)).replace("*", "x").replace("х", "x"))
    return out


def clean_size(filter_name: str, filter_value: str):
    """Чистый размер из фильтра: если D — размерность, берём значение E."""
    d = (filter_name or "").lower()
    if not any(h in d for h in DIMENSION_HINTS):
        return None
    v = normalize(filter_value)
    nums = num_tokens(v)
    return nums[0] if nums else None


# ---------- сопоставление cat2 -> верхняя категория каталога ----------

def load_catalog_l1():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    named = [(normalize(e["name"]), e["l1"]) for e in entries]
    l1_names = {normalize(e["l1"]): e["l1"] for e in entries}
    # быстрый индекс: токен -> {l1: вес}. Вес токена ~ 1/частота (реже = важнее).
    from collections import Counter
    df = Counter()
    per_entry = []
    for e in entries:
        toks = set(content_tokens(normalize(e["name"])))
        per_entry.append((toks, e["l1"]))
        for t in toks:
            df[t] += 1
    token_vote = {}
    for toks, l1 in per_entry:
        for t in toks:
            w = 1.0 / df[t]
            token_vote.setdefault(t, {}).setdefault(l1, 0.0)
            token_vote[t][l1] += w
    return named, l1_names, token_vote


def resolve_l1_fuzzy(text, named, l1_names, cache):
    """Точный/нечёткий поиск (медленно) — только для 43 значений Категория2."""
    key = normalize(text)
    if not key:
        return ""
    if key in cache:
        return cache[key]
    if key in l1_names:
        cache[key] = l1_names[key]
        return cache[key]
    best_l1, best = "", 0.0
    for nm, l1 in named:
        r = SequenceMatcher(None, key, nm).ratio()
        if r > best:
            best, best_l1 = r, l1
    cache[key] = best_l1 if best >= 0.5 else ""
    return cache[key]


def resolve_l1_tokens(ctokens, token_vote):
    """Быстрый поиск l1 по токенам (голосование) — для тегов без Категория2."""
    scores = {}
    for t in ctokens:
        for l1, w in token_vote.get(t, {}).items():
            scores[l1] = scores.get(l1, 0.0) + w
    if not scores:
        return ""
    return max(scores.items(), key=lambda kv: kv[1])[0]


def load_cat2_map():
    if not os.path.exists(CAT2MAP_PATH):
        return {}
    with open(CAT2MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("map", {})


def build(csv_path):
    named, l1_names, token_vote = load_catalog_l1()
    cat2_map = load_cat2_map()
    l1_cache = {}
    entries = []
    seen = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            row = [(c or "").strip() for c in row]
            if len(row) < 7:
                row += [""] * (7 - len(row))
            cat2, cat3, cat4, fname, fval, tag, h1 = row[:7]
            name = h1 or tag
            if not name or name in seen:
                continue
            seen.add(name)

            norm = normalize(name)
            ctoks = content_tokens(norm)
            # l1: 1) ручная карта cat2; 2) нечётко по cat2; 3) по токенам имени (быстро)
            l1 = ""
            if cat2:
                l1 = cat2_map.get(cat2) or resolve_l1_fuzzy(cat2, named, l1_names, l1_cache)
            if not l1:
                l1 = resolve_l1_tokens(ctoks, token_vote)

            entries.append({
                "tag": name,
                "norm": norm,
                "ctokens": ctoks,
                "nums": num_tokens(norm),
                "size": clean_size(fname, fval),
                "cat2": cat2,
                "l1": l1,
            })
    return entries


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv_path):
        print(f"❌ CSV не найден: {csv_path}")
        sys.exit(1)
    if not os.path.exists(CATALOG_PATH):
        print(f"❌ Сначала собери каталог: py build_catalog.py  (нет {CATALOG_PATH})")
        sys.exit(1)

    print(f"📥 Читаю теги: {csv_path}")
    entries = build(csv_path)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)

    no_l1 = sum(1 for e in entries if not e["l1"])
    with_size = sum(1 for e in entries if e["size"])
    print(f"✅ Теги собраны: {len(entries)} → {OUT_PATH}")
    print(f"   С чистым размером (фильтр D/E): {with_size}")
    print(f"   Без сопоставленной категории l1: {no_l1}")

    # карта cat2 -> l1 для визуальной проверки
    seen_map = {}
    for e in entries:
        if e["cat2"] and e["cat2"] not in seen_map:
            seen_map[e["cat2"]] = e["l1"]
    print("\n   Карта Категория2 (файл тегов) -> Верхняя категория каталога:")
    for k in sorted(seen_map):
        print(f"     {k:38} -> {seen_map[k]}")


if __name__ == "__main__":
    main()
