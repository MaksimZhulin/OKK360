# -*- coding: utf-8 -*-
"""
Модуль распознавания номенклатуры из текста звонка.

Гибрид, чтобы освещать ЛЮБУЮ номенклатуру и до размера:
    1) ИИ вытаскивает из транскрипции сырые упоминания товаров как есть
       ("двутавр 20", "арматура восьмёрка", "лист 3мм оцинковка").
    2) Матчер сопоставляет КАЖДОЕ упоминание:
         ОСНОВНОЙ источник — теги (data/tags.json, ~34к позиций с размером/маркой).
            Совпадение до размера: "арматура 8" -> тег "Арматура 8 мм".
         FALLBACK — каталог категорий (data/catalog.json, 3168), когда размер не
            назван или точного тега нет: "просто нужна арматура" -> "Арматура".
       Плюс словарь синонимов (снятие сленга) и слой нормализации.
    3) На выходе — тег с размером + верхняя категория, либо категория.

Публичные функции:
    match_mentions(list[str])      -> list[dict]   (основной вход из web_app)
    format_nomenclature(list[dict]) -> str
    extract_mentions_llm(text, client, model) -> list[str]
    analyze_nomenclature(text, client, model)  -> list[dict]
"""
import json
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache

_BASE = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(_BASE, "data", "catalog.json")
TAGS_PATH = os.path.join(_BASE, "data", "tags.json")
ALIASES_PATH = os.path.join(_BASE, "data", "nomenclature_aliases.json")

# Маркеры "специальных" веток. Если в упоминании их НЕТ, а в позиции они ЕСТЬ —
# штрафуем (чтобы "двутавр" не улетал в "двутавр нержавеющий"/"БУ").
SPECIALIZED_MARKERS = {
    "бу", "нержавеющий", "нержавеющая", "нержавеющее", "нержавейка", "нерж",
    "судовой", "судовая", "мостовая", "мостовой", "чугунный", "чугунная",
    "латунный", "латунная", "алюминиевый", "алюминиевая", "медный", "медная",
    "стеклопластиковая", "стеклопластиковый", "асбестоцементная", "полимерный",
    "оцинкованный", "оцинкованная",
}

# Разговорные размеры -> число (мм).
SIZE_WORDS = {
    "шестерка": "6", "восьмерка": "8", "десятка": "10", "двенашка": "12",
    "двенадцатая": "12", "четырнашка": "14", "шестнашка": "16",
    "двадцатка": "20", "двадцатая": "20",
}

# Сленг -> каноничные слова (перед матчингом по тегам).
SLANG = {
    "профтруба": "труба профильная", "проф труба": "труба профильная",
    "кругляк": "круг стальной", "оцинковка": "оцинкованный",
    "нержавейка": "нержавеющий", "нерж": "нержавеющий",
}

STOP_TOKENS = {"и", "в", "с", "на", "для", "из", "по", "от", "до", "мм", "м", "см", "шт", "тн", "т", "кг"}

# «Голые» слова-категории для fallback-словаря: применяем только если точного нет.
GENERIC_ALIAS_KEYS = {"труба", "лист", "листовой металл"}


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(norm_text: str):
    return [t for t in norm_text.split() if t and t not in STOP_TOKENS]


def content_tokens(norm_text: str):
    """Смысловые токены: слова длиной >=3 или буквенно-цифровые марки. Числа отбрасываем."""
    out = []
    for t in norm_text.split():
        if t in STOP_TOKENS or t.isdigit():
            continue
        if len(t) >= 3 or re.search(r"[a-zа-я]\d|\d[a-zа-я]", t):
            out.append(t)
    return out


def num_tokens(norm_text: str):
    """Числовые/размерные токены: '8', '40x20x2', '100х50х3'->'100x50x3'."""
    out = []
    for m in re.finditer(r"\d+(?:\s*[xх*]\s*\d+)+|\d+", norm_text):
        out.append(re.sub(r"\s*", "", m.group(0)).replace("*", "x").replace("х", "x"))
    return out


def _apply_slang(norm_text: str) -> str:
    t = norm_text
    for word, num in SIZE_WORDS.items():
        t = re.sub(rf"\b{word}\b", num, t)
    for slang, canon in SLANG.items():
        t = re.sub(rf"\b{re.escape(slang)}\b", canon, t)
    return t


def _extract_size(mention_norm: str):
    """Размер из упоминания: '40x20x2', '8'; сленг 'восьмерка'->8. Не берём цифры внутри марок."""
    nums = num_tokens(mention_norm)
    # предпочитаем составной размер (40x20x2), иначе первое одиночное число не внутри марки
    for n in nums:
        if "x" in n:
            return n
    m = re.search(r"(?<![0-9a-zа-я])\d{1,4}(?![0-9a-zа-я])", mention_norm)
    if m:
        return m.group(0)
    return nums[0] if nums else None


# ------------------------- индексы -------------------------

def _build_prefix_index(entries):
    idx = {}
    for i, e in enumerate(entries):
        for t in e["ctokens"]:
            idx.setdefault(t[:4], set()).add(i)
    return idx


@lru_cache(maxsize=1)
def _load_tags_index():
    if not os.path.exists(TAGS_PATH):
        return [], {}
    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        e.setdefault("ctokens", content_tokens(e.get("norm", "")))
        e.setdefault("nums", num_tokens(e.get("norm", "")))
    return entries, _build_prefix_index(entries)


@lru_cache(maxsize=1)
def _load_catalog_index():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    path_index = {}
    for e in entries:
        e["_ctokens"] = content_tokens(e["norm"])
        pkey = " / ".join(e["path"]).lower().replace("ё", "е")
        path_index[pkey] = e
    prefix_index = {}
    for i, e in enumerate(entries):
        for t in e["_ctokens"]:
            prefix_index.setdefault(t[:4], set()).add(i)
    return entries, path_index, prefix_index


@lru_cache(maxsize=1)
def _load_aliases():
    entries, path_index, _ = _load_catalog_index()
    resolved, unresolved = {}, []
    if os.path.exists(ALIASES_PATH):
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for alias, path_str in data.get("aliases", {}).items():
            key = path_str.lower().replace("ё", "е").strip()
            entry = path_index.get(key)
            if entry is None:
                last = normalize(path_str.split("/")[-1])
                entry = next((e for e in entries if e["norm"] == last), None)
            (resolved.__setitem__(normalize(alias), entry) if entry
             else unresolved.append((alias, path_str)))
    specific = {k: v for k, v in resolved.items() if k not in GENERIC_ALIAS_KEYS}
    generic = {k: v for k, v in resolved.items() if k in GENERIC_ALIAS_KEYS}
    so = sorted(specific.items(), key=lambda kv: -len(kv[0].split()))
    go = sorted(generic.items(), key=lambda kv: -len(kv[0].split()))
    return so, go, unresolved


# ------------------------- матчинг -------------------------

def _tok_match(a, b):
    if a == b:
        return True
    if len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.8


def _covered(a_tokens, b_tokens):
    if not a_tokens:
        return 0.0
    return sum(1 for at in a_tokens if any(_tok_match(at, bt) for bt in b_tokens)) / len(a_tokens)


def _product_score(m_ctokens, m_norm, e_ctokens, e_norm):
    if not e_ctokens or not m_ctokens:
        return 0.0
    token_score = _covered(m_ctokens, e_ctokens)   # что сказал клиент — есть в позиции
    entry_cov = _covered(e_ctokens, m_ctokens)      # позиция не добавляет лишнего
    ratio = SequenceMatcher(None, m_norm, e_norm).ratio()
    return 0.5 * token_score + 0.2 * entry_cov + 0.3 * ratio


def _match_tags(m_norm, m_ctokens, m_size, m_markers, threshold):
    entries, prefix_index = _load_tags_index()
    if not entries:
        return None
    cand = set()
    for t in m_ctokens:
        cand |= prefix_index.get(t[:4], set())
    if not cand:
        return None

    best, best_score = None, 0.0
    for i in cand:
        e = entries[i]
        base = _product_score(m_ctokens, m_norm, e["ctokens"], e["norm"])
        if base <= 0:
            continue
        # штраф за спецветку (нерж/бу/оцинк…), которой нет в упоминании.
        # Проверяем по ВСЕМ токенам имени (в т.ч. коротким, как "бу").
        e_all = set(e["norm"].split())
        if (e_all & SPECIALIZED_MARKERS) - m_markers:
            base *= 0.5
        # фактор размера
        tag_sizes = set(e.get("nums", []))
        if e.get("size"):
            tag_sizes.add(e["size"])
        if m_size:
            if m_size in tag_sizes:
                factor = 1.15
            elif tag_sizes:
                factor = 0.45           # у тега другой размер — скорее не то
            else:
                factor = 0.8            # тег без размера, а клиент назвал
        else:
            factor = 0.75 if tag_sizes else 1.0   # клиент без размера — тянем к общему тегу
        score = min(1.0, base * factor)
        if score > best_score:
            best, best_score = e, score

    if best and best_score >= threshold:
        return {
            "kind": "tag", "matched": True, "name": best["tag"],
            "tag": best["tag"], "category": best.get("l1", ""),
            "path": None, "score": round(best_score, 3), "method": "tag",
        }
    return None


def _match_catalog(m_norm, m_ctokens, m_markers, threshold):
    entries, _, prefix_index = _load_catalog_index()
    specific_aliases, generic_aliases, _ = _load_aliases()
    m_tokens = set(tokenize(m_norm))

    def _try(alias_list, method):
        for alias_norm, entry in alias_list:
            a_tokens = alias_norm.split()
            if all(any(SequenceMatcher(None, at, mt).ratio() >= 0.85 for mt in m_tokens) for at in a_tokens):
                if m_markers - set(entry["_ctokens"]):
                    continue
                return {
                    "kind": "category", "matched": True, "name": entry["name"],
                    "tag": None, "category": entry["l1"], "path": entry["path"],
                    "score": 1.0 if method == "alias" else 0.6, "method": method,
                }
        return None

    hit = _try(specific_aliases, "alias")
    if hit:
        return hit

    cand = set()
    for t in m_ctokens:
        cand |= prefix_index.get(t[:4], set())
    candidates = [entries[i] for i in cand] if cand else entries
    best, best_score = None, 0.0
    for e in candidates:
        s = _product_score(m_ctokens, m_norm, e["_ctokens"], e["norm"])
        if (set(e["norm"].split()) & SPECIALIZED_MARKERS) - m_markers:
            s *= 0.5
        if s > best_score or (abs(s - best_score) < 1e-9 and best and e["level"] < best["level"]):
            best, best_score = e, s
    if best and best_score >= threshold:
        return {
            "kind": "category", "matched": True, "name": best["name"],
            "tag": None, "category": best["l1"], "path": best["path"],
            "score": round(best_score, 3), "method": "fuzzy",
        }

    return _try(generic_aliases, "alias_generic")


def match_one(mention, tag_threshold=0.5, cat_threshold=0.55):
    raw = (mention or "").strip()
    m_norm = _apply_slang(normalize(raw))
    result = {
        "raw": raw, "matched": False, "kind": None,
        "name": None, "tag": None, "category": None, "path": None,
        "size": _extract_size(m_norm), "score": 0.0, "method": None,
    }
    if not m_norm:
        return result

    m_ctokens = content_tokens(m_norm)
    m_markers = set(tokenize(m_norm)) & SPECIALIZED_MARKERS
    m_size = result["size"]

    hit = _match_tags(m_norm, m_ctokens, m_size, m_markers, tag_threshold)
    if hit is None:
        hit = _match_catalog(m_norm, m_ctokens, m_markers, cat_threshold)
    if hit:
        result.update(hit)
    return result


def match_mentions(mentions, tag_threshold=0.5, cat_threshold=0.55):
    if not mentions:
        return []
    if isinstance(mentions, str):
        mentions = [mentions]
    out, seen = [], set()
    for m in mentions:
        r = match_one(m, tag_threshold=tag_threshold, cat_threshold=cat_threshold)
        key = (r["kind"], r["name"]) if r["matched"] else ("raw", normalize(r["raw"]))
        if not r["raw"] or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def format_nomenclature(items):
    """Читаемый вывод: если определён тег — сам тег; если только категория — категория;
    если не определить ничего — 'Не определена'.
    Пример: 'Заглушка желоба 200 мм; Стальная арматура'."""
    names, seen = [], set()
    for it in items:
        if not it.get("matched"):
            continue
        name = it.get("name")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return "; ".join(names) if names else "Не определена"


def extract_mentions_llm(transcript_text, client, model, local_mode=False):
    """Отдельный вызов ИИ (если номенклатуру не встроили в основной промпт)."""
    prompt = (
        "Из транскрипции звонка металлоторговой компании выпиши ВСЕ упоминания "
        "товаров/номенклатуры, которые интересуют клиента или обсуждаются в заказе.\n"
        "Пиши каждую позицию как звучит в разговоре, вместе с размером/маркой если названы "
        "(\"двутавр 20\", \"арматура 8 мм А500С\", \"лист 3мм оцинкованный\").\n"
        "Не выдумывай. Если товаров нет — верни пустой массив.\n"
        "Верни СТРОГО JSON-массив строк.\n\n"
        f"Текст:\n{transcript_text[:10000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты извлекаешь номенклатуру из звонков. Отвечаешь строго JSON-массивом строк."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1, max_tokens=500,
        )
        txt = resp.choices[0].message.content.strip()
        s, e = txt.find("["), txt.rfind("]")
        if s != -1 and e != -1:
            return [str(x) for x in json.loads(txt[s:e + 1]) if str(x).strip()]
    except Exception as ex:
        print(f"⚠️ extract_mentions_llm: {ex}")
    return []


def analyze_nomenclature(transcript_text, client, model, local_mode=False):
    raw = extract_mentions_llm(transcript_text, client, model, local_mode=local_mode)
    return match_mentions(raw)


if __name__ == "__main__":
    tests = [
        "двутавр 20", "двутавровая балка", "арматура 8 мм", "арматура восьмёрка",
        "уголок 50", "швеллер 12", "лист 3мм оцинкованный", "профильная труба 40х20х2",
        "круг стальной 20", "профнастил с8", "нержавеющий лист", "балка бу",
        "мне нужна арматура", "труба вгп 25", "магистральная труба 530",
        "гвозди строительные", "болт м12",
    ]
    tags, _ = _load_tags_index()
    cat, _, _ = _load_catalog_index()
    _, _, unresolved = _load_aliases()
    print(f"Тегов: {len(tags)} | Каталог: {len(cat)} | Неразрешённых алиасов: {len(unresolved)}")
    for u in unresolved:
        print("  ⚠️ алиас без пути:", u)
    print("-" * 70)
    for t in tests:
        r = match_one(t)
        if r["matched"]:
            print(f"{t:30} -> {r['name']}  ({r['category']})  [{r['method']}/{r['score']}]")
        else:
            print(f"{t:30} -> ❌ не распознано")
