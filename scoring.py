# -*- coding: utf-8 -*-
"""
Критерии оценки звонка и подсчёт баллов.
Чистая логика (без Streamlit и внешних сервисов) — вынесено из web_app.py.
"""

# === Критерии блока "ПОТРЕБНОСТЬ" (выявление ситуации клиента) ===
# (ключ для JSON/таблицы, подпись для интерфейса)
NEED_CRITERIA = [
    ("need_purpose",            "Узнал цель покупки клиента"),
    ("need_project_details",    "Узнал детали проекта клиента"),
    ("need_geography",          "Узнал географию работ / объект"),
    ("need_supplier_criteria",  "Узнал критерии выбора поставщика"),
    ("need_interaction_terms",  "Узнал условия взаимодействия"),
    ("need_purchase_frequency", "Узнал частоту закупок"),
    ("need_competitors",        "Выявил конкурентов / предложения"),
    ("need_other_projects",     "Узнал о других проектах клиента"),
]
NEED_KEYS = [k for k, _ in NEED_CRITERIA]

# === Критерии блока "ВОЗРАЖЕНИЯ" (отработка возражений клиента) ===
OBJECTION_CRITERIA = [
    ("obj_active_listening",  "Активное слушание (принял сторону клиента)"),
    ("obj_no_interrupt",      "Не перебивал, дал выразить мысль"),
    ("obj_no_argue",          "Не спорил с клиентом"),
    ("obj_clarify_reason",    "Уточнял причину возражения"),
    ("obj_direct_answer",     "Прямо ответил на сомнения"),
    ("obj_arguments",         "Привёл аргументы и контраргументы"),
    ("obj_leading_questions", "Наводящие вопросы (клиент сам закрыл сомнения)"),
]
OBJ_KEYS = [k for k, _ in OBJECTION_CRITERIA]

# === Критерии блока "ДОЖИМ" (закрытие сделки / дожатие клиента) ===
DOZHIM_CRITERIA = [
    ("dozhim_concrete_solution", "Предложил конкретное решение после закрытия возражений"),
    ("dozhim_action_plan",       "Предложил план действий и вовлёк клиента"),
    ("dozhim_detailed_offer",    "Детализировал предложение под потребности клиента"),
    ("dozhim_no_pressure",       "Не давил, не создавал стресс"),
    ("dozhim_alternative",       "Предложил альтернативное решение"),
    ("dozhim_better_terms",      "Предложил условия лучше озвученных ранее"),
    ("dozhim_scarcity",          "Создал ограничение по времени/составу (дефицит)"),
    ("dozhim_upsell",            "Предложил дополнительные услуги"),
]
DOZHIM_KEYS = [k for k, _ in DOZHIM_CRITERIA]

# === Блок "КОНТАКТНЫЕ ДАННЫЕ / Кл-счёт" (обмен контактами с клиентом) ===
CONTACT_CRITERIA = [
    ("contact_preferred_channel", "Узнал предпочитаемый способ связи"),
    ("contact_email",             "Узнал/уточнил контактную почту"),
    ("contact_phone",             "Узнал/уточнил контактный телефон"),
    ("contact_other_person",      "Узнал иное контактное лицо для связи"),
    ("contact_convenient_time",   "Узнал удобное время связи (часовой пояс)"),
    ("contact_additional",        "Узнал дополнительные контактные данные"),
]
CONTACT_KEYS = [k for k, _ in CONTACT_CRITERIA]

# === Блок "СЛЕДУЮЩИЙ ШАГ / ЗАВЕРШЕНИЕ" (закрытие звонка) ===
NEXTSTEP_CRITERIA = [
    ("next_fixed_agreement", "Зафиксировал договорённость / промежуточный результат"),
    ("next_time_set",        "Установил конкретные дату и время след. контакта"),
    ("next_own_action",      "Чётко обозначил своё следующее действие"),
    ("next_result_details",  "Обозначил характеристики результата (товары в счёте)"),
    ("next_benefits",        "Обозначил преимущества следующего шага"),
    ("next_polite_close",    "Вежливо завершил диалог"),
]
NEXTSTEP_KEYS = [k for k, _ in NEXTSTEP_CRITERIA]

# === Блок "РЕЧЬ" (качество речи менеджера) ===
SPEECH_CRITERIA = [
    ("speech_literacy", "Грамотность: логичное, последовательное изложение"),
    ("speech_empathy",  "Эмпатия: фразы активного слушания (понимаю/согласен/верно)"),
]
SPEECH_KEYS = [k for k, _ in SPEECH_CRITERIA]


def block_score_1_5(analysis, keys):
    """Балл блока 1-5: 0 действий -> 1, далее = число выполненных действий, потолок 5."""
    count = sum(int(analysis.get(k, 0) or 0) for k in keys)
    return max(1, min(count, 5))


# Базовые критерии (ключ, подпись) — для рекомендаций
BASE_CRITERIA_LABELS = [
    ("establishing_contact", "Установление контакта"),
    ("client_type", "Определение типа клиента (физ/юр)"),
    ("clarifying_questions", "Уточняющие вопросы"),
    ("knowledge_quality", "Качество консультации"),
    ("software_proficiency", "Работа в программах"),
    ("politeness", "Вежливость"),
]
# Блоки 1-5: (название, ключи, что улучшить при низком балле)
REC_BLOCKS = [
    ("Потребность", NEED_KEYS, "глубже выявлять ситуацию клиента — цель покупки, детали проекта, критерии выбора"),
    ("Возражения", OBJ_KEYS, "отрабатывать возражения — уточнять причину, аргументировать, не спорить"),
    ("Дожим", DOZHIM_KEYS, "активнее закрывать сделку — предлагать решение, план, выгоды"),
    ("Кл/счёт, контакты", CONTACT_KEYS, "полнее собирать контакты — почта, телефон, удобное время, ЛПР"),
    ("Следующий шаг", NEXTSTEP_KEYS, "фиксировать договорённость и назначать конкретный следующий шаг"),
    ("Речь", SPEECH_KEYS, "следить за грамотностью и использовать фразы эмпатии"),
]
GRAND_MAX = 33  # 6 базовых + Потребность/Возражения/Дожим/Кл-счёт/Шаг по 5 + Речь 2


def build_recommendation(analysis):
    """Детерминированная рекомендация: вердикт по % от максимума + конкретные слабые зоны."""
    if str(analysis.get("technical_issue", "0")).strip() == "1":
        return "Оценка не объективна из-за брака связи / обрыва звонка — звонок исключён из статистики."

    had_obj = int(analysis.get("had_objections", 1) or 0) == 1
    grand = sum(int(analysis.get(k, 0) or 0) for k, _ in BASE_CRITERIA_LABELS)
    weak = [label for k, label in BASE_CRITERIA_LABELS if int(analysis.get(k, 0) or 0) == 0]

    max_total = GRAND_MAX
    for name, keys, advice in REC_BLOCKS:
        if name == "Возражения" and not had_obj:
            max_total -= 5  # возражений не было — блок не применим
            continue
        sc = block_score_1_5(analysis, keys)
        grand += sc
        if sc <= 1:
            weak.append(f"{name} ({advice})")

    pct = round(grand / max_total * 100) if max_total else 0
    if pct < 30:
        verdict = "🔴 Критично"
    elif pct < 50:
        verdict = "🟠 Слабо"
    elif pct < 70:
        verdict = "🟡 Средне"
    elif pct < 85:
        verdict = "🟢 Хорошо"
    else:
        verdict = "🟢 Отлично"

    rec = f"{verdict} — {grand}/{max_total} ({pct}%)."
    rec += (" Зоны роста: " + "; ".join(weak) + ".") if weak else " Слабых зон не выявлено."

    note = str(analysis.get("recommendations", "")).strip()
    if note and note not in ("Не определено", "0") and len(note) > 5:
        rec += f" Комментарий ИИ: {note}"
    return rec
