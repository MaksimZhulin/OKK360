# -*- coding: utf-8 -*-
"""
Единая конфигурация проекта. Раньше эти значения дублировались в web_app.py,
analyze_transcripts.py, upload_transcripts.py, llm_analysis.py.
"""

# Google Sheets
SPREADSHEET_ID = "1Oe-dKF_0oPhCdlwcj6jeco7BSIBi37jPuO3rSG4C930"
CREDENTIALS_FILE = "credentials.json"
SHEET_TITLE = "Выгрузка из проекта"       # чистое имя листа (для сравнения title)
SHEET_NAME = f"'{SHEET_TITLE}'"           # имя в кавычках — для A1-нотации диапазонов

# LLM-прокси (tokengate / LiteLLM, OpenAI-совместимый эндпоинт)
LLM_BASE_URL = "https://litellm.tokengate.ru/v1"
