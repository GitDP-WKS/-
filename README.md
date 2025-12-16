# Avaya CMS Shift Analyzer (Streamlit)

## Как запустить локально
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Для Streamlit Cloud
1. Залей эту папку в репозиторий GitHub.
2. В Streamlit Cloud выбери файл запуска: `app.py`
3. Убедись, что `requirements.txt` лежит рядом.

## Что нужно
- HTML отчёт Avaya CMS (Call Records)
- (Опционально) файл маппинга операторов CSV/XLSX (agent_code -> agent_name)
