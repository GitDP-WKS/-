
# app.py — минимальный “неубиваемый” анализатор Avaya Call Records (ночь/день)
# Принято = ANS + CONN
# Пропущено = ABAN (можно ограничить по skills 1/3/9)
#
# Streamlit Cloud: положи этот файл как app.py + requirements.txt

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


# -------------------- БАЗОВЫЕ НАСТРОЙКИ --------------------

WATCHED_SKILLS_DEFAULT = ["1", "3", "9"]
SKILL_NAMES = {"1": "Надежность", "3": "Качество э/э", "9": "ЭЗС"}

ACCEPTED = {"ANS", "CONN"}
MISSED = {"ABAN"}

NIGHT_START = (18, 30)
NIGHT_END = (6, 30)
DAY_START = (6, 30)
DAY_END = (18, 30)


@dataclass(frozen=True)
class ShiftSpec:
    name: str
    start_hm: Tuple[int, int]
    end_hm: Tuple[int, int]

    def bounds(self, base: date) -> Tuple[datetime, datetime]:
        s = datetime.combine(base, time(self.start_hm[0], self.start_hm[1]))
        e = datetime.combine(base, time(self.end_hm[0], self.end_hm[1]))
        if e <= s:  # через полночь
            e += timedelta(days=1)
        return s, e


SHIFT_NIGHT = ShiftSpec("Ночь (18:30–06:30)", NIGHT_START, NIGHT_END)
SHIFT_DAY = ShiftSpec("День (06:30–18:30)", DAY_START, DAY_END)


# -------------------- УТИЛИТЫ --------------------

def safe_decode(b: bytes) -> str:
    for enc in ("cp1251", "windows-1251", "utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def clean_txt(s: str) -> str:
    return (s or "").replace("\xa0", " ").strip()


def find_best_table(soup: BeautifulSoup):
    """Берём таблицу, которая больше всего похожа на Avaya Call Records."""
    tables = soup.find_all("table")
    if not tables:
        return None

    def score(tbl) -> int:
        th = [clean_txt(x.get_text(" ", strip=True)).lower() for x in tbl.find_all("th")]
        if not th:
            return -999
        keys = ["дата", "время", "размещение", "split/skill", "имена пользователей"]
        return sum(any(k in h for h in th) for k in keys) * 10 + len(th)

    return max(tables, key=score)


@st.cache_data(show_spinner=False)
def parse_html_to_raw_df(html_bytes: bytes) -> pd.DataFrame:
    try:
        html = safe_decode(html_bytes)
        soup = BeautifulSoup(html, "html.parser")
        tbl = find_best_table(soup)
        if tbl is None:
            return pd.DataFrame()

        headers = [clean_txt(th.get_text(" ", strip=True)) for th in tbl.find_all("th")]
        if not headers:
            return pd.DataFrame()

        rows = []
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            row = [clean_txt(td.get_text(" ", strip=True)) for td in tds]
            # выравниваем под headers
            if len(row) < len(headers):
                row += [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[: len(headers)]
            rows.append(row)

        return pd.DataFrame(rows, columns=headers)
    except Exception:
        return pd.DataFrame()


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Нормализация к единому формату: dt_start, disposition, skill_code, agent_code."""
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    col_date = pick_col(df, ["Дата", "Date"])
    col_time = pick_col(df, ["Время нач.", "Время нач", "Time"])
    col_disp = pick_col(df, ["Размещение", "Disposition"])
    col_skill = pick_col(df, ["Split/Skill", "Skill", "Split"])
    col_agent = pick_col(df, ["Имена пользователей", "Agent", "Пользователь", "User"])

    if not (col_date and col_time and col_disp):
        return pd.DataFrame()

    dt_str = (df[col_date].astype(str).str.strip() + " " + df[col_time].astype(str).str.strip()).str.strip()
    df["dt_start"] = pd.to_datetime(dt_str, format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt_start"]).reset_index(drop=True)

    df["disposition"] = df[col_disp].astype(str).str.strip().str.upper()
    df["skill_raw"] = "" if col_skill is None else df[col_skill].astype(str).str.strip()
    df["skill_code"] = df["skill_raw"].str.extract(r"(\d+)", expand=False).fillna("").astype(str)
    df["agent_code"] = "" if col_agent is None else df[col_agent].astype(str).str.strip()

    df["call_date"] = df["dt_start"].dt.date
    return df


def load_mapping(uploaded) -> Dict[str, str]:
    """CSV/XLSX: 1-я колонка = code, 2-я = name (или распознаём по заголовкам)."""
    if uploaded is None:
        return {}
    try:
        name = (getattr(uploaded, "name", "") or "").lower()
        data = uploaded.getvalue()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            mdf = pd.read_excel(io.BytesIO(data), dtype=str).fillna("")
        else:
            # CSV / txt
            mdf = None
            for sep in [",", ";", "\t"]:
                try:
                    tmp = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str).fillna("")
                    if tmp.shape[1] >= 2:
                        mdf = tmp
                        break
                except Exception:
                    pass
            if mdf is None:
                return {}

        cols = [c.lower().strip() for c in mdf.columns]
        code_i = None
        name_i = None
        for i, c in enumerate(cols):
            if code_i is None and any(k in c for k in ["код", "agent", "id", "номер", "code"]):
                code_i = i
            if name_i is None and any(k in c for k in ["имя", "фио", "name"]):
                name_i = i
        if code_i is None or name_i is None:
            code_i, name_i = 0, 1

        code_col = mdf.columns[code_i]
        name_col = mdf.columns[name_i]

        mp: Dict[str, str] = {}
        for _, r in mdf.iterrows():
            code = str(r[code_col]).strip()
            nm = str(r[name_col]).strip()
            if code and nm:
                mp[code] = nm
        return mp
    except Exception:
        return {}


def apply_mapping(df: pd.DataFrame, mp: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["agent_name"] = df["agent_code"].astype(str).map(mp).fillna(df["agent_code"].astype(str))
    return df


def shift_dates(df: pd.DataFrame, shift: ShiftSpec) -> List[date]:
    """Даты, где реально есть события внутри окна смены."""
    if df.empty:
        return []
    dates = sorted(set(df["call_date"].tolist()))
    out = []
    for d in dates:
        s, e = shift.bounds(d)
        m = (df["dt_start"] >= s) & (df["dt_start"] < e)
        if m.any():
            out.append(d)
    return out


def filter_shift(df: pd.DataFrame, shift: ShiftSpec, base: Optional[date]) -> Tuple[pd.DataFrame, Optional[Tuple[datetime, datetime]]]:
    if df.empty:
        return df, None

    if base is None:
        ds = shift_dates(df, shift)
        if not ds:
            return df.iloc[0:0].copy(), None
        parts = []
        for d in ds:
            s, e = shift.bounds(d)
            parts.append(df[(df["dt_start"] >= s) & (df["dt_start"] < e)])
        return pd.concat(parts, ignore_index=True), (shift.bounds(ds[0])[0], shift.bounds(ds[-1])[1])

    s, e = shift.bounds(base)
    return df[(df["dt_start"] >= s) & (df["dt_start"] < e)].copy().reset_index(drop=True), (s, e)


def compute(df: pd.DataFrame, watched_skills: List[str]) -> Dict[str, pd.DataFrame]:
    """Считаем понятные метрики. Принято=ANS+CONN."""
    out: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return {"events": df}

    d = df.copy()

    d["is_accepted"] = d["disposition"].isin(ACCEPTED)
    d["is_missed"] = d["disposition"].isin(MISSED)
    d["is_missed_watched"] = d["is_missed"] & d["skill_code"].isin(watched_skills)
    d["missed_no_agent"] = d["is_missed_watched"] & (d["agent_code"].str.strip() == "")
    d["missed_with_agent"] = d["is_missed_watched"] & (d["agent_code"].str.strip() != "")
    d["is_other"] = ~(d["disposition"].isin(ACCEPTED | MISSED))

    kpi = {
        "Принято (ANS+CONN)": int(d["is_accepted"].sum()),
        "Пропущено (ABAN, по выбранным skills)": int(d["is_missed_watched"].sum()),
        "  ├─ пропущено оператором": int(d["missed_with_agent"].sum()),
        "  └─ без оператора": int(d["missed_no_agent"].sum()),
        "Прочие статусы": int(d["is_other"].sum()),
        "Всего событий": int(len(d)),
    }
    denom = kpi["Принято (ANS+CONN)"] + kpi["Пропущено (ABAN, по выбранным skills)"]
    kpi["% пропущенных (ABAN/(ANS+CONN+ABAN))"] = 0.0 if denom == 0 else round(100.0 * kpi["Пропущено (ABAN, по выбранным skills)"] / denom, 2)
    out["kpi"] = pd.DataFrame([kpi])

    disp = (
        d.groupby("disposition", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    out["dispositions"] = disp

    # Топ операторов по пропущенным (только where agent известен)
    ops = d[d["agent_code"].str.strip() != ""].groupby("agent_name").agg(
        Принято=("is_accepted", "sum"),
        Пропущено=("is_missed_watched", "sum"),
        Всего=("disposition", "size")
    ).reset_index()
    denom2 = (ops["Принято"] + ops["Пропущено"]).replace(0, pd.NA)
    ops["% пропущенных"] = (100.0 * ops["Пропущено"] / denom2).fillna(0.0).round(2)
    ops = ops.sort_values(["Пропущено", "Принято"], ascending=[False, False])
    out["operators"] = ops

    # Тематики по пропущенным
    d["topic"] = d["skill_code"].map(SKILL_NAMES).fillna(d["skill_code"].replace("", "Без skill"))
    skills = d.groupby("topic").agg(
        Принято=("is_accepted", "sum"),
        Пропущено=("is_missed_watched", "sum"),
        Всего=("disposition", "size")
    ).reset_index().sort_values(["Пропущено", "Принято"], ascending=[False, False])
    out["topics"] = skills

    out["events"] = d.sort_values("dt_start").reset_index(drop=True)
    return out


# -------------------- UI --------------------

st.set_page_config(page_title="Avaya: ночь/день", layout="wide")
st.title("Avaya Call Records — анализ смен")

with st.sidebar:
    st.header("Файлы")
    html = st.file_uploader("HTML отчёт Avaya", type=["html", "htm"])
    mp_file = st.file_uploader("Маппинг операторов (CSV/XLSX) — опционально", type=["csv", "txt", "xlsx", "xls"])

    st.divider()
    st.header("Смена")
    mode = st.radio("Режим", ["Ночь", "День"], horizontal=True)
    shift = SHIFT_NIGHT if mode == "Ночь" else SHIFT_DAY

    st.divider()
    st.header("Фильтрация")
    watched_skills = st.multiselect("Skills, которые считать для ABAN", options=sorted(set(WATCHED_SKILLS_DEFAULT)), default=WATCHED_SKILLS_DEFAULT)

    st.divider()
    st.header("Отображение")
    top_n = st.slider("Топ N", 3, 30, 10)
    show_events = st.checkbox("Показать события (таблица)", value=False)

if html is None:
    st.info("Загрузи HTML отчёт слева.")
    st.stop()

raw = parse_html_to_raw_df(html.getvalue())
df = normalize(raw)
if df.empty:
    st.error("Не смог распарсить отчёт. Нужны колонки: Дата, Время нач., Размещение (и желательно Split/Skill, Имена пользователей).")
    st.stop()

mp = load_mapping(mp_file)
df = apply_mapping(df, mp)

# выбор дня (если в файле несколько дней)
dates = shift_dates(df, shift)
opts = ["Все смены в файле"] + [d.strftime("%d.%m.%Y") for d in dates]
chosen = st.selectbox("Дата смены (если в файле несколько дней):", options=opts, index=0)
base = None if chosen == "Все смены в файле" else datetime.strptime(chosen, "%d.%m.%Y").date()

df_shift, win = filter_shift(df, shift, base)
if df_shift.empty:
    st.warning("В выбранном окне смены нет событий.")
    st.stop()

metrics = compute(df_shift, watched_skills)

# Заголовок окна
if win:
    s, e = win
    st.subheader(f"Окно анализа: {s:%d.%m.%Y %H:%M} → {e:%d.%m.%Y %H:%M} ({shift.name})")

# KPI — сразу “сходится”
k = metrics["kpi"].iloc[0].to_dict()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Принято (ANS+CONN)", k["Принято (ANS+CONN)"])
c2.metric("Пропущено (ABAN)", k["Пропущено (ABAN, по выбранным skills)"])
c3.metric("Без оператора (ABAN)", k["  └─ без оператора"])
c4.metric("Прочие статусы", k["Прочие статусы"])
c5.metric("Всего событий", k["Всего событий"])

st.caption("Формула: % пропущенных = ABAN / (ANS+CONN+ABAN) по выбранным skills.")
st.metric("% пропущенных", f'{k["% пропущенных (ABAN/(ANS+CONN+ABAN))"]}%')

# Таблица по статусам — объясняет “почему всего больше”
with st.expander("Пояснение: из чего складывается 'Всего'"):
    st.write("**Всего событий** = Принято (ANS+CONN) + Пропущено (ABAN по выбранным skills) + Прочие статусы.")
    st.dataframe(metrics["dispositions"], use_container_width=True)

# Топы
colA, colB = st.columns(2)
with colA:
    st.markdown(f"### Топ-{top_n} операторов по пропущенным")
    st.dataframe(metrics["operators"].head(top_n), use_container_width=True)
with colB:
    st.markdown(f"### Топ-{top_n} тематик по пропущенным")
    st.dataframe(metrics["topics"].head(top_n), use_container_width=True)

# Экспорт
csv = metrics["events"].to_csv(index=False).encode("utf-8")
st.download_button("Скачать CSV событий (фильтрованная смена)", data=csv, file_name="avaya_shift_events.csv", mime="text/csv")

if show_events:
    st.markdown("### События (первые 2000 строк)")
    st.dataframe(metrics["events"].head(2000), use_container_width=True)
