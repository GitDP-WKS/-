
# app_shift_unified_v3.py
# Streamlit: универсальный анализатор Avaya CMS Call Records (ночь/день/мульти-дни)
#
# Запуск:
#   pip install streamlit pandas beautifulsoup4 openpyxl lxml
#   streamlit run app_shift_unified_v3.py
#
# Основное:
# - Ночь/День кнопками (ночь по умолчанию, как было)
# - Если в HTML несколько дней: появляется выбор "дата смены" или "все смены в файле"
# - Маппинг операторов (код->имя): встроенный + загрузка CSV/XLSX + вставка текста из реестра
# - Максимально "неубиваемый": все опасные места защищены, pivot reindex (без KeyError)

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


# -------------------------
# Встроенный маппинг (из твоих скринов)
# Только "Оператор - ... (КОД)" — номера станций/телефонов не нужны
# -------------------------
DEFAULT_AGENT_MAP: Dict[str, str] = {
    "7599449": "Абусаитов ДМ",
    "7599415": "Аспенбетова АА",
    "7599497": "Ахметзянова РР",
    "7599437": "Воронцов ВВ",
    "7599458": "Гайфуллина ДИ",
    "7599473": "Галиева АР",
    "7599413": "Гараев РР",
    "7599498": "Заббаров АИ",
    "7599405": "Зайнудтинова ЛС",
    "7599411": "Ибрагимова ЛИ",
    "7599403": "Минибаева АИ",
    "7599408": "Сагетдинов МИ",
    "7599478": "Хузахметов АР",
}

WATCHED_SKILLS_DEFAULT = ["1", "3", "9"]
SKILL_NAMES_DEFAULT = {"1": "Надежность", "3": "Качество э/э", "9": "ЭЗС"}

NIGHT_START = (18, 30)
NIGHT_END = (6, 30)
DAY_START = (6, 30)
DAY_END = (18, 30)

DEFAULT_LEFTOVER_MIN = 60

DISP_ACCEPTED = {"ANS"}     # можно добавить CONN галкой
DISP_MISSED = {"ABAN"}      # пропущенные


@dataclass(frozen=True)
class ShiftSpec:
    name: str
    start_hm: Tuple[int, int]
    end_hm: Tuple[int, int]
    crosses_midnight: Optional[bool] = None  # None -> auto (end<=start)

    def bounds(self, base_date: date) -> Tuple[datetime, datetime]:
        start_dt = datetime.combine(base_date, time(self.start_hm[0], self.start_hm[1]))
        end_dt = datetime.combine(base_date, time(self.end_hm[0], self.end_hm[1]))
        crosses = (end_dt <= start_dt) if self.crosses_midnight is None else self.crosses_midnight
        if crosses:
            end_dt += timedelta(days=1)
        return start_dt, end_dt


SHIFT_PRESETS = {
    "night": ShiftSpec("Ночная (18:30–06:30)", NIGHT_START, NIGHT_END, None),
    "day": ShiftSpec("Дневная (06:30–18:30)", DAY_START, DAY_END, False),
    "custom": ShiftSpec("Кастом", NIGHT_START, NIGHT_END, None),
}


# -------------------------
# Надежные утилиты
# -------------------------

def _safe_decode(data: bytes) -> str:
    for enc in ("cp1251", "windows-1251", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\xa0", " ").strip()
    return "" if s in {"&nbsp;", "\u00a0"} else s


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_target_table(soup: BeautifulSoup):
    tables = soup.find_all("table")
    if not tables:
        return None

    def score(tbl) -> int:
        ths = [(_clean_text(th.get_text(" ", strip=True)) or "").lower() for th in tbl.find_all("th")]
        if not ths:
            return 0
        s = 0
        for key in ("id вызова", "размещение", "split/skill", "имена пользователей", "время нач"):
            if any(key in th for th in ths):
                s += 3
        s += min(len(ths), 40) // 4
        return s

    best = max(tables, key=score)
    if score(best) == 0:
        return None
    return best


@st.cache_data(show_spinner=False)
def parse_avaya_html(html_bytes: bytes) -> pd.DataFrame:
    try:
        text = _safe_decode(html_bytes)
        soup = BeautifulSoup(text, "lxml")  # lxml обычно лучше Avaya-таблицы
        tbl = _find_target_table(soup)
        if tbl is None:
            return pd.DataFrame()

        headers = [_clean_text(th.get_text(" ", strip=True)) for th in tbl.find_all("th")]
        if not headers:
            return pd.DataFrame()

        rows = []
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            row = [_clean_text(td.get_text(" ", strip=True)) for td in tds]
            # выравниваем под headers
            if len(row) < len(headers):
                row += [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[: len(headers)]
            rows.append(row)

        return pd.DataFrame(rows, columns=headers)
    except Exception:
        return pd.DataFrame()


def normalize_calls(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Приводим к каноническим колонкам: dt_start, disposition, agent_code, skill_code, call_date, slot"""
    if df_raw.empty:
        return pd.DataFrame()

    col_date = _first_existing_col(df_raw, ["Дата", "Date"])
    col_time = _first_existing_col(df_raw, ["Время нач.", "Время нач", "Time"])
    col_disp = _first_existing_col(df_raw, ["Размещение", "Disposition"])
    col_skill = _first_existing_col(df_raw, ["Split/Skill", "Skill", "Split"])
    col_agent = _first_existing_col(df_raw, ["Имена пользователей", "Agent", "Пользователь", "User"])

    if not (col_date and col_time and col_disp):
        return pd.DataFrame()

    df = df_raw.copy()
    dt_str = (df[col_date].astype(str).str.strip() + " " + df[col_time].astype(str).str.strip()).str.strip()
    df["dt_start"] = pd.to_datetime(dt_str, format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt_start"]).reset_index(drop=True)

    df["disposition"] = df[col_disp].astype(str).str.strip().str.upper()
    df["skill_raw"] = "" if col_skill is None else df[col_skill].astype(str).str.strip()
    df["agent_code"] = "" if col_agent is None else df[col_agent].astype(str).str.strip()

    df["skill_code"] = df["skill_raw"].str.extract(r"(\d+)", expand=False).fillna("").astype(str)
    df["call_date"] = df["dt_start"].dt.date

    # слот (получас)
    df["slot"] = df["dt_start"].apply(lambda x: pd.Timestamp(x).replace(minute=(0 if x.minute < 30 else 30), second=0, microsecond=0))

    return df


def extract_agents_from_registry_text(text: str) -> Dict[str, str]:
    """
    Парсим только строки вида:
      "Оператор - Фамилия ИО (7599413)"
    Всё остальное (Телефон, станции и т.п.) игнорируем.
    """
    if not text:
        return {}
    out: Dict[str, str] = {}
    # допускаем пробелы, тире, двойные пробелы
    pattern = re.compile(r"Оператор\s*-\s*([^(]+)\((\d{5,12})\)", re.IGNORECASE)
    for line in text.splitlines():
        line = line.strip()
        m = pattern.search(line)
        if not m:
            continue
        name = m.group(1).strip()
        code = m.group(2).strip()
        if code and name:
            out[code] = name
    return out


def load_mapping_file(uploaded) -> Dict[str, str]:
    if uploaded is None:
        return {}
    try:
        name = getattr(uploaded, "name", "").lower()
        data = uploaded.getvalue()

        if name.endswith(".csv") or name.endswith(".txt"):
            df = None
            for sep in [",", ";", "\t"]:
                try:
                    df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str)
                    if df is not None and df.shape[1] >= 2:
                        break
                except Exception:
                    df = None
            if df is None:
                return {}
        else:
            df = pd.read_excel(io.BytesIO(data), dtype=str)

        df = df.fillna("")
        cols = [c.lower().strip() for c in df.columns]
        # пробуем угадать
        code_col = None
        name_col = None
        for c, raw in zip(cols, df.columns):
            if code_col is None and any(k in c for k in ["agent", "code", "номер", "id", "код"]):
                code_col = raw
            if name_col is None and any(k in c for k in ["name", "имя", "фио"]):
                name_col = raw

        if code_col is None or name_col is None:
            code_col, name_col = df.columns[0], df.columns[1]

        mapping: Dict[str, str] = {}
        for _, r in df.iterrows():
            code = str(r[code_col]).strip()
            nm = str(r[name_col]).strip()
            if code and nm:
                mapping[code] = nm
        return mapping
    except Exception:
        return {}


def apply_agent_map(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["agent_name"] = df["agent_code"].astype(str).map(mapping).fillna(df["agent_code"].astype(str))
    return df


def available_shift_dates(df: pd.DataFrame, shift: ShiftSpec) -> List[date]:
    if df.empty:
        return []
    base_dates = sorted(set(df["call_date"].tolist()))
    ok = []
    for bd in base_dates:
        st_dt, en_dt = shift.bounds(bd)
        m = (df["dt_start"] >= pd.Timestamp(st_dt)) & (df["dt_start"] < pd.Timestamp(en_dt))
        if m.any():
            ok.append(bd)
    return ok


def filter_by_shift(df: pd.DataFrame, shift: ShiftSpec, base_date: Optional[date]) -> Tuple[pd.DataFrame, Optional[Tuple[datetime, datetime]]]:
    if df.empty:
        return df, None

    if base_date is None:
        dates = available_shift_dates(df, shift)
        if not dates:
            return df.iloc[0:0].copy(), None
        parts = []
        for d in dates:
            st_dt, en_dt = shift.bounds(d)
            m = (df["dt_start"] >= pd.Timestamp(st_dt)) & (df["dt_start"] < pd.Timestamp(en_dt))
            parts.append(df.loc[m])
        out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
        st_dt = shift.bounds(dates[0])[0]
        en_dt = shift.bounds(dates[-1])[1]
        return out, (st_dt, en_dt)

    st_dt, en_dt = shift.bounds(base_date)
    m = (df["dt_start"] >= pd.Timestamp(st_dt)) & (df["dt_start"] < pd.Timestamp(en_dt))
    return df.loc[m].copy().reset_index(drop=True), (st_dt, en_dt)


def drop_leftovers(df_shift: pd.DataFrame, window: Tuple[datetime, datetime], minutes: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Исключаем операторов, которые появлялись только в первые minutes минут смены и потом исчезли.
    """
    if df_shift.empty:
        return df_shift, []
    st_dt, _ = window
    cutoff = st_dt + timedelta(minutes=minutes)

    agents = df_shift.loc[df_shift["agent_code"].astype(str).str.strip() != "", "agent_code"].astype(str).unique().tolist()
    excluded: List[str] = []
    for a in agents:
        df_a = df_shift[df_shift["agent_code"] == a]
        if df_a.empty:
            continue
        last = df_a["dt_start"].max()
        if last < cutoff:
            excluded.append(a)

    if excluded:
        df_shift = df_shift[~df_shift["agent_code"].isin(excluded)].copy().reset_index(drop=True)
    return df_shift, excluded


def compute_metrics(
    df_shift: pd.DataFrame,
    watched_skills: List[str],
    only_watched_for_missed: bool,
    treat_conn_as_accepted: bool,
) -> Dict[str, pd.DataFrame]:
    if df_shift.empty:
        return {"events": df_shift}

    df = df_shift.copy()
    accepted_set = set(DISP_ACCEPTED) | ({"CONN"} if treat_conn_as_accepted else set())

    df["is_accepted"] = df["disposition"].isin(accepted_set)
    df["is_missed"] = df["disposition"].isin(DISP_MISSED)

    if only_watched_for_missed and watched_skills:
        df["is_missed"] = df["is_missed"] & df["skill_code"].isin(watched_skills)

    df["is_missed_no_agent"] = df["is_missed"] & (df["agent_code"].astype(str).str.strip() == "")

    # KPI
    accepted = int(df["is_accepted"].sum())
    missed = int(df["is_missed"].sum())
    missed_no_agent = int(df["is_missed_no_agent"].sum())
    denom = accepted + missed
    pct_missed = 0.0 if denom == 0 else round(100.0 * missed / denom, 2)

    kpi = pd.DataFrame([{
        "Принято": accepted,
        "Пропущено": missed,
        "Пропущено без оператора": missed_no_agent,
        "% пропущенных": pct_missed,
        "Всего событий": int(len(df)),
    }])

    # Timeseries
    ts = (
        df.groupby("slot", dropna=False)
        .agg(
            Принято=("is_accepted", "sum"),
            Пропущено=("is_missed", "sum"),
            Пропущено_без_оператора=("is_missed_no_agent", "sum"),
            Всего=("disposition", "size"),
        )
        .reset_index()
        .sort_values("slot")
    )

    # Operators
    df_with_agent = df[df["agent_code"].astype(str).str.strip() != ""].copy()
    if df_with_agent.empty:
        ops = pd.DataFrame(columns=["agent_name", "Принято", "Пропущено", "% пропущенных"])
    else:
        ops = (
            df_with_agent.groupby(["agent_name"], dropna=False)
            .agg(Принято=("is_accepted", "sum"), Пропущено=("is_missed", "sum"))
            .reset_index()
        )
        denom2 = (ops["Принято"] + ops["Пропущено"]).replace(0, pd.NA)
        ops["% пропущенных"] = (100.0 * ops["Пропущено"] / denom2).fillna(0.0).round(2)
        ops = ops.sort_values(["Пропущено", "Принято"], ascending=[False, False])

    # Pivot (важно: без KeyError)
    if df_with_agent.empty:
        piv_acc = pd.DataFrame()
        piv_mis = pd.DataFrame()
    else:
        piv_acc = (
            df_with_agent[df_with_agent["is_accepted"]]
            .pivot_table(index="slot", columns="agent_name", values="is_accepted", aggfunc="sum", fill_value=0)
            .sort_index()
        )
        piv_mis = (
            df_with_agent[df_with_agent["is_missed"]]
            .pivot_table(index="slot", columns="agent_name", values="is_missed", aggfunc="sum", fill_value=0)
            .sort_index()
        )

    # Skills
    df["skill_name"] = df["skill_code"].map(SKILL_NAMES_DEFAULT).fillna(df["skill_code"].replace("", "Без тематики"))
    skills = (
        df.groupby(["skill_name"], dropna=False)
        .agg(Принято=("is_accepted", "sum"), Пропущено=("is_missed", "sum"))
        .reset_index()
        .sort_values(["Пропущено", "Принято"], ascending=[False, False])
    )

    # Bad slots
    bad = ts[(ts["Пропущено"] > 0) & (ts["Принято"] == 0)][["slot", "Пропущено", "Пропущено_без_оператора", "Всего"]]

    return {
        "kpi": kpi,
        "timeseries": ts,
        "operators": ops,
        "pivot_accepted": piv_acc,
        "pivot_missed": piv_mis,
        "skills": skills,
        "bad_slots": bad,
        "events": df.sort_values("dt_start").reset_index(drop=True),
    }


def safe_reindex_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Главный фикс твоего KeyError: даже если колонки отсутствуют — добавим их с нулями."""
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.reindex(columns=cols)
    return out.fillna(0).astype(int)


# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="Avaya CMS: смены (ночь/день)", layout="wide")
st.title("Avaya CMS — анализ смен (ночь/день)")

if "mode" not in st.session_state:
    st.session_state.mode = "night"

with st.sidebar:
    st.header("Файл")
    html_file = st.file_uploader("HTML отчёт Avaya", type=["html", "htm"])

    st.divider()
    st.header("Смена")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Ночь", use_container_width=True):
            st.session_state.mode = "night"
    with c2:
        if st.button("День", use_container_width=True):
            st.session_state.mode = "day"

    mode = st.radio("Режим", options=["night", "day", "custom"], format_func=lambda x: SHIFT_PRESETS[x].name, index=["night","day","custom"].index(st.session_state.mode))
    st.session_state.mode = mode

    if mode == "custom":
        st.caption("Если конец меньше начала — смена через полночь.")
        sh = st.number_input("Старт: час", 0, 23, NIGHT_START[0])
        sm = st.number_input("Старт: мин", 0, 59, NIGHT_START[1])
        eh = st.number_input("Конец: час", 0, 23, NIGHT_END[0])
        em = st.number_input("Конец: мин", 0, 59, NIGHT_END[1])
        SHIFT_PRESETS["custom"] = ShiftSpec("Кастом", (int(sh), int(sm)), (int(eh), int(em)), None)

    shift = SHIFT_PRESETS[mode]

    st.divider()
    st.header("Правила")
    treat_conn_as_accepted = st.checkbox("Считать CONN как 'Принято'", value=False)
    only_watched_for_missed = st.checkbox("Пропущено считать только по skills", value=(mode == "night"))
    watched_skills = st.multiselect("Skills для пропущенных", options=sorted(set(WATCHED_SKILLS_DEFAULT)), default=WATCHED_SKILLS_DEFAULT)

    st.divider()
    st.header("Хвосты")
    enable_leftovers = st.checkbox("Убирать операторов-«хвостов» (только начало смены)", value=(mode == "night"))
    leftover_min = st.number_input("Окно начала (мин)", 10, 240, DEFAULT_LEFTOVER_MIN, step=5)

    st.divider()
    st.header("Наглядность")
    top_n = st.slider("Топ N", 3, 30, 10)
    max_pivot_cols = st.slider("Макс операторов в матрице", 3, 60, 15)
    show_raw = st.checkbox("Показывать сырые события", value=False)

    st.divider()
    st.header("Операторы (код → имя)")
    show_mapping = st.checkbox("Показать настройки/редактор маппинга", value=False)

    # Маппинг: хранится в session_state, чтобы можно было дополнять
    if "agent_map" not in st.session_state:
        st.session_state.agent_map = dict(DEFAULT_AGENT_MAP)

    if show_mapping:
        mapping_file = st.file_uploader("Загрузить маппинг CSV/XLSX", type=["csv", "txt", "xlsx", "xls"])
        pasted = st.text_area("Вставь текст из реестра (берём только строки 'Оператор - ... (код)')", height=150)

        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("Добавить из файла", use_container_width=True) and mapping_file is not None:
                st.session_state.agent_map.update(load_mapping_file(mapping_file))
        with colB:
            if st.button("Добавить из текста", use_container_width=True) and pasted:
                st.session_state.agent_map.update(extract_agents_from_registry_text(pasted))
        with colC:
            if st.button("Сбросить к встроенным", use_container_width=True):
                st.session_state.agent_map = dict(DEFAULT_AGENT_MAP)

        # Редактор (по кнопке/чекбоксу)
        df_map = pd.DataFrame(
            sorted(st.session_state.agent_map.items(), key=lambda x: x[0]),
            columns=["agent_code", "agent_name"],
        )
        edited = st.data_editor(df_map, use_container_width=True, num_rows="dynamic", key="map_editor")
        # сохранить изменения
        try:
            edited = edited.fillna("")
            new_map = {}
            for _, r in edited.iterrows():
                c = str(r["agent_code"]).strip()
                n = str(r["agent_name"]).strip()
                if c and n:
                    new_map[c] = n
            st.session_state.agent_map = new_map
        except Exception:
            pass

        st.download_button(
            "Скачать текущий маппинг (CSV)",
            data=pd.DataFrame(sorted(st.session_state.agent_map.items()), columns=["agent_code","agent_name"]).to_csv(index=False).encode("utf-8"),
            file_name="agent_mapping.csv",
            mime="text/csv",
        )

if html_file is None:
    st.info("Загрузи HTML отчёт Avaya слева, потом выбирай Ночь/День и дату смены.")
    st.stop()

# Парсинг + нормализация
df_raw = parse_avaya_html(html_file.getvalue())
df = normalize_calls(df_raw)

if df.empty:
    st.error("Не смог распознать Avaya-таблицу (нужны колонки: Дата, Время нач., Размещение).")
    st.stop()

# Применяем маппинг
agent_map = st.session_state.get("agent_map", dict(DEFAULT_AGENT_MAP))
df = apply_agent_map(df, agent_map)

# Выбор даты смены (если в файле несколько дней)
dates = available_shift_dates(df, shift)
if len(dates) <= 1:
    base_date = dates[0] if dates else None
    if base_date:
        st.caption(f"В файле найдена 1 смена для режима: {base_date.strftime('%d.%m.%Y')}")
else:
    options = ["Все смены в файле"] + [d.strftime("%d.%m.%Y") for d in dates]
    chosen = st.selectbox("Дата смены (если в HTML несколько дней):", options=options, index=0)
    base_date = None if chosen == "Все смены в файле" else datetime.strptime(chosen, "%d.%m.%Y").date()

df_shift, window = filter_by_shift(df, shift, base_date)
if df_shift.empty:
    st.warning("В выбранном окне смены нет событий. Попробуй другую дату/режим.")
    st.stop()

excluded = []
if enable_leftovers and window is not None:
    df_shift, excluded = drop_leftovers(df_shift, window, int(leftover_min))

metrics = compute_metrics(
    df_shift=df_shift,
    watched_skills=[str(x) for x in watched_skills],
    only_watched_for_missed=only_watched_for_missed,
    treat_conn_as_accepted=treat_conn_as_accepted,
)

# Заголовок окна
if window is not None:
    st_dt, en_dt = window
    st.subheader(f"Окно анализа: {st_dt:%d.%m.%Y %H:%M} → {en_dt:%d.%m.%Y %H:%M} ({shift.name})")
else:
    st.subheader(f"Окно анализа: {shift.name}")

if excluded:
    st.caption("Исключены «хвосты»: " + ", ".join(excluded[:20]) + (" ..." if len(excluded) > 20 else ""))

# KPI
kpi = metrics["kpi"].iloc[0].to_dict()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Принято", int(kpi["Принято"]))
c2.metric("Пропущено", int(kpi["Пропущено"]))
c3.metric("Без оператора", int(kpi["Пропущено без оператора"]))
c4.metric("% пропущенных", f'{kpi["% пропущенных"]}%')
c5.metric("Всего", int(kpi["Всего событий"]))

# Динамика
st.markdown("### Динамика по получасам")
ts = metrics["timeseries"].set_index("slot")[["Принято", "Пропущено", "Пропущено_без_оператора"]]
st.line_chart(ts)

# Топы
st.markdown("### Топы")
ops = metrics["operators"]
skills = metrics["skills"]

colA, colB = st.columns(2)
with colA:
    st.markdown(f"**Топ-{top_n} операторов по пропущенным**")
    st.dataframe(ops.head(top_n), use_container_width=True)
with colB:
    st.markdown(f"**Топ-{top_n} тематик по пропущенным**")
    st.dataframe(skills.head(top_n), use_container_width=True)

# Матрица (без KeyError)
st.markdown("### Матрица (получас × оператор)")
piv_acc = metrics["pivot_accepted"]
piv_mis = metrics["pivot_missed"]

# список операторов берём из объединения pivot-колонок и ops (чтобы не ловить KeyError)
all_ops = sorted(set((list(piv_acc.columns) if not piv_acc.empty else []) + (list(piv_mis.columns) if not piv_mis.empty else []) + (ops["agent_name"].tolist() if not ops.empty else [])))
# дефолт: top по пропущенным + ограничение max_pivot_cols
default_ops = (ops["agent_name"].tolist() if not ops.empty else [])[: max_pivot_cols]
default_ops = [o for o in default_ops if o in all_ops][:max_pivot_cols]

selected_ops = st.multiselect("Выбери операторов для матрицы:", options=all_ops, default=default_ops)

if selected_ops:
    acc_show = safe_reindex_columns(piv_acc, selected_ops)
    mis_show = safe_reindex_columns(piv_mis, selected_ops)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Принятые**")
        st.dataframe(acc_show, use_container_width=True)
    with c2:
        st.markdown("**Пропущенные**")
        st.dataframe(mis_show, use_container_width=True)
else:
    st.caption("Выбери хотя бы одного оператора, чтобы показать матрицу.")

# Аномалии
st.markdown("### Аномалии")
st.dataframe(metrics["bad_slots"], use_container_width=True)

# Экспорт
st.markdown("### Экспорт")
events = metrics["events"]
st.download_button(
    "Скачать CSV событий (выбранная смена)",
    data=events.to_csv(index=False).encode("utf-8"),
    file_name="avaya_shift_events.csv",
    mime="text/csv",
)

if show_raw:
    st.markdown("### Сырые события (первые 2000 строк)")
    st.dataframe(events.head(2000), use_container_width=True)
