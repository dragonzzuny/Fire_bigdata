"""기상청 API 허브 — 과거 기상 자료와 화재위험 관련 파생 지표.

화재는 건조하고 바람이 셀 때 잘 난다. 그런데 격자별 기상 차이는 이 해상도에서
의미가 없다(도시 하나에 관측 지점이 몇 개뿐이다). 그래서 기상은
**격자를 구분하는 피처가 아니라 시점(월)을 구분하는 피처**로 쓴다.
연 단위 패널에 붙이면 모든 격자가 같은 값이 되어 순위에 아무 영향이 없다.

API 허브의 시간자료(`kma_sfctm3.php`)를 쓴다. 일자료(`kma_sfcdd3.php`)는
별도 활용신청이 필요해 기본으로는 막혀 있다 — 신청이 되면 그쪽이 더 싸다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://apihub.kma.go.kr/api/typ01/url"
ENDPOINT = f"{BASE}/kma_sfctm3.php"          # 지상 시간자료
ENDPOINT_DAILY = f"{BASE}/kma_sfcdd3.php"    # 지상 일자료 (활용신청 필요)

#: 일자료 컬럼 (API 명세 순서). 45개 중 앞쪽만 쓴다.
DAILY_COLUMNS = [
    "tm", "stn", "ws_avg", "wr_day", "wd_max", "ws_max", "ws_max_tm",
    "wd_ins", "ws_ins", "ws_ins_tm", "ta_avg", "ta_max", "ta_max_tm",
    "ta_min", "ta_min_tm", "td_avg", "ts_avg", "tg_min", "hm_avg", "hm_min",
    "hm_min_tm", "pv_avg", "ev_s", "ev_l", "fg_dur", "pa_avg", "ps_avg",
    "ps_max", "ps_max_tm", "ps_min", "ps_min_tm", "ca_tot", "ss_day",
    "ss_dur", "ss_cmb", "si_day", "si_60m_max", "si_60m_max_tm",
    "rn_day", "rn_d99", "rn_dur", "rn_60m_max", "rn_60m_max_tm",
    "rn_10m_max", "rn_10m_max_tm",
]

#: 건조주의보 기준. 실효습도 35% 이하가 2일 이상 계속될 때 발표된다.
#: 경보는 25% 이하 2일 이상. (기상청 특보 발표 기준)
DRY_ADVISORY_EH = 35.0
DRY_WARNING_EH = 25.0

#: 시도 -> 종관기상관측(ASOS) 대표 지점 번호
STATION = {"ulsan": 152, "sejong": 239, "busan": 159, "seoul": 108, "daegu": 143}

#: 시간자료 고정폭 컬럼 (헤더 주석의 순서를 따른다)
COLUMNS = ["tm", "stn", "wd", "ws", "gst_wd", "gst_ws", "gst_tm", "pa", "ps",
           "pt", "pr", "ta", "td", "hm", "pv", "rn", "rn_day", "rn_jun",
           "rn_int", "sd_hr3", "sd_day", "sd_tot", "wc", "wp", "ww"]

MISSING = {-9.0, -99.0, -999.0, -9.9, -99.9}


def _to_num(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    return v.mask(v.isin(MISSING) | (v <= -9.0))


def parse_response(text: str) -> pd.DataFrame:
    """`#` 로 시작하는 주석을 걷어내고 공백 구분 표를 읽는다."""
    rows = [ln.split() for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    width = min(len(COLUMNS), max(len(r) for r in rows))
    df = pd.DataFrame([r[:width] for r in rows], columns=COLUMNS[:width])
    df["datetime"] = pd.to_datetime(df["tm"], format="%Y%m%d%H%M", errors="coerce")
    for c in ("ta", "hm", "ws", "rn"):
        if c in df.columns:
            df[c] = _to_num(df[c])
    return df.dropna(subset=["datetime"])


def parse_daily(text: str) -> pd.DataFrame:
    """일자료 응답을 표로. 관측일(tm)은 YYYYMMDD."""
    rows = [ln.split() for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]
    if not rows:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    width = min(len(DAILY_COLUMNS), max(len(r) for r in rows))
    df = pd.DataFrame([r[:width] for r in rows], columns=DAILY_COLUMNS[:width])
    df["date"] = pd.to_datetime(df["tm"], format="%Y%m%d", errors="coerce")
    for c in ("ta_avg", "ta_max", "ta_min", "hm_avg", "hm_min",
              "ws_avg", "ws_max", "ws_ins", "rn_day", "ss_day"):
        if c in df.columns:
            df[c] = _to_num(df[c])
    return df.dropna(subset=["date"])


def fetch_daily(city: str, start: str, end: str, api_key: str, *,
                timeout: int = 90) -> pd.DataFrame:
    """일자료 한 구간. start/end 는 'YYYYMMDD'.

    시간자료보다 요청 수가 12분의 1이라, 8년치를 8회로 받는다.
    """
    stn = STATION.get(city)
    if stn is None:
        raise KeyError(f"'{city}' 의 관측 지점을 모른다. weather.STATION 에 추가하라.")
    r = requests.get(ENDPOINT_DAILY, timeout=timeout,
                     params={"tm1": start, "tm2": end, "stn": stn, "authKey": api_key})
    r.raise_for_status()
    text = r.content.decode("cp949", errors="replace")
    if '"status" : 4' in text or "활용신청" in text:
        raise PermissionError(
            f"지상 일자료 API 가 거절했다: {text[:160]}\n"
            f"  apihub.kma.go.kr → 지상관측 → '1. 지상 관측자료 조회' 활용신청 필요.")
    return parse_daily(text)


def fetch_daily_range(city: str, year_min: int, year_max: int, api_key: str,
                      *, cache_dir: Path | None = None) -> pd.DataFrame:
    """연도별로 나눠 받아 이어붙인다. 캐시가 있으면 API 를 다시 부르지 않는다."""
    frames = []
    for year in range(year_min, year_max + 1):
        cache = (cache_dir / f"weather_daily_{city}_{year}.parquet") if cache_dir else None
        if cache and cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        log.info("기상 일자료 수집 %s %d년", city, year)
        df = fetch_daily(city, f"{year}0101", f"{year}1231", api_key)
        if not df.empty and cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- 실효습도

def effective_humidity(daily_humidity: pd.Series, r: float = 0.7) -> pd.Series:
    """실효습도 — 건조주의보의 실제 기준값.

        He(t) = (1-r) * Σ r^k * H(t-k)

    당일 습도만 보면 '어제까지 며칠 말랐는가'가 빠진다. 실효습도는 과거 습도를
    기하급수적으로 감쇠시켜 누적하므로, 며칠째 건조한 상태를 잡아낸다.
    기상청은 r=0.7 을 쓴다.

    과거 특보 발효 이력은 API 허브에서 받을 수 없다(특보 API 는 현재 시점만 준다).
    그래서 기준값을 직접 계산해 재현한다 — 이 편이 전 기간에 대해 만들 수 있다.
    """
    h = pd.to_numeric(daily_humidity, errors="coerce")
    # ewm 의 adjust=False 가 정확히 (1-r)*Σ r^k * H(t-k) 를 만든다.
    return h.ewm(alpha=1 - r, adjust=False).mean()


def add_dry_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """일자료에 실효습도와 건조 판정을 붙인다."""
    if daily.empty or "hm_avg" not in daily.columns:
        return daily
    out = daily.sort_values("date").copy()
    out["eh"] = effective_humidity(out["hm_avg"])
    out["dry_advisory"] = out["eh"] <= DRY_ADVISORY_EH
    out["dry_warning"] = out["eh"] <= DRY_WARNING_EH
    # 실제 특보는 '2일 이상 계속'될 때 발표된다. 그 조건까지 재현한다.
    out["dry_advisory_2d"] = out["dry_advisory"] & out["dry_advisory"].shift(1, fill_value=False)
    # 건조하면서 바람이 센 날 — 화재 확산 조건에 가장 가깝다
    if "ws_avg" in out.columns:
        out["dry_windy"] = out["dry_advisory"] & (out["ws_avg"] >= 3.0)
    return out


def monthly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """일자료 -> 월별 지표. 시간자료판(monthly_features)보다 지표가 풍부하다."""
    if daily.empty:
        return pd.DataFrame()
    d = add_dry_indicators(daily)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    g = d.groupby(["year", "month"])
    out = pd.DataFrame({
        "temp_mean": g["ta_avg"].mean(),
        "temp_max": g["ta_max"].max(),
        "humidity_mean": g["hm_avg"].mean(),
        "humidity_min": g["hm_min"].min(),
        "eh_mean": g["eh"].mean(),
        "eh_min": g["eh"].min(),
        "dry_days": g["dry_advisory"].sum(),
        "dry_advisory_days": g["dry_advisory_2d"].sum(),
        "dry_warning_days": g["dry_warning"].sum(),
        "wind_mean": g["ws_avg"].mean(),
        "wind_max": g["ws_max"].max(),
        "rain_mm": g["rn_day"].sum(),
        "rain_days": g["rn_day"].apply(lambda s: int((s > 0.1).sum())),
        "n_days": g["ta_avg"].size(),
    }).reset_index()
    if "dry_windy" in d.columns:
        out["dry_windy_days"] = g["dry_windy"].sum().to_numpy()
    for c in ("dry_days", "dry_advisory_days", "dry_warning_days"):
        out[c] = out[c].astype(int)
    return out


def fetch(city: str, start: str, end: str, api_key: str, *,
          timeout: int = 60) -> pd.DataFrame:
    """한 구간의 시간자료. start/end 는 'YYYYMMDDHHMM'."""
    stn = STATION.get(city)
    if stn is None:
        raise KeyError(f"'{city}' 의 관측 지점을 모른다. weather.STATION 에 추가하라.")
    r = requests.get(ENDPOINT, timeout=timeout,
                     params={"tm1": start, "tm2": end, "stn": stn, "authKey": api_key})
    r.raise_for_status()
    text = r.content.decode("cp949", errors="replace")
    if '"status" : 4' in text or "활용신청" in text:
        raise PermissionError(
            f"기상청 API 가 거절했다: {text[:160]}\n"
            f"  apihub.kma.go.kr 에서 해당 API 활용신청이 필요하다.")
    return parse_response(text)


def fetch_range(city: str, year_min: int, year_max: int, api_key: str,
                *, cache_dir: Path | None = None) -> pd.DataFrame:
    """연도별로 나눠 받아 이어붙인다. 캐시가 있으면 API 를 다시 부르지 않는다."""
    frames = []
    for year in range(year_min, year_max + 1):
        cache = (cache_dir / f"weather_{city}_{year}.parquet") if cache_dir else None
        if cache and cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        log.info("기상 수집 %s %d년", city, year)
        df = fetch(city, f"{year}01010000", f"{year}12312300", api_key)
        if not df.empty and cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 파생 지표

def monthly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """시간자료 -> 월별 화재위험 관련 지표.

    `dry_hours` 는 상대습도 30% 미만 시간 수다. 건조주의보 기준(실효습도 35% 이하)을
    그대로 쓸 수는 없어서 — 실효습도는 여러 날 습도의 가중합이라 시간자료만으로는
    재현이 어렵다 — 관측 습도로 대신한다. 지표의 뜻을 바꾼 것이므로 이름도 다르게 뒀다.
    """
    if hourly.empty:
        return pd.DataFrame()
    d = hourly.copy()
    d["year"] = d["datetime"].dt.year
    d["month"] = d["datetime"].dt.month
    g = d.groupby(["year", "month"])
    out = pd.DataFrame({
        "temp_mean": g["ta"].mean(),
        "humidity_mean": g["hm"].mean(),
        "humidity_min": g["hm"].min(),
        "wind_mean": g["ws"].mean(),
        "wind_max": g["ws"].max(),
        "dry_hours": g["hm"].apply(lambda s: int((s < 30).sum())),
        "windy_dry_hours": g.apply(
            lambda x: int(((x["hm"] < 30) & (x["ws"] >= 4.0)).sum()), include_groups=False),
        "n_obs": g["ta"].size(),
    }).reset_index()
    # 건조·강풍이 겹치는 시간의 비율 — 화재 확산 조건에 가장 가깝다
    out["dry_share"] = out["dry_hours"] / out["n_obs"].replace(0, np.nan)
    out["windy_dry_share"] = out["windy_dry_hours"] / out["n_obs"].replace(0, np.nan)
    return out


def yearly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """연 단위 패널용. 격자를 구분하지 못하므로 순위 성능에는 기여하지 않는다.

    그래도 남겨 두는 이유는 '그 해가 건조했는가'를 보고서에 적기 위해서다.
    """
    if monthly.empty:
        return pd.DataFrame()
    g = monthly.groupby("year")
    return pd.DataFrame({
        "temp_mean": g["temp_mean"].mean(),
        "humidity_mean": g["humidity_mean"].mean(),
        "wind_mean": g["wind_mean"].mean(),
        "dry_hours": g["dry_hours"].sum(),
        "windy_dry_hours": g["windy_dry_hours"].sum(),
    }).reset_index()


def fire_weather_correlation(monthly: pd.DataFrame,
                             fires_by_month: pd.DataFrame) -> pd.DataFrame:
    """월별 화재 건수와 기상 지표의 상관. 무엇이 실제로 연관되는지 먼저 본다.

    fires_by_month: DataFrame[year, month, fires]
    """
    if monthly.empty or fires_by_month.empty:
        return pd.DataFrame()
    m = monthly.merge(fires_by_month, on=["year", "month"], how="inner")
    if len(m) < 6:
        return pd.DataFrame()
    # 시간자료판과 일자료판의 지표 이름이 다르다. 둘 다 본다.
    candidates = ("temp_mean", "temp_max", "humidity_mean", "humidity_min",
                  "eh_mean", "eh_min", "dry_days", "dry_advisory_days",
                  "dry_warning_days", "dry_windy_days", "wind_mean", "wind_max",
                  "rain_mm", "rain_days", "dry_hours", "dry_share", "windy_dry_share")
    cols = [c for c in candidates if c in m.columns and m[c].nunique() > 1]
    rows = [{"지표": c, "상관계수": float(m[c].corr(m["fires"])), "n": int(len(m))}
            for c in cols]
    return (pd.DataFrame(rows).dropna(subset=["상관계수"])
            .reindex(pd.DataFrame(rows).dropna(subset=["상관계수"])["상관계수"]
                     .abs().sort_values(ascending=False).index)
            .reset_index(drop=True))
