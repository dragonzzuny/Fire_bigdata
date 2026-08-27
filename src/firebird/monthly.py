"""월 단위 화재위험 — '이번 달, 어디를' 에 답한다.

연 단위 모델은 '어디를' 만 답한다. 그런데 예방순찰은 매달 짜야 하고,
화재는 계절을 탄다. 기상(건조·바람)도 격자를 구분하지 못하므로
연 단위 패널에 붙이면 아무 영향이 없다 — 시점을 구분하는 축이 있어야
비로소 쓸 수 있다.

설계: **격자 순위(연 단위) x 월 위험 계수(시점)** 로 분해한다.

    이번 달 이 격자의 위험 = 격자 위험도 x 그 달의 위험 계수

격자를 월별로 다시 학습하지 않는 이유는 두 가지다.
  · 격자x월 셀이 14만 개인데 화재는 1만 건뿐이라 라벨이 지나치게 희박하다.
  · 기존 연 단위 수치(포착률 68.5% 등)와의 연속성이 끊긴다.
분해하면 검증된 격자 순위를 그대로 쓰면서 시점만 얹을 수 있다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MONTH_LABEL = [f"{m}월" for m in range(1, 13)]


def fires_by_month(fires: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    """화재 원본 -> DataFrame[year, month, fires]."""
    if fires.empty:
        return pd.DataFrame(columns=["year", "month", "fires"])
    d = fires.dropna(subset=["grid_id"]).copy()
    dt = pd.to_datetime(d["occurred_at"].astype(str).str.replace(r"\D", "", regex=True)
                        .str.slice(0, 8), format="%Y%m%d", errors="coerce")
    d = d.assign(year=dt.dt.year, month=dt.dt.month).dropna(subset=["year", "month"])
    d = d[d["year"].between(year_min, year_max)]
    out = d.groupby(["year", "month"]).size().rename("fires").reset_index()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out


def seasonal_index(monthly_fires: pd.DataFrame) -> pd.DataFrame:
    """월별 계절 지수 — 그 달이 연평균 대비 몇 배인가.

    연도별 총량 차이를 먼저 제거한다. 2014년이 2021년보다 화재가 많았다는 사실이
    '1월이 위험하다'로 잘못 읽히면 안 된다.
    """
    if monthly_fires.empty:
        return pd.DataFrame(columns=["month", "seasonal_index", "n_years"])
    d = monthly_fires.copy()
    year_mean = d.groupby("year")["fires"].transform("mean")
    d["rel"] = d["fires"] / year_mean.replace(0, np.nan)
    out = (d.groupby("month")["rel"].agg(["mean", "std", "size"])
             .rename(columns={"mean": "seasonal_index", "std": "sd", "size": "n_years"})
             .reset_index())
    out["month_label"] = [MONTH_LABEL[m - 1] for m in out["month"]]
    return out


def fit_month_risk(monthly_fires: pd.DataFrame,
                   weather_monthly: pd.DataFrame | None = None) -> dict:
    """월 위험 계수 모델.

    기상이 있으면 계절지수에 기상을 더해 회귀로 보정한다. 없으면 계절지수만 쓴다.
    기상을 넣어 설명력이 늘지 않으면 넣지 않는다 — 그 사실도 함께 돌려준다.
    """
    season = seasonal_index(monthly_fires)
    result: dict = {"seasonal": season.to_dict(orient="records"),
                    "uses_weather": False}
    if season.empty:
        return result

    d = monthly_fires.merge(season[["month", "seasonal_index"]], on="month", how="left")
    year_mean = d.groupby("year")["fires"].transform("mean")
    d["rel"] = d["fires"] / year_mean.replace(0, np.nan)
    d = d.dropna(subset=["rel"])
    result["baseline_r2"] = float(_r2(d["rel"], d["seasonal_index"]))

    if weather_monthly is None or weather_monthly.empty:
        return result

    from sklearn.linear_model import RidgeCV

    w = weather_monthly.copy()
    feats = [c for c in ("humidity_mean", "eh_mean", "temp_mean", "wind_mean",
                         "dry_days", "rain_days")
             if c in w.columns and w[c].nunique() > 1]
    if not feats:
        return result

    m = d.merge(w[["year", "month"] + feats], on=["year", "month"], how="inner").dropna()
    if len(m) < 24:
        log.info("월 관측치 %d개 — 기상 보정을 하기엔 부족하다", len(m))
        return result

    X = m[["seasonal_index"] + feats].to_numpy(dtype=float)
    y = m["rel"].to_numpy(dtype=float)
    # 표준화해야 규제가 공평하게 걸린다.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    model = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit((X - mu) / sd, y)
    pred = model.predict((X - mu) / sd)

    r2_season = float(_r2(m["rel"], m["seasonal_index"]))
    r2_full = float(_r2(y, pred))
    result.update({
        "baseline_r2": r2_season,
        "weather_r2": r2_full,
        "gain": r2_full - r2_season,
        "features": feats,
        "coef": {f: float(c) for f, c in zip(["seasonal_index"] + feats, model.coef_)},
        "uses_weather": bool(r2_full > r2_season + 0.01),
        "n_months": int(len(m)),
        "_model": model, "_mu": mu, "_sd": sd,
    })
    if not result["uses_weather"]:
        log.info("기상을 넣어도 설명력이 늘지 않는다(%.3f -> %.3f). 계절지수만 쓴다.",
                 r2_season, r2_full)
    return result


def _r2(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def month_multiplier(fit: dict, month: int,
                     weather_row: pd.Series | None = None) -> float:
    """그 달의 위험 계수. 1.0 이면 연평균 수준."""
    season = pd.DataFrame(fit.get("seasonal", []))
    if season.empty:
        return 1.0
    row = season[season["month"] == int(month)]
    base = float(row["seasonal_index"].iloc[0]) if len(row) else 1.0

    if not fit.get("uses_weather") or weather_row is None or "_model" in fit is False:
        return base
    model, mu, sd = fit.get("_model"), fit.get("_mu"), fit.get("_sd")
    if model is None:
        return base
    feats = fit["features"]
    try:
        x = np.array([base] + [float(weather_row[f]) for f in feats], dtype=float)
    except (KeyError, TypeError, ValueError):
        return base
    return float(model.predict(((x - mu) / sd).reshape(1, -1))[0])


def monthly_plan(panel_year: pd.DataFrame, risk_col: str, fit: dict,
                 weather_monthly: pd.DataFrame | None = None,
                 year: int | None = None) -> pd.DataFrame:
    """월별 위험 계수 표 — 순찰·점검 일정을 짜는 근거."""
    season = pd.DataFrame(fit.get("seasonal", []))
    if season.empty:
        return pd.DataFrame()
    rows = []
    for m in range(1, 13):
        wrow = None
        if weather_monthly is not None and not weather_monthly.empty and year:
            sel = weather_monthly[(weather_monthly["year"] == year)
                                  & (weather_monthly["month"] == m)]
            if len(sel):
                wrow = sel.iloc[0]
        mult = month_multiplier(fit, m, wrow)
        r = {"월": MONTH_LABEL[m - 1], "month": m, "위험계수": round(mult, 3)}
        if wrow is not None:
            for c in ("humidity_mean", "eh_mean", "wind_mean", "dry_days"):
                if c in wrow:
                    r[c] = round(float(wrow[c]), 1)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["등급"] = pd.cut(out["위험계수"], bins=[-np.inf, 0.9, 1.0, 1.1, np.inf],
                       labels=["낮음", "보통", "주의", "높음"])
    total = float(pd.to_numeric(panel_year[risk_col], errors="coerce").sum())
    out["예상화재_건"] = (out["위험계수"] * total / 12).round(1)
    return out
