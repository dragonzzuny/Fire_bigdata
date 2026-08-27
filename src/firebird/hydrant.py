"""대응취약 분석: 고위험인데 소화전이 없는 격자.

'전체 격자의 몇 %에 소화전이 없는가'는 그 자체로는 정책 근거가 되기 어렵다.
빈 산지에도 소화전은 없기 때문이다. 위험 상위 구간과 교차했을 때
비로소 '어디에 먼저 놓아야 하는가'라는 답이 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hydrant_coverage(panel_year: pd.DataFrame) -> dict:
    """소화전 보유 현황 요약."""
    if "n_hydrant" not in panel_year.columns:
        return {"available": False}
    n = len(panel_year)
    none = int((panel_year["n_hydrant"] <= 0).sum())
    return {
        "available": True,
        "n_grids": n,
        "grids_without_hydrant": none,
        "share_without_hydrant": (none / n) if n else 0.0,
        "median_hydrants": float(panel_year["n_hydrant"].median()),
    }


def blind_spots(panel_year: pd.DataFrame, risk: pd.Series, cfg) -> pd.DataFrame:
    """위험 상위 p% 이면서 소화전이 없는(또는 먼) 격자."""
    if "n_hydrant" not in panel_year.columns:
        return pd.DataFrame()
    hconf = cfg["hydrant"]
    p = float(hconf["high_risk_percentile"])
    max_dist = float(hconf["max_dist_m"])

    df = panel_year.copy()
    df["risk"] = np.asarray(risk, dtype=float)
    n_top = max(1, int(round(len(df) * p / 100.0)))
    top = df.nlargest(n_top, "risk")

    far = top["n_hydrant"] <= 0
    if "dist_hydrant_m" in top.columns:
        far = far | (top["dist_hydrant_m"] > max_dist)

    cols = [c for c in ["grid_id", "sgg", "lon", "lat", "risk", "fires", "fires_cum",
                        "n_hydrant", "dist_hydrant_m", "target_total", "biz_total"]
            if c in top.columns]
    out = top.loc[far, cols].sort_values("risk", ascending=False).reset_index(drop=True)
    out.insert(0, "우선순위", range(1, len(out) + 1))
    return out


def surge_alert(panel: pd.DataFrame, year: int, *, min_prev: float = 1.0,
                ratio: float = 2.0) -> pd.DataFrame:
    """화재 급증 경보: 직전연도 대비 크게 늘어난 격자."""
    cur = panel[panel["year"] == year][["grid_id", "sgg", "fires", "fires_lag1"]].copy()
    cur = cur[(cur["fires_lag1"] >= min_prev) &
              (cur["fires"] >= cur["fires_lag1"] * ratio)]
    cur["증가배수"] = cur["fires"] / cur["fires_lag1"].replace(0, np.nan)
    return cur.sort_values("증가배수", ascending=False).reset_index(drop=True)
