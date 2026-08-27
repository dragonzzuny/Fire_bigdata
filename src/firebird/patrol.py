"""예방순찰: 언제(시간대·요일) 어디를 어떤 순서로 돌 것인가."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .operations import two_opt

WEEKDAY_KOR = ["월", "화", "수", "목", "금", "토", "일"]


def hour_profile(fires: pd.DataFrame) -> pd.DataFrame:
    """시간대별 화재 분포. '오후·심야에 몰린다'를 숫자로 확인한다."""
    if "hour" not in fires.columns:
        return pd.DataFrame(columns=["hour", "n", "share"])
    s = fires["hour"].dropna().astype(int)
    tab = s.value_counts().reindex(range(24), fill_value=0).sort_index()
    return pd.DataFrame({"hour": tab.index, "n": tab.to_numpy(),
                         "share": tab.to_numpy() / max(tab.sum(), 1)})


def weekday_profile(fires: pd.DataFrame) -> pd.DataFrame:
    if "weekday" not in fires.columns:
        return pd.DataFrame(columns=["weekday", "요일", "n", "share"])
    s = fires["weekday"].dropna().astype(int)
    tab = s.value_counts().reindex(range(7), fill_value=0).sort_index()
    return pd.DataFrame({"weekday": tab.index,
                         "요일": [WEEKDAY_KOR[i] for i in tab.index],
                         "n": tab.to_numpy(), "share": tab.to_numpy() / max(tab.sum(), 1)})


def peak_windows(fires: pd.DataFrame, top_n: int = 3, window: int = 3) -> list[dict]:
    """연속 window 시간 중 화재가 가장 많은 구간 top_n."""
    prof = hour_profile(fires)
    if prof.empty:
        return []
    n = prof["n"].to_numpy()
    sums = [(int(h), float(n[[(h + i) % 24 for i in range(window)]].sum()))
            for h in range(24)]
    sums.sort(key=lambda t: -t[1])
    chosen: list[dict] = []
    used: set[int] = set()
    for start, total in sums:
        hours = {(start + i) % 24 for i in range(window)}
        if hours & used:
            continue
        used |= hours
        chosen.append({"start_hour": start, "end_hour": (start + window) % 24,
                       "fires": total, "share": total / max(n.sum(), 1)})
        if len(chosen) >= top_n:
            break
    return chosen


def patrol_route(top_grids: pd.DataFrame, start_grid: str | None = None) -> pd.DataFrame:
    """최근접 이웃 순서로 순찰 동선을 만든다.

    최적 경로(TSP)가 아니라 '지금 바로 쓸 수 있는 합리적 순서'다.
    도로망을 쓰지 않으므로 직선거리 기준이라는 점을 표에 남긴다.
    """
    df = top_grids.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["순번", "grid_id", "lon", "lat", "이동거리_m"])

    # 직선거리 계산은 미터 좌표에서. 경위도 차이를 그대로 쓰면 위도별로 왜곡된다.
    lat0 = float(df["lat"].mean())
    mx = df["lon"].to_numpy() * 88_800 * np.cos(np.radians(lat0)) / 0.8
    my = df["lat"].to_numpy() * 111_000
    pts = np.column_stack([mx, my])

    start = 0
    if start_grid is not None and (df["grid_id"] == start_grid).any():
        start = int(df.index[df["grid_id"] == start_grid][0])

    order = [start]
    remaining = set(range(len(df))) - {start}
    while remaining:
        cur = order[-1]
        nxt = min(remaining, key=lambda j: float(np.hypot(*(pts[j] - pts[cur]))))
        order.append(nxt)
        remaining.discard(nxt)

    # 최근접 이웃만 쓰면 마지막에 먼 격자로 되돌아가는 교차 구간이 남는다.
    # 2-opt 로 그걸 풀면 같은 격자를 도는데 이동거리가 크게 준다.
    if len(order) > 3:
        order = two_opt(pts, order)

    dists = [0.0] + [float(np.hypot(*(pts[order[i]] - pts[order[i - 1]])))
                     for i in range(1, len(order))]
    out = df.iloc[order].copy().reset_index(drop=True)
    out.insert(0, "순번", range(1, len(out) + 1))
    out["이동거리_m"] = np.round(dists, 0)
    out["누적거리_m"] = out["이동거리_m"].cumsum()
    return out
