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

# ---------------------------------------------------------------- 격자별 시간대

#: 순찰 근무와 맞물리는 네 토막. 24시간을 그대로 쓰면 격자당 화재가
#: 중앙 7건이라 어느 시간에 한두 건 몰린 것이 '그 구역의 성격'으로 읽힌다.
TIME_BLOCKS = (
    ("심야", 0, 6),
    ("오전", 6, 12),
    ("오후", 12, 18),
    ("저녁", 18, 24),
)

#: 축소추정의 사전 무게. 이 구역에 화재가 이만큼 있어야 자기 분포를 절반쯤
#: 믿는다는 뜻이다. 화재 2건짜리 구역이 '심야 100%'라고 말하지 않게 한다.
PRIOR_WEIGHT = 20.0


def _block_of(hour):
    h = pd.to_numeric(hour, errors="coerce")
    out = pd.Series(pd.NA, index=h.index, dtype="object")
    for name, lo, hi in TIME_BLOCKS:
        out = out.mask(h.between(lo, hi - 1), name)
    return out


def city_block_share(fires: pd.DataFrame) -> pd.Series:
    """도시 전체의 시간대 구성. 축소추정이 끌어당길 기준점이다."""
    f = fires
    if "has_time" in f.columns:
        f = f[f["has_time"] == True]                       # noqa: E712
    b = _block_of(f.get("hour", pd.Series(dtype=float))).dropna()
    names = [n for n, _, _ in TIME_BLOCKS]
    if b.empty:
        return pd.Series([0.25] * len(names), index=names)
    share = b.value_counts(normalize=True)
    return share.reindex(names).fillna(0.0)


def grid_hour_profile(fires: pd.DataFrame, *,
                      prior_weight: float = PRIOR_WEIGHT) -> pd.DataFrame:
    """격자별 시간대 구성 — 표본이 적으면 도시 전체 쪽으로 당긴다.

    격자 하나의 화재는 중앙 7건뿐이라 그대로 비율을 내면 우연이 성격으로
    둔갑한다. 그래서 경험적 베이즈로 섞는다:

        p = (n · p_격자 + k · p_도시) / (n + k)

    화재가 많은 구역은 자기 분포를 따르고, 적은 구역은 도시 평균에 가깝게
    남는다. '표본이 적다'와 '특징이 없다'가 같은 답으로 나오는 것이 맞다.

    반환: grid_id, 심야/오전/오후/저녁 비율(축소추정), n_심야… 원 건수,
          n(시각이 있는 화재 수)
    """
    names = [n for n, _, _ in TIME_BLOCKS]
    city = city_block_share(fires)

    f = fires
    if "has_time" in f.columns:
        f = f[f["has_time"] == True]                       # noqa: E712
    f = f.dropna(subset=["grid_id"]).copy()
    f["_b"] = _block_of(f.get("hour", pd.Series(dtype=float)))
    f = f.dropna(subset=["_b"])
    cols = ["grid_id", *names, *[f"n_{x}" for x in names], "n"]
    if f.empty:
        return pd.DataFrame(columns=cols)

    cnt = (f.groupby(["grid_id", "_b"]).size().unstack(fill_value=0)
             .reindex(columns=names, fill_value=0))
    n = cnt.sum(axis=1)
    shrunk = cnt.add(city * prior_weight, axis=1).div(n + prior_weight, axis=0)

    out = shrunk.reset_index()
    for name in names:
        out[f"n_{name}"] = cnt[name].to_numpy()
    out["n"] = n.to_numpy()
    out.attrs["city"] = city.to_dict()
    return out


def _wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """비율의 95% 신뢰구간 아래끝. 표본이 작으면 크게 내려간다."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    margin = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, center - margin)


#: 특이시간대를 말하기 위한 최소 화재 수. 이보다 적으면 신뢰구간이 통과해도
#: 실무에서 쓸 말이 못 된다 — 화재 3건으로 순찰 시간을 바꾸지는 않는다.
MIN_FIRES_FOR_LABEL = 5


def busiest_block(profile_row) -> str:
    """이 구역에서 화재가 **가장 많은** 시간대. 원 건수 그대로."""
    best, best_n = "", -1
    for name, _, _ in TIME_BLOCKS:
        k = int(profile_row.get(f"n_{name}", 0) or 0)
        if k > best_n:
            best, best_n = name, k
    return best if best_n > 0 else ""


def unusual_block(profile_row, city: dict | None = None) -> str:
    """도시 평균보다 **통계적으로 많은** 시간대. 없으면 빈 문자열.

    '가장 많은 시간대'와 다른 질문이다. 어느 구역은 화재의 32%가 오후에
    나지만 도시 전체가 35%라면 그 구역의 오후는 유별나지 않다. 반대로 심야가
    18%인데 도시가 14.5%면 그 구역은 심야에 유별나다. 순찰 시간을 도시 기준에서
    **옮길 근거**가 되는 것은 뒤쪽이다.

    처음엔 축소추정 비율이 도시 평균의 1.15배를 넘으면 이름을 붙였다. 그랬더니
    화재 3건짜리 격자 367개 중 165개에 이름이 붙었다 — 심야 화재 한 건이면
    그 구역이 '심야형'이 됐다. 도시 전체에서 심야 비중이 가장 낮아 분모가 작은
    탓이다. 그래서 배수가 아니라 원 건수의 신뢰구간으로 판단한다.
    """
    city = city or {}
    n = int(profile_row.get("n", 0) or 0)
    if n < MIN_FIRES_FOR_LABEL or not city:
        return ""
    best, best_gap = "", 0.0
    for name, _, _ in TIME_BLOCKS:
        k = int(profile_row.get(f"n_{name}", 0) or 0)
        gap = _wilson_lower(k, n) - float(city.get(name, 0.0))
        if gap > best_gap:
            best, best_gap = name, gap
    return best


def label_blocks(profile: pd.DataFrame) -> pd.DataFrame:
    """profile 에 '최다시간대'(그 구역에서 가장 많은 때)와
    '특이시간대'(도시 대비 유별난 때)를 붙인다. 둘은 다를 수 있고,
    다를 때가 오히려 정상이다."""
    if profile.empty:
        return profile.assign(최다시간대="", 특이시간대="")
    city = profile.attrs.get("city", {})
    out = profile.copy()
    out["최다시간대"] = out.apply(busiest_block, axis=1)
    out["특이시간대"] = out.apply(lambda r: unusual_block(r, city), axis=1)
    out.attrs["city"] = city
    return out


def block_hours(name: str) -> tuple[int, int] | None:
    for n, lo, hi in TIME_BLOCKS:
        if n == name:
            return lo, hi
    return None

