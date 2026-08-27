"""도로 기준 거리와 다중 팀 순찰 경로.

직선거리로 짠 동선은 현장에서 못 쓴다. 울산은 태화강과 산이 도시를 가르고
있어서, 직선으로 가까운 두 격자가 도로로는 크게 돌아가는 경우가 많다.

세 가지를 한다:
  1) 도로 거리행렬   OSRM 공개 서버로 실제 주행거리·시간을 받는다.
                     실패하면 직선거리 x 우회계수로 물러나되, 어느 쪽을 썼는지 기록한다.
  2) 구역 분할       동시에 도는 팀 수만큼 지역을 나눈다. 서로 겹치지 않아야
                     같은 격자를 두 팀이 가는 낭비가 없다.
  3) 경로 최적화     팀별로 최근접 이웃 + 2-opt.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

from .operations import two_opt

log = logging.getLogger(__name__)

OSRM_BASE = "https://router.project-osrm.org"

#: 직선거리를 도로거리로 환산할 때 쓰는 우회계수.
#: 도심 도로망의 경험적 값(1.2~1.5)의 중간. OSRM 이 되면 이 값은 쓰이지 않는다.
DETOUR_FACTOR = 1.35

#: OSRM 공개 서버의 table 좌표 상한. 넘으면 나눠서 부른다.
OSRM_TABLE_LIMIT = 90


def haversine_matrix(lons, lats) -> np.ndarray:
    """직선거리 행렬(m). 지구 곡률을 반영한다."""
    lon = np.radians(np.asarray(lons, dtype=float))
    lat = np.radians(np.asarray(lats, dtype=float))
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _osrm_table(lons, lats, *, timeout: int = 60) -> tuple[np.ndarray, np.ndarray] | None:
    """OSRM table API 로 (거리, 소요시간) 행렬. 실패하면 None."""
    coords = ";".join(f"{lo:.6f},{la:.6f}" for lo, la in zip(lons, lats))
    url = f"{OSRM_BASE}/table/v1/driving/{coords}"
    try:
        r = requests.get(url, params={"annotations": "distance,duration"}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("OSRM 호출 실패(%s) — 직선거리 x 우회계수로 대체", type(exc).__name__)
        return None
    if data.get("code") != "Ok":
        log.warning("OSRM 응답 코드 %s — 직선거리로 대체", data.get("code"))
        return None
    dist = np.asarray(data["distances"], dtype=float)
    dur = np.asarray(data.get("durations", dist / 11.0), dtype=float)
    return dist, dur


def road_distance_matrix(lons, lats, *, use_road: bool = True,
                         timeout: int = 60) -> tuple[np.ndarray, np.ndarray, str]:
    """(거리 m, 소요시간 s, 출처). 출처는 'osrm' 또는 'straight_line_x_detour'.

    좌표가 많으면 OSRM 상한에 걸리므로 블록으로 나눠 받는다.
    한 블록이라도 실패하면 전체를 직선거리로 물러난다 — 두 방식이 섞인
    행렬은 어느 구간이 무슨 기준인지 알 수 없어 더 나쁘다.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    n = len(lons)
    straight = haversine_matrix(lons, lats)

    if not use_road or n < 2:
        return straight * DETOUR_FACTOR, straight * DETOUR_FACTOR / 11.0, "straight_line_x_detour"

    if n <= OSRM_TABLE_LIMIT:
        got = _osrm_table(lons, lats, timeout=timeout)
        if got is None:
            return (straight * DETOUR_FACTOR, straight * DETOUR_FACTOR / 11.0,
                    "straight_line_x_detour")
        dist, dur = got
        # OSRM 이 도달 불가로 표시한 칸은 직선거리로 메운다.
        bad = ~np.isfinite(dist)
        if bad.any():
            dist = np.where(bad, straight * DETOUR_FACTOR, dist)
            dur = np.where(bad, straight * DETOUR_FACTOR / 11.0, dur)
        return dist, dur, "osrm"

    log.info("좌표 %d개 — OSRM 상한(%d)을 넘어 블록으로 나눠 호출", n, OSRM_TABLE_LIMIT)
    dist = np.zeros((n, n))
    dur = np.zeros((n, n))
    step = OSRM_TABLE_LIMIT // 2
    for i in range(0, n, step):
        for j in range(0, n, step):
            idx = list(range(i, min(i + step, n))) + list(range(j, min(j + step, n)))
            idx = sorted(set(idx))
            got = _osrm_table(lons[idx], lats[idx], timeout=timeout)
            if got is None:
                return (straight * DETOUR_FACTOR, straight * DETOUR_FACTOR / 11.0,
                        "straight_line_x_detour")
            sub_d, sub_t = got
            for a, ga in enumerate(idx):
                for b, gb in enumerate(idx):
                    dist[ga, gb] = sub_d[a, b]
                    dur[ga, gb] = sub_t[a, b]
    bad = ~np.isfinite(dist) | (dist <= 0)
    np.fill_diagonal(bad, False)
    if bad.any():
        dist = np.where(bad, straight * DETOUR_FACTOR, dist)
        dur = np.where(bad, straight * DETOUR_FACTOR / 11.0, dur)
    return dist, dur, "osrm"


# ---------------------------------------------------------------- 구역 분할

def partition_teams(df: pd.DataFrame, n_teams: int, *, weight_col: str | None = None,
                    seed: int = 42, max_iter: int = 60) -> np.ndarray:
    """순찰 격자를 팀 수만큼 서로 겹치지 않는 구역으로 나눈다.

    용량 제한 k-means: 각 팀이 비슷한 양(격자 수 또는 위험 가중)을 맡도록
    중심점에 가까운 순서로 배정하되 정원을 넘기지 않는다. 그냥 k-means 만 쓰면
    한 팀이 30격자, 다른 팀이 3격자를 맡는 계획이 나온다.
    """
    from sklearn.cluster import KMeans

    n = len(df)
    k = max(1, min(int(n_teams), n))
    if k == 1:
        return np.zeros(n, dtype=int)

    xy = np.column_stack([
        df["lon"].astype(float).to_numpy() * np.cos(np.radians(df["lat"].astype(float).mean())),
        df["lat"].astype(float).to_numpy(),
    ])
    w = (df[weight_col].astype(float).to_numpy()
         if weight_col and weight_col in df.columns else np.ones(n))
    w = np.clip(w, 1e-9, None)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=max_iter)
    km.fit(xy, sample_weight=w)
    centers = km.cluster_centers_

    # 용량: 총 가중의 1/k 에 약간의 여유
    cap = w.sum() / k * 1.15
    d = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)

    order = np.argsort(d.min(axis=1))          # 확신이 큰 점부터 배정
    labels = np.full(n, -1, dtype=int)
    load = np.zeros(k)
    for i in order:
        for c in np.argsort(d[i]):
            if load[c] + w[i] <= cap or (labels == c).sum() == 0:
                labels[i] = c
                load[c] += w[i]
                break
        if labels[i] < 0:                       # 전부 찼으면 가장 덜 찬 팀
            c = int(np.argmin(load))
            labels[i] = c
            load[c] += w[i]
    return labels


# ---------------------------------------------------------------- 경로

def solve_route(dist: np.ndarray, start: int = 0) -> list[int]:
    """거리행렬 위에서 최근접 이웃 + 2-opt."""
    n = len(dist)
    if n <= 1:
        return list(range(n))
    order = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        cur = order[-1]
        nxt = min(remaining, key=lambda j: dist[cur, j])
        order.append(nxt)
        remaining.discard(nxt)
    if n > 3:
        order = _two_opt_matrix(dist, order)
    return order


def _two_opt_matrix(dist: np.ndarray, order: list[int], max_rounds: int = 60) -> list[int]:
    """행렬 기반 2-opt. operations.two_opt 은 좌표 기반이라 도로거리를 못 쓴다."""
    best = list(order)
    improved, rounds = True, 0
    while improved and rounds < max_rounds:
        improved, rounds = False, rounds + 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b, c, d = best[i - 1], best[i], best[j], best[j + 1]
                if dist[a, b] + dist[c, d] > dist[a, c] + dist[b, d] + 1e-9:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
    return best


def route_length(dist: np.ndarray, order: list[int]) -> float:
    return float(sum(dist[order[i - 1], order[i]] for i in range(1, len(order))))


def allocate_teams_to_groups(df: pd.DataFrame, n_teams: int, group_col: str = "sgg",
                             weight_col: str | None = None) -> dict[str, int]:
    """관할별로 팀을 몇 개씩 둘지 정한다 (최대잔여법).

    한 팀이 두 관할에 걸치면 지휘 계통이 갈린다. 관할이 곧 담당 소방서이므로,
    관할 안에서 팀을 나누는 편이 현장 운영에 맞다.
    관할 수보다 팀이 적으면 위험이 큰 관할부터 한 팀씩 준다.
    """
    if group_col not in df.columns:
        return {}
    w = (df[weight_col].astype(float) if weight_col and weight_col in df.columns
         else pd.Series(1.0, index=df.index))
    share = w.groupby(df[group_col].fillna("").astype(str)).sum()
    share = share[share.index.str.strip() != ""]
    if share.empty:
        return {}
    share = share / share.sum()

    k = max(1, int(n_teams))
    if len(share) >= k:                       # 팀이 부족하면 위험 큰 관할부터
        top = share.sort_values(ascending=False).head(k)
        return {g: 1 for g in top.index}

    raw = share * k
    base = np.floor(raw).astype(int)
    base[base < 1] = 1                        # 관할마다 최소 한 팀
    left = k - int(base.sum())
    if left > 0:                              # 남는 팀은 잔여가 큰 관할에
        for g in (raw - np.floor(raw)).sort_values(ascending=False).index[:left]:
            base[g] += 1
    elif left < 0:                            # 초과분은 위험 작은 관할에서 회수
        for g in share.sort_values().index:
            while left < 0 and base[g] > 1:
                base[g] -= 1
                left += 1
    return {str(g): int(v) for g, v in base.items()}


def plan_patrol(grids: pd.DataFrame, n_teams: int = 1, *, use_road: bool = True,
                weight_col: str | None = None, seed: int = 42,
                timeout: int = 60, respect_groups: bool = False,
                group_col: str = "sgg") -> dict:
    """지역·팀 단위 순찰 계획.

    grids: lon/lat 을 가진 순찰 대상 격자 (보통 위험 상위 격자)
    반환: {'routes': [팀별 DataFrame], 'summary': DataFrame, 'distance_source': str}
    """
    df = grids.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if df.empty:
        return {"routes": [], "summary": pd.DataFrame(), "distance_source": "none"}

    dist, dur, source = road_distance_matrix(df["lon"], df["lat"],
                                             use_road=use_road, timeout=timeout)

    if respect_groups and group_col in df.columns:
        # 관할 안에서만 팀을 나눈다 — 한 팀이 두 소방서 관할에 걸치지 않게.
        quota = allocate_teams_to_groups(df, n_teams, group_col, weight_col)
        labels = np.full(len(df), -1, dtype=int)
        nxt = 0
        for g, n_g in quota.items():
            mask = (df[group_col].fillna("").astype(str) == g).to_numpy()
            if not mask.any():
                continue
            sub = df.loc[mask]
            sub_labels = partition_teams(sub, n_g, weight_col=weight_col, seed=seed)
            labels[mask] = sub_labels + nxt
            nxt += int(sub_labels.max()) + 1
        # 배정받지 못한 관할(팀보다 관할이 많은 경우)은 가장 가까운 팀에 붙인다.
        if (labels < 0).any():
            done = labels >= 0
            if done.any():
                xy = np.column_stack([df["lon"].astype(float), df["lat"].astype(float)])
                for i in np.where(~done)[0]:
                    j = int(np.argmin(np.linalg.norm(xy[done] - xy[i], axis=1)))
                    labels[i] = labels[done][j]
            else:
                labels[:] = 0
    else:
        labels = partition_teams(df, n_teams, weight_col=weight_col, seed=seed)

    df = df.assign(순찰팀=labels + 1)

    routes, rows = [], []
    for team in sorted(df["순찰팀"].unique()):
        idx = np.where(df["순찰팀"].to_numpy() == team)[0]
        sub_dist = dist[np.ix_(idx, idx)]
        # 팀 구역에서 가장 위험한 격자를 출발점으로 — 순찰은 위험한 곳부터 본다
        start = 0
        if "pred" in df.columns:
            start = int(np.argmax(df.iloc[idx]["pred"].to_numpy()))
        order = solve_route(sub_dist, start=start)

        r = df.iloc[idx[order]].copy().reset_index(drop=True)
        r.insert(0, "순번", range(1, len(r) + 1))
        legs = [0.0] + [float(sub_dist[order[i - 1], order[i]]) for i in range(1, len(order))]
        times = [0.0] + [float(dur[np.ix_(idx, idx)][order[i - 1], order[i]])
                         for i in range(1, len(order))]
        r["이동거리_m"] = np.round(legs, 0)
        r["누적거리_m"] = r["이동거리_m"].cumsum()
        r["이동시간_분"] = np.round(np.array(times) / 60.0, 1)
        r["누적시간_분"] = r["이동시간_분"].cumsum()
        routes.append(r)
        rows.append({
            "순찰팀": int(team),
            "격자수": int(len(r)),
            "총이동거리_km": round(float(r["누적거리_m"].iloc[-1]) / 1000, 1),
            "총이동시간_분": round(float(r["누적시간_분"].iloc[-1]), 0),
            "관할": ", ".join(sorted(set(r["sgg"].dropna().astype(str)))) if "sgg" in r else "",
        })

    summary = pd.DataFrame(rows)
    return {"routes": routes, "summary": summary,
            "distance_source": source,
            "n_teams": len(routes),
            "respect_groups": bool(respect_groups),
            "total_km": round(float(summary["총이동거리_km"].sum()), 1) if len(summary) else 0.0,
            "max_team_km": round(float(summary["총이동거리_km"].max()), 1) if len(summary) else 0.0}
