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

def nearest_neighbor(dist: np.ndarray, start: int = 0) -> list[int]:
    """최근접 이웃 생성해. 빠르지만 마지막에 먼 곳으로 튀는 구간이 남는다."""
    n = len(dist)
    order = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        cur = order[-1]
        nxt = min(remaining, key=lambda j: dist[cur, j])
        order.append(nxt)
        remaining.discard(nxt)
    return order


def _randomized_greedy(dist: np.ndarray, start: int, rng: np.random.Generator,
                       top: int = 2) -> list[int]:
    """최근접 이웃을 흔든 것. 매번 가장 가까운 곳 대신 가까운 두 곳 중 하나로 간다.

    열린 경로에서는 출발점을 바꿀 수 없으므로(관서가 정해져 있다) 다중 출발
    대신 이 방법으로 서로 다른 초기해를 만든다.
    """
    n = len(dist)
    order = [start]
    remaining = list(set(range(n)) - {start})
    while remaining:
        cur = order[-1]
        remaining.sort(key=lambda j: dist[cur, j])
        pick = int(rng.integers(0, min(top, len(remaining))))
        order.append(remaining.pop(pick))
    return order


def route_length(dist: np.ndarray, order: list[int], *, closed: bool = False) -> float:
    """경로 길이. closed=True 면 마지막 지점에서 출발점으로 돌아오는 구간까지 더한다.

    순찰은 관서에서 나가 관서로 돌아온다. 돌아오는 거리를 빼고 최적화하면
    '가장 먼 곳에서 끝나는' 동선이 최선으로 뽑히는데, 실제로는 그만큼 되돌아와야
    하므로 총 이동은 오히려 늘어난다. 무엇을 최적화하는지와 무엇을 보고하는지가
    같아야 한다.
    """
    if len(order) < 2:
        return 0.0
    total = float(sum(dist[order[i - 1], order[i]] for i in range(1, len(order))))
    if closed:
        total += float(dist[order[-1], order[0]])
    return total


def solve_route(dist: np.ndarray, start: int = 0, *, restarts: int = 0,
                or_opt: bool = True, max_rounds: int = 60,
                closed: bool = False, seed: int = 42) -> list[int]:
    """Depot 고정 TSP 를 휴리스틱으로 푼다.

    구성:  최근접 이웃(여러 초기해) → 2-opt → Or-opt, 더 나아지지 않을 때까지 반복.

    · **2-opt** 는 경로의 두 변을 끊고 사이를 뒤집는다. 교차(X자)를 없앤다.
      열린 경로에서는 '꼬리를 통째로 뒤집는' 수도 유효한데, 이 수를 빼면
      최적해에서 20% 넘게 벌어지는 사례가 실제로 나온다
      (tests_firebird/test_algorithms.py 에서 완전탐색과 대조).
    · **Or-opt** 는 연속한 1~3개 지점을 통째로 다른 위치로 옮긴다. 방향을 뒤집어
      넣는 경우까지 본다. 2-opt 로는 못 고치는 '한 곳만 동떨어져 붙은' 구간을 푼다.
    둘은 서로 못 고치는 것을 고쳐서, 같이 쓰면 각각보다 낫다.

    closed=True 면 왕복(관서 → … → 관서)을 최소화한다. 순찰 계획은 왕복이다.

    최적해를 보장하지 않는다. TSP 는 NP-난해여서 격자 수십 개라도 정확해를 구하려면
    분기한정이나 전용 솔버가 필요한데, 순찰 계획에서 몇 백 미터 차이는 도로 사정·
    신호·주차에 묻힌다. 대신 **빠르고 안정적으로** 좋은 해를 낸다.
    지점 9개 이하에서 완전탐색과 대조한 결과는 위 시험 파일에 있다.

    start(=0, 보통 관서)는 항상 경로의 첫 지점으로 고정된다.
    """
    n = len(dist)
    if n <= 2:
        return list(range(n))

    rng = np.random.default_rng(seed)
    inits = [nearest_neighbor(dist, start)]
    if closed:
        # 순환이므로 다른 지점에서 만든 해도 회전시키면 같은 출발점을 갖는다.
        others = [i for i in range(n) if i != start]
        step = max(1, len(others) // max(1, restarts)) if restarts else 1
        for seed_node in (others[::step][:restarts] if restarts else []):
            o = nearest_neighbor(dist, seed_node)
            z = o.index(start)
            inits.append(o[z:] + o[:z])
    for _ in range(restarts):
        inits.append(_randomized_greedy(dist, start, rng))

    best, best_len = None, float("inf")
    for order in inits:
        order = _improve(dist, order, or_opt=or_opt, max_rounds=max_rounds, closed=closed)
        length = route_length(dist, order, closed=closed)
        if length < best_len:
            best, best_len = order, length
    return best


def _improve(dist: np.ndarray, order: list[int], *, or_opt: bool = True,
             max_rounds: int = 60, closed: bool = False) -> list[int]:
    """2-opt 와 Or-opt 를 번갈아 돌린다. 더 이상 줄지 않으면 멈춘다."""
    cur = list(order)
    for _ in range(max_rounds):
        before = route_length(dist, cur, closed=closed)
        cur = _two_opt_matrix(dist, cur, max_rounds=1, closed=closed)
        if or_opt:
            cur = _or_opt_matrix(dist, cur, closed=closed)
        if route_length(dist, cur, closed=closed) >= before - 1e-9:
            break
    return cur


def _or_opt_matrix(dist: np.ndarray, order: list[int], max_seg: int = 3, *,
                   closed: bool = False) -> list[int]:
    """연속한 최대 3개 지점을 다른 위치로 옮겨 본다 (출발점은 고정).

    옮길 때 구간의 방향을 뒤집는 경우까지 본다 — 도로거리는 방향에 따라 다르고,
    뒤집어 넣어야 짧아지는 배치가 실제로 있다.
    """
    best = list(order)
    n = len(best)
    improved = True
    while improved:
        improved = False
        cur_len = route_length(dist, best, closed=closed)
        for seg in range(1, max_seg + 1):
            for i in range(1, n - seg + 1):
                block = best[i:i + seg]
                rest = best[:i] + best[i + seg:]
                for piece in ((block, block[::-1]) if seg > 1 else (block,)):
                    for j in range(1, len(rest) + 1):
                        if j == i and piece is block:
                            continue
                        cand = rest[:j] + list(piece) + rest[j:]
                        if cand[0] != best[0]:      # 출발점이 앞이어야 한다
                            continue
                        if route_length(dist, cand, closed=closed) < cur_len - 1e-9:
                            best, improved = cand, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best


def _two_opt_matrix(dist: np.ndarray, order: list[int], max_rounds: int = 60, *,
                    closed: bool = False) -> list[int]:
    """행렬 기반 2-opt. operations.two_opt 은 좌표 기반이라 도로거리를 못 쓴다.

    구간 [i, j] 를 뒤집는다. 뒤집기 전후로 바뀌는 변은 (i-1, i) 와 (j, j+1) 둘뿐이다.
    열린 경로에서 j 가 마지막 지점이면 (j, j+1) 변이 없으므로 첫 변만 비교한다 —
    이 경우를 빼면 '꼬리가 통째로 뒤집혀야 하는' 배치를 영영 못 고친다.
    """
    best = list(order)
    n = len(best)
    improved, rounds = True, 0
    while improved and rounds < max_rounds:
        improved, rounds = False, rounds + 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c = best[i - 1], best[i], best[j]
                if j + 1 < n:
                    d = best[j + 1]
                elif closed:
                    d = best[0]                     # 순환이면 마지막 다음은 출발점
                else:
                    # 열린 경로의 꼬리 뒤집기 — 끝쪽 변이 없다.
                    if dist[a, b] > dist[a, c] + 1e-9:
                        best[i:] = best[i:][::-1]
                        improved = True
                    continue
                if dist[a, b] + dist[c, d] > dist[a, c] + dist[b, d] + 1e-9:
                    best[i:j + 1] = list(reversed(best[i:j + 1]))
                    improved = True
    return best


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


def split_by_budget(order: list[int], dur: np.ndarray, dist: np.ndarray,
                    budget_min: float, *, stay_min: float = 5.0) -> list[list[int]]:
    """한 번에 다 못 도는 경로를 근무시간 안에 들어오도록 회차로 쪼갠다.

    순찰은 근무 중 한 토막이다. '3시간짜리 동선'을 주면 현장에서는 그냥 안 쓴다.
    각 회차는 관서(0)에서 출발해 관서로 복귀하는 것으로 계산한다.
    stay_min 은 격자 한 곳에서 머무는 시간(하차·확인·기록).
    """
    if budget_min <= 0 or len(order) <= 1:
        return [order]
    trips, cur = [], [order[0]]
    elapsed = 0.0
    for node in order[1:]:
        leg = float(dur[cur[-1], node]) / 60.0 + stay_min
        back = float(dur[node, order[0]]) / 60.0
        if cur[-1] != order[0] and elapsed + leg + back > budget_min:
            trips.append(cur)
            cur = [order[0], node]
            elapsed = float(dur[order[0], node]) / 60.0 + stay_min
        else:
            cur.append(node)
            elapsed += leg
    if len(cur) > 1:
        trips.append(cur)
    return trips or [order]


def plan_from_stations(grids: pd.DataFrame, stations: pd.DataFrame, *,
                       level: str = "center", use_road: bool = True,
                       timeout: int = 60, teams_per_station: int = 1,
                       budget_min: float = 0.0, stay_min: float = 5.0,
                       restarts: int = 4) -> dict:
    """관서에서 출발해 관할을 돌고 관서로 돌아오는 왕복 순찰 계획.

    순찰은 격자에서 시작하지 않는다. 119안전센터에서 차를 타고 나가 돌고
    돌아온다. 출발점을 빼고 짠 동선은 실제 이동거리를 과소평가하고,
    "어느 센터가 나가느냐"라는 가장 중요한 질문에도 답하지 못한다.

    grids: 순찰 대상 격자 (level 컬럼으로 관할이 정해져 있어야 한다)
    stations: stations.station_table() 결과 (name/lon/lat)
    """
    df = grids.dropna(subset=["lon", "lat"]).copy()
    if df.empty or stations.empty or level not in df.columns:
        return {"routes": [], "summary": pd.DataFrame(), "distance_source": "none",
                "n_teams": 0, "total_km": 0.0, "max_team_km": 0.0}

    sloc = stations.dropna(subset=["lon", "lat"]).set_index("name")
    routes, rows = [], []
    source = "osrm" if use_road else "straight_line_x_detour"

    for name, g in df.groupby(level):
        name = str(name).strip()
        if not name or name not in sloc.index or g.empty:
            continue
        st = sloc.loc[name]
        # 0번을 관서로 두고 격자를 뒤에 붙인다. 왕복이므로 마지막에 관서로 복귀한다.
        lons = np.r_[float(st["lon"]), g["lon"].astype(float).to_numpy()]
        lats = np.r_[float(st["lat"]), g["lat"].astype(float).to_numpy()]
        dist, dur, src = road_distance_matrix(lons, lats, use_road=use_road, timeout=timeout)
        if src != "osrm":
            source = src

        n_t = max(1, int(teams_per_station))
        if n_t > 1 and len(g) > n_t:
            labels = partition_teams(g.reset_index(drop=True), n_t,
                                     weight_col="순찰점수" if "순찰점수" in g else None)
        else:
            labels = np.zeros(len(g), dtype=int)

        for t in sorted(set(labels)):
            local = np.where(labels == t)[0]            # g 안의 위치
            idx = np.r_[0, local + 1]                   # 0 = 관서
            sub = dist[np.ix_(idx, idx)]
            subt = dur[np.ix_(idx, idx)]
            # 왕복(관서 → 격자들 → 관서)을 최소화한다. 편도만 줄이면 돌아오는
            # 거리가 늘어 총 이동이 오히려 커진다.
            order = solve_route(sub, start=0, restarts=restarts, closed=True)
            if order[0] != 0:
                z = order.index(0)
                order = order[z:] + order[:z]

            trips = (split_by_budget(order, subt, sub, budget_min, stay_min=stay_min)
                     if budget_min > 0 else [order])

            for trip_no, trip in enumerate(trips, 1):
                if len(trip) < 2:
                    continue
                r = g.iloc[local[np.array(trip[1:]) - 1]].copy().reset_index(drop=True)
                r.insert(0, "순번", range(1, len(r) + 1))
                legs = [float(sub[trip[i - 1], trip[i]]) for i in range(1, len(trip))]
                times = [float(subt[trip[i - 1], trip[i]]) / 60.0 + stay_min
                         for i in range(1, len(trip))]
                back_km = float(sub[trip[-1], 0]) / 1000.0
                back_min = float(subt[trip[-1], 0]) / 60.0

                r["이동거리_m"] = np.round(legs, 0)
                r["누적거리_m"] = r["이동거리_m"].cumsum()
                r["이동시간_분"] = np.round(times, 1)
                r["누적시간_분"] = r["이동시간_분"].cumsum()
                r["출동관서"] = name
                r["회차"] = trip_no
                r.attrs["depot"] = {"name": name, "lon": float(st["lon"]),
                                    "lat": float(st["lat"])}
                routes.append(r)

                rows.append({
                    "출동관서": name,
                    "팀": int(t) + 1,
                    "회차": trip_no,
                    "격자수": int(len(r)),
                    "관서→첫격자_km": round(legs[0] / 1000, 1) if legs else 0.0,
                    "순찰거리_km": round(float(r["누적거리_m"].iloc[-1]) / 1000, 1),
                    "복귀거리_km": round(back_km, 1),
                    "총_km": round(float(r["누적거리_m"].iloc[-1]) / 1000 + back_km, 1),
                    "총_분": round(float(r["누적시간_분"].iloc[-1]) + back_min, 0),
                })

    summary = pd.DataFrame(rows)
    return {"routes": routes, "summary": summary, "distance_source": source,
            "n_teams": len(routes),
            "total_km": round(float(summary["총_km"].sum()), 1) if len(summary) else 0.0,
            "max_team_km": round(float(summary["총_km"].max()), 1) if len(summary) else 0.0,
            "from_stations": True}


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
        # 이 계획은 관서 복귀를 포함하지 않는 '격자 간 이동'이므로 편도로 푼다.
        order = solve_route(sub_dist, start=start, restarts=2)

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
