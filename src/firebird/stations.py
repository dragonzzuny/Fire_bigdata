"""소방관서(소방서·119안전센터) 위치.

순찰은 격자에서 시작하지 않는다. **119안전센터에서 차를 타고 나가 관할을 돌고
센터로 돌아온다.** 출발점을 빼고 짠 동선은 실제 이동거리를 과소평가하고,
"어느 센터가 가느냐"라는 가장 중요한 질문에도 답하지 못한다.

카카오 로컬의 **키워드 검색**을 쓴다. 주소 검색(`search/address`)은 시설명을
못 찾는다 — '울산광역시 삼산119안전센터' 로는 24%만 매칭되지만
키워드 검색은 시설명을 그대로 찾는다.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .geocode import load_api_key

log = logging.getLogger(__name__)

KEYWORD_ENDPOINT = "https://dapi.kakao.com/v2/local/search/keyword.json"
CACHE_NAME = "station_cache.json"

#: 관서 좌표가 자기 관할 격자 중심에서 이보다 멀면 잘못 찾은 것으로 본다.
#: 관할이 넓은 군 지역(울주군)에서도 관서는 관할 안에 있다.
MAX_OFFSET_M = 25_000.0


#: 시설명 비교 시 떼어 낼 행정구역 접두사.
_PREFIXES = ("울산광역시", "울산", "세종특별자치시", "세종", "부산광역시", "부산",
             "대구광역시", "대구", "인천광역시", "인천", "광주광역시", "광주",
             "대전광역시", "대전", "서울특별시", "서울", "경기도", "경기")


def _norm_place(name: str) -> str:
    t = re.sub(r"\s+", "", str(name))
    for p in _PREFIXES:
        if t.startswith(p):
            t = t[len(p):]
            break
    return t


def _score(query: str, place: str) -> int:
    """검색 결과가 찾던 관서가 맞는지 점수로 매긴다.

    카카오 키워드 검색은 첫 결과를 그대로 믿으면 안 된다. '울주소방서' 로
    검색하면 '울산남울주소방서' 가 1순위로 오는데, 이 둘은 다른 관서다.
    실제로 그 오류 때문에 울주소방서의 순찰 동선이 온산 쪽에서 출발했다.

    3 = 행정구역 접두사를 뗀 이름이 같거나, 그 이름으로 **시작**한다.
        ('울산남부소방서' ← '남부소방서', '조치원소방서 전기차충전소' ← '조치원소방서')
        뒤에 붙은 말은 같은 부지 안의 시설이므로 좌표를 그대로 써도 된다.
    1 = 이름이 가운데에 들어 있을 뿐이다.
        ('울산남울주소방서' ← '울주소방서' — 앞에 '남'이 붙은 다른 관서다)
    0 = 아니다

    가운데에 들어 있는 것으로는 부족하다. 3점이 아니면 좌표를 쓰지 않고 관할
    중심으로 물러난다. 엉뚱한 관서에서 출발하는 동선보다, 관할 중심에서
    출발하는 대략의 동선이 낫다.
    """
    q, p = _norm_place(query), _norm_place(place)
    if not q or not p:
        return 0
    if q == p or p.startswith(q):
        return 3
    if q in p:
        return 1
    return 0


def _query_keyword(session: requests.Session, key: str, query: str,
                   *, name: str = "", timeout: int = 10, size: int = 5) -> dict:
    """관서명으로 좌표를 찾는다. 후보 여러 개를 받아 이름이 맞는 것을 고른다."""
    try:
        params = {"query": query, "size": size}
        r = session.get(KEYWORD_ENDPOINT, headers={"Authorization": f"KakaoAK {key}"},
                        params=params, timeout=timeout)
        if r.status_code == 429:
            time.sleep(2)
            r = session.get(KEYWORD_ENDPOINT, headers={"Authorization": f"KakaoAK {key}"},
                            params=params, timeout=timeout)
        r.raise_for_status()
        docs = r.json().get("documents", [])
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {"lon": None, "lat": None, "matched": False, "reason": str(exc)[:80]}
    if not docs:
        return {"lon": None, "lat": None, "matched": False, "reason": "no_result"}

    want = name or query
    best, best_s = None, -1
    for d in docs:
        sc = _score(want, d.get("place_name", ""))
        if sc > best_s:
            best, best_s = d, sc
    # 이름이 그저 '들어 있는' 수준이면 다른 관서일 수 있다. 좌표를 쓰지 않는다.
    if best is None or best_s < 3:
        return {"lon": None, "lat": None, "matched": False,
                "reason": f"name_mismatch(best={best.get('place_name','') if best else ''})",
                "place_name": best.get("place_name", "") if best else "",
                "address": (best.get("road_address_name")
                            or best.get("address_name", "")) if best else ""}
    return {"lon": float(best["x"]), "lat": float(best["y"]), "matched": True,
            "match_score": best_s,
            "place_name": best.get("place_name", ""),
            "address": best.get("road_address_name") or best.get("address_name", "")}


def locate_stations(names: list[str], cfg, *, city_label: str = "",
                    allow_network: bool = True) -> pd.DataFrame:
    """관서명 목록 -> DataFrame[name, lon, lat, matched, place_name, address].

    캐시에 있으면 다시 부르지 않는다. 네트워크가 막히면 캐시분만 돌려준다.
    """
    cache_path = cfg.paths.cache / CACHE_NAME
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("관서 캐시 손상 — 새로 시작")

    names = [str(n).strip() for n in dict.fromkeys(names) if str(n).strip()]

    # 예전 기준으로 받아 둔 캐시에는 이름이 다른 관서가 섞여 있다.
    # (예: '울주소방서' 로 검색해 '울산남울주소방서' 를 받아 둔 것)
    # 읽을 때 다시 채점해, 기준에 못 미치면 버리고 새로 찾는다.
    def _stale(n: str, v: dict) -> bool:
        sc = _score(n, v.get("place_name", ""))
        if v.get("matched"):
            return sc < 3                      # 예전 기준으로 받아들인 잘못된 결과
        # 기준이 느슨해져 이제는 받아들일 수 있는 결과. 좌표가 없으니 다시 찾는다.
        return str(v.get("reason", "")).startswith("name_mismatch") and sc >= 3

    # 이번에 찾는 관서만 손댄다. 캐시 전체를 훑어 지우면, 지금 다시 찾지도 않을
    # 다른 도시의 관서까지 지워져 매번 API 를 부르게 된다.
    stale = [n for n in names if n in cache and _stale(n, cache[n])]
    for n in stale:
        log.info("관서 좌표 재확인: %s -> %r", n, cache[n].get("place_name", ""))
        cache.pop(n, None)

    todo = [n for n in names if n not in cache]

    if todo and allow_network:
        api_key = load_api_key()
        if not api_key:
            log.warning("KAKAO_REST_API_KEY 가 없어 관서 위치를 못 찾는다")
        else:
            session = requests.Session()
            for n in todo:
                # 시도명을 앞에 붙여야 동명 관서를 구분한다(전국에 '중부소방서'가 여럿 있다).
                cache[n] = _query_keyword(session, api_key,
                                          f"{city_label} {n}".strip(), name=n)
                time.sleep(0.12)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0),
                                  encoding="utf-8")

    rows = []
    for n in names:
        v = cache.get(n) or {}
        rows.append({"name": n, "lon": v.get("lon"), "lat": v.get("lat"),
                     "matched": bool(v.get("matched")),
                     "place_name": v.get("place_name", ""),
                     "address": v.get("address", "")})
    return pd.DataFrame(rows)


def station_table(panel_year: pd.DataFrame, cfg, *, level: str = "center",
                  city_label: str = "", allow_network: bool = True) -> pd.DataFrame:
    """관할 격자 통계 + 관서 좌표.

    관서 좌표를 못 찾으면 그 관할 격자의 중심을 대신 쓴다 — 순찰 계획이
    좌표 하나 때문에 통째로 막히면 안 된다. 어느 쪽인지는 `coord_source` 에 남긴다.
    """
    if level not in panel_year.columns:
        return pd.DataFrame()
    df = panel_year[panel_year[level].astype(str).str.strip() != ""]
    if df.empty:
        return pd.DataFrame()

    agg = (df.groupby(level)
             .agg(격자수=("grid_id", "nunique"),
                  위험합=("pred", "sum") if "pred" in df.columns else ("grid_id", "size"),
                  점검대상=("target_total", "sum") if "target_total" in df.columns
                  else ("grid_id", "size"),
                  실제화재=("fires", "sum") if "fires" in df.columns else ("grid_id", "size"),
                  중심경도=("lon", "mean"), 중심위도=("lat", "mean"))
             .reset_index().rename(columns={level: "name"}))

    loc = locate_stations(agg["name"].tolist(), cfg, city_label=city_label,
                          allow_network=allow_network)
    out = agg.merge(loc[["name", "lon", "lat", "matched", "address"]], on="name", how="left")
    out["matched"] = out["matched"].fillna(False)

    # 이름이 맞아도 좌표가 자기 관할과 멀면 다른 곳을 찾은 것이다.
    # 관할 격자 중심에서 지나치게 먼 좌표는 쓰지 않는다.
    far = pd.Series(False, index=out.index)
    m = out["matched"] & out["lon"].notna()
    if m.any():
        lat0 = float(np.nanmean(out.loc[m, "lat"].astype(float)))
        dx = ((out.loc[m, "lon"].astype(float) - out.loc[m, "중심경도"])
              * 88800.0 * np.cos(np.radians(lat0)))
        dy = (out.loc[m, "lat"].astype(float) - out.loc[m, "중심위도"]) * 111000.0
        far.loc[m] = np.sqrt(dx ** 2 + dy ** 2) > MAX_OFFSET_M
        for name in out.loc[far, "name"]:
            log.warning("관서 좌표가 관할 중심에서 %.0fkm 넘게 떨어져 있어 "
                        "관할 중심을 쓴다: %s", MAX_OFFSET_M / 1000, name)

    bad = ~out["matched"] | far
    out.loc[bad, ["lon", "lat", "address"]] = [np.nan, np.nan, ""]
    out["matched"] = ~bad
    out["coord_source"] = np.where(
        out["matched"], "관서 위치",
        np.where(far, "관할 중심(좌표 이상)", "관할 중심(검색 실패)"))
    out["lon"] = out["lon"].fillna(out["중심경도"])
    out["lat"] = out["lat"].fillna(out["중심위도"])
    return out.drop(columns=["중심경도", "중심위도"]).sort_values(
        "위험합", ascending=False).reset_index(drop=True)


def assign_dispatch(targets: pd.DataFrame, stations: pd.DataFrame, *,
                    level: str = "center", respect_jurisdiction: bool = True) -> pd.DataFrame:
    """순찰 대상 격자마다 '어느 관서가 나가는 게 좋은가'를 붙인다.

    기본은 관할 존중이다. 관할을 넘겨 배정하면 지휘 계통이 갈리고 무전·보고가
    꼬인다. 다만 관할 관서가 없거나(자료 결측) 관할을 무시하도록 지정하면
    가장 가까운 관서로 보낸다.
    """
    if targets.empty or stations.empty:
        return targets.assign(출동관서="", 관서까지_km=np.nan)

    out = targets.copy()
    sxy = np.column_stack([stations["lon"].astype(float), stations["lat"].astype(float)])
    snames = stations["name"].to_numpy()

    lat0 = float(np.nanmean(out["lat"].astype(float)))
    scale = np.array([88800.0 * np.cos(np.radians(lat0)), 111000.0])

    assigned, dist_km = [], []
    for _, row in out.iterrows():
        here = np.array([float(row["lon"]), float(row["lat"])])
        own = str(row.get(level, "")).strip()
        if respect_jurisdiction and own and own in set(snames):
            j = int(np.where(snames == own)[0][0])
        else:
            d = np.linalg.norm((sxy - here) * scale, axis=1)
            j = int(np.argmin(d))
        assigned.append(snames[j])
        dist_km.append(float(np.linalg.norm((sxy[j] - here) * scale)) / 1000.0)

    out["출동관서"] = assigned
    out["관서까지_km"] = np.round(dist_km, 1)
    return out
