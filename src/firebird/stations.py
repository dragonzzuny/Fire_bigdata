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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .geocode import load_api_key

log = logging.getLogger(__name__)

KEYWORD_ENDPOINT = "https://dapi.kakao.com/v2/local/search/keyword.json"
CACHE_NAME = "station_cache.json"


def _query_keyword(session: requests.Session, key: str, query: str,
                   *, timeout: int = 10) -> dict:
    try:
        r = session.get(KEYWORD_ENDPOINT, headers={"Authorization": f"KakaoAK {key}"},
                        params={"query": query, "size": 1}, timeout=timeout)
        if r.status_code == 429:
            time.sleep(2)
            r = session.get(KEYWORD_ENDPOINT, headers={"Authorization": f"KakaoAK {key}"},
                            params={"query": query, "size": 1}, timeout=timeout)
        r.raise_for_status()
        docs = r.json().get("documents", [])
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {"lon": None, "lat": None, "matched": False, "reason": str(exc)[:80]}
    if not docs:
        return {"lon": None, "lat": None, "matched": False, "reason": "no_result"}
    d = docs[0]
    return {"lon": float(d["x"]), "lat": float(d["y"]), "matched": True,
            "place_name": d.get("place_name", ""),
            "address": d.get("road_address_name") or d.get("address_name", "")}


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
    todo = [n for n in names if n not in cache]

    if todo and allow_network:
        api_key = load_api_key()
        if not api_key:
            log.warning("KAKAO_REST_API_KEY 가 없어 관서 위치를 못 찾는다")
        else:
            session = requests.Session()
            for n in todo:
                # 시도명을 앞에 붙여야 동명 관서를 구분한다(전국에 '중부소방서'가 여럿 있다).
                cache[n] = _query_keyword(session, api_key, f"{city_label} {n}".strip())
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
    out["coord_source"] = np.where(out["matched"].fillna(False), "관서 위치", "관할 중심")
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
