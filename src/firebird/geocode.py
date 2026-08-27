"""카카오 로컬 API 지오코딩 + 디스크 캐시.

(시군구·도로명) 키 단위로만 질의한다. 같은 도로의 수천 건 주소가
하나의 질의로 줄고, 캐시가 있으면 재실행 때 API 를 다시 부르지 않는다.
API 키가 없으면 캐시에 있는 것만 채우고 나머지는 결측으로 남긴다
(파이프라인은 좌표 확보율을 보고서에 기록한다).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)


def load_api_key(env_name: str = "KAKAO_REST_API_KEY") -> str | None:
    """환경변수 또는 repo 루트 .env 에서 키를 읽는다."""
    key = os.environ.get(env_name)
    if key:
        return key.strip()
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == env_name:
                return v.strip().strip('"').strip("'")
    return None


class GeocodeCache:
    """geo_key -> {lon, lat, matched} JSON 캐시. 매 저장마다 원자적 교체."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("캐시가 손상됨, 새로 시작: %s", self.path)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(self.path)


def drop_sgg(key: str) -> str:
    """'시도 시군구 도로' -> '시도 도로'. 토큰이 3개일 때만 의미가 있다."""
    parts = key.split()
    if len(parts) < 3:
        return ""
    return f"{parts[0]} {parts[-1]}"


def _query_kakao(session: requests.Session, key: str, query: str,
                 endpoint: str, max_retries: int) -> dict:
    """한 건 질의. 실패는 matched=False 로 남기고 예외를 올리지 않는다."""
    headers = {"Authorization": f"KakaoAK {key}"}
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(endpoint, headers=headers, params={"query": query, "size": 1}, timeout=10)
            if r.status_code == 429:                     # 쿼터 초과
                time.sleep(2 * attempt)
                continue
            r.raise_for_status()
            docs = r.json().get("documents", [])
            if not docs:
                return {"lon": None, "lat": None, "matched": False, "reason": "no_result"}
            d = docs[0]
            return {"lon": float(d["x"]), "lat": float(d["y"]), "matched": True,
                    "reason": d.get("address_type", "")}
        except (requests.RequestException, ValueError, KeyError) as exc:
            if attempt == max_retries:
                return {"lon": None, "lat": None, "matched": False, "reason": f"error:{exc}"}
            time.sleep(1.0 * attempt)
    return {"lon": None, "lat": None, "matched": False, "reason": "exhausted"}


def geocode_keys(keys: list[str], cfg, *, api_key: str | None = None,
                 allow_network: bool = True) -> pd.DataFrame:
    """지오코딩 키 목록 -> DataFrame[geo_key, lon, lat, matched].

    캐시에 있는 키는 건너뛴다. 키가 없으면 미캐시 항목은 결측으로 남는다.
    allow_network=False 면 환경변수에 키가 있어도 절대 호출하지 않는다 —
    '캐시만 쓴다'는 약속이 환경에 따라 깨지면 그건 약속이 아니다.
    """
    gconf = cfg["geocode"]
    cache = GeocodeCache(cfg.paths.cache / gconf["cache_file"])
    unique = [k for k in dict.fromkeys(k for k in keys if k) ]
    todo = [k for k in unique if k not in cache]

    if not allow_network:
        if todo:
            log.info("네트워크 비활성(allow_network=False): 미캐시 %d건은 결측으로 남긴다", len(todo))
        todo = []
        api_key = None
    else:
        api_key = api_key or load_api_key()
    if todo and not api_key:
        log.warning(
            "KAKAO_REST_API_KEY 가 없다. 캐시에 있는 %d/%d 건만 좌표를 얻는다. "
            "https://developers.kakao.com 에서 REST 키를 발급해 .env 에 넣어라.",
            len(unique) - len(todo), len(unique))
    elif todo:
        log.info("지오코딩 대상 %d건 (캐시 적중 %d건)", len(todo), len(unique) - len(todo))
        sleep_s = 1.0 / max(1, int(gconf.get("rate_limit_per_sec", 8)))
        session = requests.Session()
        endpoint = gconf["endpoint"]
        retries = int(gconf.get("max_retries", 3))
        for i, key in enumerate(todo, 1):
            res = _query_kakao(session, api_key, key, endpoint, retries)
            # '시도 시군구 도로'가 실패하면 시군구를 빼고 한 번 더 묻는다.
            # 도로가 행정구역 경계를 넘거나(예: 온산읍 처용로) 원본의 시군구가
            # 틀린 경우가 흔한데, 그것 때문에 좌표를 통째로 버릴 이유는 없다.
            if not res.get("matched"):
                reduced = drop_sgg(key)
                if reduced and reduced != key:
                    time.sleep(sleep_s)
                    alt = _query_kakao(session, api_key, reduced, endpoint, retries)
                    if alt.get("matched"):
                        alt["reason"] = f"fallback_no_sgg:{alt.get('reason','')}"
                        res = alt
            cache.put(key, res)
            time.sleep(sleep_s)
            if i % 200 == 0:
                cache.save()
                log.info("  %d/%d", i, len(todo))
        cache.save()

    rows = []
    for k in unique:
        v = cache.get(k) or {"lon": None, "lat": None, "matched": False}
        rows.append({"geo_key": k, "lon": v.get("lon"), "lat": v.get("lat"),
                     "matched": bool(v.get("matched"))})
    return pd.DataFrame(rows)


def attach_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """geo_key 기준으로 lon/lat 을 붙인다."""
    return df.merge(coords[["geo_key", "lon", "lat"]], on="geo_key", how="left")


def coverage(df: pd.DataFrame) -> dict[str, float]:
    """좌표 확보율. 기획서의 95.3% 에 대응하는 수치를 보고서에 남긴다.

    lon 만 있고 lat 이 없는 행은 격자에 배정되지 않는다. 그런 행을 '확보'로
    세면 manifest 가 실제보다 낙관적인 숫자를 말하게 되므로 둘 다 요구한다.
    """
    n = len(df)
    if {"lon", "lat"}.issubset(df.columns):
        got = int((df["lon"].notna() & df["lat"].notna()).sum())
    else:
        got = 0
    return {"rows": n, "with_coords": got, "rate": (got / n) if n else 0.0}
