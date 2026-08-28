"""건축물대장 연계 — 연면적·건축면적·건축연도.

관리대장 [별지 제11호서식] 의 연면적·건축면적·건축연도 칸은 공개된 소방
데이터에 없다. 없다고 비워 두는 것과, 어디서 가져오면 되는지 알고도 안 하는
것은 다르다. 국토교통부 건축물대장은 무료 공개 API 이므로 가져올 수 있다.

**격자에 어떻게 붙이는가**

건축물대장은 (시군구코드, 법정동코드) 단위로 준다. 우리 분석 단위는 500m
격자이고 법정동보다 작다. 법정동 합계를 격자에 그대로 쓰면 한 동의 연면적이
그 동의 모든 격자에 중복으로 들어간다 — 결재 문서에 들어가면 안 되는 값이다.

그래서 건물 하나하나의 주소를 좌표로 바꿔 격자에 배정한 뒤 격자별로 합산한다.
대상물·업소를 처리한 것과 같은 방식이고, 같은 지오코딩 캐시를 쓴다.

**키가 없으면 아무것도 하지 않는다.** 서식의 해당 칸은 지금처럼 비어 있고
'건축물대장 미연계'로 남는다. 반쯤 채워진 값을 넣는 것보다 낫다.
키 신청: https://www.data.go.kr/data/15044713/openapi.do (국토교통부_건축물대장정보 서비스)
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from . import geocode as GC
from . import grid as G

log = logging.getLogger(__name__)

#: 표제부 조회. 건축물대장정보 서비스(v2).
TITLE_URL = "http://apis.data.go.kr/1613000/BldRgstService_v2/getBrTitleInfo"
#: 좌표 -> 법정동코드. 격자 중심에서 어느 동인지 알아낸다.
KAKAO_REGION_URL = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"

KEY_ENV = "DATA_GO_KR_KEY"
CACHE_NAME = "buildings"

#: 응답에서 쓰는 항목만 남긴다. 전부 담으면 캐시가 커지고 읽기 어렵다.
FIELDS = {
    "platPlc": "지번주소",
    "newPlatPlc": "도로명주소",
    "bldNm": "건물명",
    "totArea": "연면적",
    "archArea": "건축면적",
    "useAprDay": "사용승인일",
    "grndFlrCnt": "지상층수",
    "mainPurpsCdNm": "주용도",
}


def load_key() -> str | None:
    """공공데이터포털 서비스 키. 없으면 None."""
    return GC.load_api_key(KEY_ENV)


def region_code(lon: float, lat: float, kakao_key: str,
                *, timeout: int = 10) -> tuple[str, str] | None:
    """좌표 -> (시군구코드 5자리, 법정동코드 5자리).

    법정동코드 10자리의 앞 5자리가 시군구, 뒤 5자리가 법정동이다.
    """
    try:
        r = requests.get(KAKAO_REGION_URL,
                         headers={"Authorization": f"KakaoAK {kakao_key}"},
                         params={"x": f"{lon:.6f}", "y": f"{lat:.6f}"},
                         timeout=timeout)
        r.raise_for_status()
        docs = r.json().get("documents", [])
    except (requests.RequestException, ValueError) as exc:
        log.warning("법정동코드 조회 실패: %s", type(exc).__name__)
        return None
    for d in docs:
        if d.get("region_type") == "B":          # B = 법정동
            code = str(d.get("code", ""))
            if len(code) >= 10:
                return code[:5], code[5:10]
    return None


def _rows(xml_text: str) -> tuple[list[dict], int]:
    """표제부 응답에서 항목과 전체 건수. 실패하면 빈 목록."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], 0
    code = (root.findtext(".//resultCode") or "").strip()
    if code not in ("00", "0", ""):
        log.warning("건축물대장 응답 코드 %s: %s", code,
                    (root.findtext(".//resultMsg") or "").strip()[:80])
        return [], 0
    total = int((root.findtext(".//totalCount") or "0").strip() or 0)
    out = []
    for item in root.findall(".//item"):
        row = {ko: (item.findtext(en) or "").strip() for en, ko in FIELDS.items()}
        if any(row.values()):
            out.append(row)
    return out, total


def fetch_title(sigungu_cd: str, bjdong_cd: str, key: str, *,
                rows_per_page: int = 1000, max_pages: int = 20,
                timeout: int = 30, pause: float = 0.2) -> pd.DataFrame:
    """한 법정동의 표제부를 모두 받는다."""
    got: list[dict] = []
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(TITLE_URL, timeout=timeout, params={
                "serviceKey": key, "sigunguCd": sigungu_cd, "bjdongCd": bjdong_cd,
                "numOfRows": rows_per_page, "pageNo": page, "_type": "xml"})
            r.raise_for_status()
        except requests.RequestException as exc:
            log.warning("건축물대장 호출 실패(%s쪽): %s", page, type(exc).__name__)
            break
        rows, total = _rows(r.text)
        got += rows
        if not rows or len(got) >= total:
            break
        time.sleep(pause)
    df = pd.DataFrame(got)
    if df.empty:
        return df
    for c in ("연면적", "건축면적", "지상층수"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["사용승인연도"] = pd.to_numeric(
        df["사용승인일"].str.slice(0, 4), errors="coerce")
    return df


def collect(cfg, grid_ids, *, refresh: bool = False,
            allow_network: bool = True) -> pd.DataFrame:
    """격자 목록이 걸치는 법정동들의 건축물대장을 모아 격자에 배정한다.

    반환: 격자별 집계 (연면적, 건축면적, 사용승인연도, 건물수).
    키가 없거나 네트워크가 막히면 빈 표를 돌려준다 — 서식은 그대로 비워 둔다.
    """
    cache = cfg.paths.cache / f"{CACHE_NAME}_grid.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        return df[df["grid_id"].isin(list(grid_ids))] if len(grid_ids) else df

    key, kakao = load_key(), GC.load_api_key()
    if not (key and kakao and allow_network):
        log.info("건축물대장 연계 건너뜀 (키 없음 또는 --no-api)")
        return pd.DataFrame()

    centers = G.grid_centers(pd.Series(list(dict.fromkeys(grid_ids))), cfg)
    seen: set[tuple[str, str]] = set()
    frames = []
    for r in centers.itertuples():
        rc = region_code(float(r.lon), float(r.lat), kakao)
        if rc is None or rc in seen:
            continue
        seen.add(rc)
        got = fetch_title(rc[0], rc[1], key)
        if not got.empty:
            frames.append(got)
        time.sleep(0.15)
    if not frames:
        return pd.DataFrame()

    b = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["지번주소"])
    b["geo_key"] = b["도로명주소"].where(b["도로명주소"].str.strip() != "",
                                     b["지번주소"]).str.strip()
    coords = GC.geocode_keys(sorted(x for x in b["geo_key"].unique() if x), cfg,
                             allow_network=allow_network)
    b = GC.attach_coordinates(b, coords)
    b = b.dropna(subset=["lon", "lat"])
    if b.empty:
        return pd.DataFrame()
    b = G.assign_grid(b, cfg)

    # 열 이름에 ㎡ 를 쓰면 파이썬 식별자로 못 쓴다. 단위는 서식이 인쇄해 준다.
    out = (b.groupby("grid_id")
             .agg(**{"연면적": ("연면적", "sum"),
                     "건축면적": ("건축면적", "sum"),
                     "사용승인연도": ("사용승인연도", "median"),
                     "건물수": ("지번주소", "size")})
             .reset_index())
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    log.info("건축물대장 %d개 법정동 → 격자 %d개", len(seen), len(out))
    return out


def stats_for(df: pd.DataFrame, grid_id: str) -> dict:
    """서식에 넣을 형태로. 없으면 빈 사전."""
    if df is None or df.empty or "grid_id" not in df.columns:
        return {}
    hit = df[df["grid_id"].astype(str) == str(grid_id)]
    if hit.empty:
        return {}
    r = hit.iloc[0]
    out = {}
    if pd.notna(r.get("연면적")) and float(r["연면적"]) > 0:
        out["연면적"] = f"{float(r['연면적']):,.0f}"
    if pd.notna(r.get("건축면적")) and float(r["건축면적"]) > 0:
        out["건축면적"] = f"{float(r['건축면적']):,.0f}"
    if pd.notna(r.get("사용승인연도")):
        out["건축연도"] = f"{int(r['사용승인연도'])}"
    out["_n"] = int(r.get("건물수", 0) or 0)
    return out
