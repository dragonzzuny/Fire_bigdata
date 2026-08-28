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
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests

from . import geocode as GC
from . import grid as G

log = logging.getLogger(__name__)

#: 표제부 조회. 건축HUB 건축물대장정보 서비스.
TITLE_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
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
    """공공데이터포털 서비스 키. 없으면 None.

    포털은 같은 키를 Encoding/Decoding 두 가지로 준다. Encoding 키(%2F, %3D 가
    들어간 것)를 그대로 requests 의 params 에 넘기면 % 가 다시 %25 로 인코딩돼
    인증이 깨진다. 어느 쪽을 넣었든 여기서 풀어 두고, 인코딩은 requests 에 맡긴다.
    """
    key = GC.load_api_key(KEY_ENV)
    if not key:
        return None
    return unquote(key) if "%" in key else key


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


#: 한 번에 받을 건수. 1000 을 넣으면 서버가 빈 본문을 200 으로 돌려준다
#: (오류가 아니라 그냥 아무것도 안 온다). 100 은 안정적으로 응답한다.
ROWS_PER_PAGE = 100


def _get_page(session: requests.Session, params: dict, *, timeout: int,
              tries: int = 6) -> str:
    """한 쪽을 받는다. 실패하면 점점 더 오래 쉬고 다시.

    이 서버는 두 가지로 실패한다. 연결을 그냥 끊거나(RemoteDisconnected),
    HTTP 200 에 **빈 본문**을 준다. 둘 다 오류 코드가 아니라서 그대로 두면
    '자료가 없는 동'으로 조용히 넘어간다 — 서식에 연면적이 비는 이유가
    자료가 없어서인지 서버가 끊어서인지 구분되지 않는다. 그래서 다시 부른다.
    같은 요청을 연달아 두 번 보내면 한 번은 3,784건, 한 번은 0건이 오는 것을
    실제로 확인했다.
    """
    for attempt in range(tries):
        try:
            r = session.get(TITLE_URL, params=params, timeout=timeout)
            r.raise_for_status()
            if r.text.strip():
                return r.text
        except requests.RequestException as exc:
            if attempt == tries - 1:
                log.warning("건축물대장 호출 실패(%s쪽): %s",
                            params.get("pageNo"), type(exc).__name__)
        time.sleep(0.8 * (attempt + 1))
    return ""


def fetch_title(sigungu_cd: str, bjdong_cd: str, key: str, *, cfg=None,
                rows_per_page: int = ROWS_PER_PAGE, timeout: int = 60,
                pause: float = 0.35, rounds: int = 4,
                refresh: bool = False) -> pd.DataFrame:
    """한 법정동의 표제부를 모두 받는다. 받은 것은 디스크에 남긴다.

    쪽 하나가 실패했다고 거기서 멈추면 안 된다. 쪽 번호는 서로 독립이라
    3쪽이 실패해도 4쪽은 받을 수 있고, 실패한 쪽만 나중에 다시 부르면 된다.
    처음엔 실패 즉시 멈추게 했더니 달동이 3,784건 중 1,100건에서 끊겼다.

    다 받은 경우에만 캐시한다. 반쯤 받은 것을 캐시하면 그게 정답이 돼 버린다.
    """
    cache = None
    if cfg is not None:
        cache = cfg.paths.cache / CACHE_NAME / f"{sigungu_cd}_{bjdong_cd}.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

    session = requests.Session()
    base = {"serviceKey": key, "sigunguCd": sigungu_cd, "bjdongCd": bjdong_cd,
            "numOfRows": rows_per_page, "_type": "xml"}

    first = _get_page(session, {**base, "pageNo": 1}, timeout=timeout)
    rows, total = _rows(first)
    if not rows:
        return pd.DataFrame()
    pages: dict[int, list[dict]] = {1: rows}
    n_pages = max(1, -(-total // rows_per_page))

    pending = [p for p in range(2, n_pages + 1)]
    for rnd in range(rounds):
        if not pending:
            break
        still = []
        for pno in pending:
            got, _ = _rows(_get_page(session, {**base, "pageNo": pno},
                                     timeout=timeout))
            if got:
                pages[pno] = got
            else:
                still.append(pno)
            time.sleep(pause)
        if len(still) == len(pending):        # 한 쪽도 못 받았다면 더 해도 같다
            pending = still
            break
        pending = still
        if pending:
            log.info("%s-%s %d쪽 재시도 (%d회차)", sigungu_cd, bjdong_cd,
                     len(pending), rnd + 2)
            time.sleep(1.5 * (rnd + 1))

    got_rows = [r for p in sorted(pages) for r in pages[p]]
    df = pd.DataFrame(got_rows)
    complete = not pending and len(got_rows) >= total
    if not complete:
        log.warning("%s-%s 표제부 %d/%d 건 (%d쪽 실패)", sigungu_cd, bjdong_cd,
                    len(got_rows), total, len(pending))
    if df.empty:
        return df
    for c in ("연면적", "건축면적", "지상층수"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["사용승인연도"] = pd.to_numeric(
        df["사용승인일"].str.slice(0, 4), errors="coerce")
    if cache is not None and complete:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
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

    ids = list(dict.fromkeys(str(g) for g in grid_ids))
    centers = G.grid_centers(pd.Series(ids), cfg)

    # 한 격자가 법정동 하나에 딱 들어가는 경우는 드물다. 중심점만 보면
    # 격자에 걸친 다른 동의 건물이 통째로 빠져 연면적이 과소 집계된다.
    # 네 모서리와 중심을 모두 물어 걸치는 동을 전부 찾는다.
    probes: list[tuple[float, float]] = []
    for gid in ids:
        try:
            corners = G.cell_bounds(gid, cfg)
        except Exception:                                    # noqa: BLE001
            corners = []
        probes += [(float(x), float(y)) for x, y in corners]
    probes += [(float(r.lon), float(r.lat)) for r in centers.itertuples()]

    seen: set[tuple[str, str]] = set()
    frames = []
    for lon, lat in probes:
        rc = region_code(lon, lat, kakao)
        if rc is None or rc in seen:
            continue
        seen.add(rc)
        got = fetch_title(rc[0], rc[1], key, cfg=cfg)
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

# ---------------------------------------------------------------- 연도별 노후도

#: 사용승인연도로 인정할 범위. 원본에 978년, 2026년 같은 값이 섞여 있다.
YEAR_MIN, YEAR_MAX = 1900, 2030

#: '노후'로 보는 경과연수. 소방에서 30년은 정밀안전진단·리모델링을 말할 때
#: 쓰는 경계선이라 현장에서 설명이 통한다.
OLD_YEARS = 30
#: '신축'으로 보는 경과연수. 준공 직후는 공사 잔재·전기 설비 초기 불량이 있다.
NEW_YEARS = 5


def load_cached(cfg) -> pd.DataFrame:
    """받아 둔 법정동 표제부를 모두 읽어 한 표로."""
    d = cfg.paths.cache / CACHE_NAME
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def emd_year_features(cfg, years, *, buildings: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """읍면동 × 연도 노후도 피처.

    **이 피처가 있는 이유**: 대상물·업소·소화전은 '언제부터 존재했는가'가 없어
    2014년 행에도 현재 값이 들어간다(스냅샷 누수). 건축물대장에는 사용승인일이
    있으므로 't년 이전에 준공된 건물'만 세면 그 해에 실제로 서 있던 것이 되고,
    스냅샷이 아니라 시계열이 된다.

    **격자가 아니라 읍면동인 이유**: 격자에 붙이려면 건물마다 주소를 좌표로
    바꿔야 하는데 울산 전체가 25만 동 규모다. 무료 지오코딩 한도를 넘는다.
    읍면동 단위는 대장이 이미 법정동별로 오므로 지오코딩이 필요 없고,
    '연도에 따라 변한다'는 핵심 성질은 그대로 남는다.

    **남는 한계**: 대장은 현재 시점 자료라 t년 이전에 헐린 건물은 들어 있지
    않다. 미래를 당겨쓰는 누수는 아니지만 생존 편향이다. 그리고 사용승인연도가
    비어 있는 건물(약 10%)은 어느 해에도 세지 않는다.
    """
    b = buildings if buildings is not None else load_cached(cfg)
    if b.empty or "지번주소" not in b.columns:
        return pd.DataFrame()

    b = b.dropna(subset=["사용승인연도"]).copy()
    b["사용승인연도"] = pd.to_numeric(b["사용승인연도"], errors="coerce")
    b = b[b["사용승인연도"].between(YEAR_MIN, YEAR_MAX)]
    if b.empty:
        return pd.DataFrame()

    # 지번주소에서 시군구·읍면동을 뽑는다. 대장은 '울산광역시 남구 달동 100' 꼴이다.
    parts = b["지번주소"].astype(str).str.split()
    b["sgg"] = parts.str[1]
    b["emd"] = parts.str[2]
    b = b[b["emd"].astype(str).str.len() > 1]
    b["연면적"] = pd.to_numeric(b["연면적"], errors="coerce").fillna(0.0)

    rows = []
    for y in sorted({int(x) for x in years}):
        cur = b[b["사용승인연도"] <= y]
        if cur.empty:
            continue
        age = y - cur["사용승인연도"]
        g = cur.assign(_age=age,
                       _old=(age >= OLD_YEARS).astype(float),
                       _new=(age <= NEW_YEARS).astype(float))
        agg = (g.groupby(["sgg", "emd"])
                 .agg(**{"bld_n": ("_age", "size"),
                         "bld_area": ("연면적", "sum"),
                         "bld_age_mean": ("_age", "mean"),
                         "bld_old_share": ("_old", "mean"),
                         "bld_new_share": ("_new", "mean")})
                 .reset_index())
        agg["year"] = y
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["bld_area_per_n"] = out["bld_area"] / out["bld_n"].replace(0, np.nan)
    return out


def attach_features(panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """패널에 읍면동×연도 노후도를 붙인다. 없으면 그대로 돌려준다."""
    if panel.empty or feats is None or feats.empty:
        return panel
    on = [c for c in ("sgg", "emd", "year") if c in panel.columns and c in feats.columns]
    if len(on) < 3:
        return panel
    out = panel.merge(feats, on=on, how="left")
    cols = [c for c in feats.columns if c.startswith("bld_")]
    # 대장을 아직 안 받은 동은 결측이다. 0 으로 채우면 '건물이 없는 동'이 되므로
    # 그대로 두고 LightGBM 이 결측으로 다루게 한다.
    got = out[cols].notna().any(axis=1).mean() if cols else 0.0
    log.info("노후도 피처 %d개 · 격자행 %.1f%% 에 붙었다", len(cols), got * 100)
    return out

