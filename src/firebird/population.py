"""통계청 SGIS 인구 연계 — 읍면동 상주인구·가구·인구밀도.

**서식의 '상주인구' 칸에는 넣지 않는다.** 그 칸이 묻는 것은 그 지구(우리에겐
500m 격자)의 상주인구인데, SGIS OpenAPI 는 읍면동까지만 준다. 읍면동 인구를
그 동의 모든 격자에 쓰면 한 동의 인구가 여러 번 세어지고, 면적으로 나누면
그건 측정이 아니라 추정이다. 결재 문서 칸에 추정값을 넣지 않는다는 원칙은
여기서도 같다.

대신 **정책 근거**로 쓴다. '고위험인데 소화전이 없는 90개 구역'이 어느 동에
걸쳐 있고 그 동에 몇 명이 사는가는 예산을 요구할 때 실제로 필요한 문장이고,
읍면동 단위로 말해도 참이다.

격자 단위 인구는 SGIS 가 파일데이터(공공데이터포털 '국가데이터처_SGIS 격자
통계 및 경계')로만 배포한다. 그 파일을 받으면 격자에 직접 붙일 수 있다.

**주의: SGIS 행정구역코드는 법정동코드가 아니다.** 법정동코드로 31140 은
울산 남구지만 SGIS 에서 31140 은 오산시다(SGIS 울산은 26). 그래서 코드를
그대로 넘기지 않고 이름으로 찾아 내려간다.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import geocode as GC

log = logging.getLogger(__name__)

BASE = "https://sgisapi.kostat.go.kr/OpenAPI3"
AUTH_URL = f"{BASE}/auth/authentication.json"
STAGE_URL = f"{BASE}/addr/stage.json"
POP_URL = f"{BASE}/stats/population.json"

KEY_ENV, SECRET_ENV = "SGIS_CONSUMER_KEY", "SGIS_CONSUMER_SECRET"
CACHE_NAME = "sgis_population.parquet"

#: 토큰은 발급 후 얼마간만 유효하다. 매 호출마다 새로 받지 않도록 잠깐 들고 있는다.
_TOKEN: tuple[str, float] | None = None
TOKEN_TTL = 1800.0


def credentials() -> tuple[str | None, str | None]:
    return GC.load_api_key(KEY_ENV), GC.load_api_key(SECRET_ENV)


def access_token(*, timeout: int = 30) -> str | None:
    global _TOKEN
    if _TOKEN and time.time() - _TOKEN[1] < TOKEN_TTL:
        return _TOKEN[0]
    key, secret = credentials()
    if not (key and secret):
        return None
    try:
        r = requests.get(AUTH_URL, timeout=timeout,
                         params={"consumer_key": key, "consumer_secret": secret})
        r.raise_for_status()
        tok = (r.json().get("result") or {}).get("accessToken")
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("SGIS 인증 실패: %s", type(exc).__name__)
        return None
    if tok:
        _TOKEN = (tok, time.time())
    return tok


def _stage(token: str, cd: str = "", *, timeout: int = 30) -> list[dict]:
    try:
        r = requests.get(STAGE_URL, timeout=timeout,
                         params={"accessToken": token, **({"cd": cd} if cd else {})})
        r.raise_for_status()
        return r.json().get("result") or []
    except (requests.RequestException, ValueError) as exc:
        log.warning("SGIS 행정구역 조회 실패: %s", type(exc).__name__)
        return []


def find_code(token: str, *names: str) -> str | None:
    """이름으로 SGIS 행정구역코드를 찾는다 (예: '울산광역시', '남구').

    코드 체계가 법정동코드와 달라 이름으로 내려가는 편이 안전하다.
    이름이 조금 달라도(광역시/시) 앞부분이 맞으면 받아들인다.
    """
    cd = ""
    for want in names:
        w = str(want).replace(" ", "")
        hit = None
        for x in _stage(token, cd):
            nm = str(x.get("addr_name", "")).replace(" ", "")
            if nm == w or nm.startswith(w) or w.startswith(nm):
                hit = str(x.get("cd"))
                break
        if hit is None:
            return None
        cd = hit
    return cd


def emd_population(sido: str, sgg: str, *, year: str = "2023",
                   timeout: int = 40) -> pd.DataFrame:
    """한 시군구의 읍면동별 인구·가구·주택·인구밀도."""
    tok = access_token()
    if not tok:
        return pd.DataFrame()
    cd = find_code(tok, sido, sgg) if sgg else find_code(tok, sido)
    if not cd:
        log.warning("SGIS 코드 못 찾음: %s %s", sido, sgg)
        return pd.DataFrame()
    try:
        r = requests.get(POP_URL, timeout=timeout, params={
            "accessToken": tok, "year": year, "adm_cd": cd, "low_search": "1"})
        r.raise_for_status()
        res = r.json().get("result") or []
    except (requests.RequestException, ValueError) as exc:
        log.warning("SGIS 인구 조회 실패: %s", type(exc).__name__)
        return pd.DataFrame()
    rows = []
    for x in res:
        rows.append({
            "sgg": sgg, "emd": str(x.get("adm_nm", "")).strip(),
            "상주인구": pd.to_numeric(x.get("tot_ppltn"), errors="coerce"),
            "가구": pd.to_numeric(x.get("tot_family"), errors="coerce"),
            "주택": pd.to_numeric(x.get("tot_house"), errors="coerce"),
            "인구밀도": pd.to_numeric(x.get("ppltn_dnsty"), errors="coerce"),
            "평균나이": pd.to_numeric(x.get("avg_age"), errors="coerce"),
            "기준연도": year,
        })
    return pd.DataFrame(rows)


def collect(cfg, panel_year: pd.DataFrame, *, city_label: str = "",
            year: str = "2023", refresh: bool = False) -> pd.DataFrame:
    """패널에 나오는 시군구를 모두 훑어 읍면동 인구표를 만든다.

    반환 컬럼: sgg, emd, 상주인구, 가구, 주택, 인구밀도, 평균나이, 기준연도.
    키가 없으면 빈 표. 이 값은 **읍면동 단위**이며 격자 값이 아니다.
    """
    cache = cfg.paths.cache / CACHE_NAME
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    if not all(credentials()):
        log.info("SGIS 연계 건너뜀 (키 없음)")
        return pd.DataFrame()

    sggs = sorted({str(x).strip() for x in panel_year.get("sgg", pd.Series(dtype=str))
                   if str(x).strip()})
    frames = []
    for sgg in sggs:
        got = emd_population(city_label, sgg, year=year)
        if not got.empty:
            frames.append(got)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["sgg", "emd"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    log.info("SGIS 읍면동 인구 %d건 (%s년)", len(out), year)
    return out


def attach(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """격자 표에 그 격자가 속한 읍면동의 인구를 붙인다.

    격자 값이 아니라 **그 격자가 속한 동 전체의 값**이다. 화면과 표에서
    그렇게 이름 붙인다 — '읍면동 인구'.
    """
    if df.empty or pop is None or pop.empty:
        return df
    keep = ["sgg", "emd", "상주인구", "인구밀도"]
    p = pop[[c for c in keep if c in pop.columns]].rename(
        columns={"상주인구": "읍면동 인구", "인구밀도": "읍면동 인구밀도"})
    on = [c for c in ("sgg", "emd") if c in df.columns and c in p.columns]
    return df.merge(p, on=on, how="left") if on else df
