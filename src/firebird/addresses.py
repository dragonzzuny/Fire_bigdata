"""주소 정규화: 원본 주소 문자열 -> (시도, 시군구, 도로명) 과 지오코딩 키.

기획서의 분석 단위는 (시군구·도로명)이다. 공개 데이터에 건물번호·좌표가
없기 때문에, 주소에서 안정적으로 뽑아낼 수 있는 가장 낮은 해상도가 도로다.
"""
from __future__ import annotations

import re

import pandas as pd

# 시도: 긴 접미사부터 봐야 '세종특별자치시'가 '시'로 잘리지 않는다.
_SIDO_RE = re.compile(
    r"(?P<sido>[가-힣]+(?:특별자치시|특별자치도|광역시|특별시|자치시|자치도|[가-힣]?도|시))\s"
)
_SGG_RE = re.compile(r"(?P<sgg>[가-힣]{1,6}(?:구|군|시))(?=\s|$)")
# 도로명: '…대로/…로/…길', 뒤에 'NN번길'이 붙을 수 있다.
_ROAD_RE = re.compile(r"(?P<road>[가-힣A-Za-z0-9]+(?:대로|로|길)(?:\s?\d+번길)?)")
_PAREN_RE = re.compile(r"[（(\[][^）)\]]*[）)\]]")
_WS_RE = re.compile(r"\s+")


def clean(addr: object) -> str:
    """괄호 주석·중복 공백 제거. NaN 은 빈 문자열."""
    if addr is None or (isinstance(addr, float) and pd.isna(addr)):
        return ""
    s = str(addr).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return ""
    s = _PAREN_RE.sub(" ", s)
    s = s.replace(",", " ")
    return _WS_RE.sub(" ", s).strip()


def parse(addr: object) -> dict[str, str]:
    """주소 한 건을 파싱한다. 실패한 필드는 빈 문자열로 남는다."""
    s = clean(addr)
    if not s:
        return {"sido": "", "sgg": "", "road": "", "clean": ""}

    sido = ""
    m = _SIDO_RE.match(s + " ")
    if m:
        sido = m.group("sido")
        rest = s[m.end("sido"):].strip()
    else:
        rest = s

    sgg = ""
    m = _SGG_RE.search(rest)
    if m:
        sgg = m.group("sgg")
        rest_after_sgg = rest[m.end("sgg"):].strip()
    else:
        rest_after_sgg = rest

    road = ""
    m = _ROAD_RE.search(rest_after_sgg) or _ROAD_RE.search(rest)
    if m:
        road = _WS_RE.sub("", m.group("road"))

    return {"sido": sido, "sgg": sgg, "road": road, "clean": s}


def geocode_key(parsed: dict[str, str], default_sido: str = "") -> str:
    """카카오 로컬 API 에 보낼 질의 문자열. 도로가 없으면 빈 문자열."""
    if not parsed.get("road"):
        return ""
    sido = parsed.get("sido") or default_sido
    return _WS_RE.sub(" ", f"{sido} {parsed.get('sgg','')} {parsed['road']}").strip()


def add_address_columns(df: pd.DataFrame,
                        addr_col: str = "address",
                        default_sido: str = "") -> pd.DataFrame:
    """DataFrame 에 sido/sgg/road/geo_key 컬럼을 붙인다(원본 불변)."""
    parsed = df[addr_col].map(parse)
    out = df.copy()
    out["sido"] = parsed.map(lambda d: d["sido"]) 
    out["sgg"] = parsed.map(lambda d: d["sgg"])
    out["road"] = parsed.map(lambda d: d["road"])
    out["geo_key"] = parsed.map(lambda d: geocode_key(d, default_sido))
    # 데이터셋에 별도 시군구 컬럼이 있으면 파싱 실패분을 그걸로 메운다.
    if "sgg" in df.columns:
        fallback = df["sgg"].astype(str).str.strip()
        out["sgg"] = out["sgg"].where(out["sgg"] != "", fallback)
    return out
