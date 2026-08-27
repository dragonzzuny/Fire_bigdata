"""주소 조각 -> 지오코딩 키 계층.

플랫폼 CSV 는 주소를 한 문자열로 주지 않는다. 시도/시군구/읍면동/리/도로명이
따로 온다. 그리고 **도로명은 자주 비어 있다** — 울산 화재발생현황 2005~2020 은
35%만 채워져 있다. 읍면동은 100% 채워져 있다.

그래서 키를 하나만 만들지 않고 정밀도 순으로 여러 개 만든다:

    road   시도 시군구 도로명      가장 정밀. 500m 격자와 해상도가 맞는다.
    emd    시도 시군구 읍면동      항상 만들 수 있다. 동 중심점이라 거칠다.
    sido   시도 읍면동             시군구가 없는 세종 같은 단층제 대응.

행마다 위에서부터 성공한 첫 키의 좌표를 쓰고, **어느 단계에서 성공했는지 기록한다**.
'좌표 확보율 95%'만 말하고 그 중 몇 %가 동 중심점인지 말하지 않으면,
격자 해상도를 실제보다 좋게 주장하는 셈이 된다.
"""
from __future__ import annotations

import re

import pandas as pd

_WS = re.compile(r"\s+")
_PAREN = re.compile(r"[（(\[][^）)\]]*[）)\]]")

#: 원본이 쓰는 시도 약칭 -> 지오코딩에 넣을 정식 명칭
SIDO_CANON = {
    "울산": "울산광역시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "서울": "서울특별시", "세종": "세종특별자치시",
    "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도",
    "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도",
    "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도",
}

#: 키 정밀도 순서. 앞의 것이 성공하면 뒤는 보지 않는다.
KEY_LEVELS = ("road", "emd", "sido_emd")


def clean(value: object) -> str:
    """괄호 주석·중복 공백 제거. 결측·자리표시자는 빈 문자열.

    pandas 의 결측은 float('nan') 만이 아니다. pd.NA / pd.NaT 도 온다.
    isinstance(float) 로만 거르면 pd.NA 가 문자열 '<NA>' 로 살아남아,
    '도로명이 100% 채워져 있다' 같은 조용한 거짓말이 된다.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass  # 배열류는 isna 가 스칼라를 안 돌려준다 — 아래 문자열 처리로 넘어간다.
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>", "nat", "-"}:
        return ""
    s = _PAREN.sub(" ", s)
    return _WS.sub(" ", s).strip()


def canon_sido(value: object, default: str = "") -> str:
    """'울산' -> '울산광역시'. 이미 정식 명칭이면 그대로 둔다."""
    s = clean(value)
    if not s:
        return default
    if s in SIDO_CANON:
        return SIDO_CANON[s]
    for short, full in SIDO_CANON.items():
        if s.startswith(short) and (s.endswith(("시", "도")) or s == short):
            return full
    return s


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    """없는 컬럼은 빈 문자열 시리즈로. 데이터셋마다 있는 조각이 다르다."""
    if col in df.columns:
        return df[col].map(clean)
    return pd.Series("", index=df.index, dtype=object)


def build_keys(df: pd.DataFrame, default_sido: str = "") -> pd.DataFrame:
    """주소 조각 -> 정규화된 조각 + 정밀도별 지오코딩 키.

    원본을 바꾸지 않고 새 컬럼만 붙인 사본을 돌려준다.
    """
    out = df.copy()
    sido = _series(df, "sido").map(lambda v: canon_sido(v, default_sido))
    sido = sido.where(sido != "", default_sido)
    sgg = _series(df, "sgg")
    emd = _series(df, "emd")
    ri = _series(df, "ri")
    road = _series(df, "road")

    out["sido"] = sido
    out["sgg"] = sgg
    out["emd"] = emd
    out["ri"] = ri
    out["road"] = road

    def join(*parts: pd.Series) -> pd.Series:
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.str.cat(p, sep=" ")
        return joined.map(lambda s: _WS.sub(" ", s).strip())

    # road: 도로명이 있을 때만. 없으면 빈 문자열이라 다음 단계로 넘어간다.
    key_road = join(sido, sgg, road)
    out["key_road"] = key_road.where(road != "", "")
    # emd: 읍면동. 리가 있으면 붙여 조금 더 좁힌다.
    key_emd = join(sido, sgg, emd, ri)
    out["key_emd"] = key_emd.where(emd != "", "")
    # sido_emd: 시군구가 비어 있는 단층제(세종) 대응.
    key_sido = join(sido, emd)
    out["key_sido_emd"] = key_sido.where(emd != "", "")
    return out


def all_keys(df: pd.DataFrame) -> list[str]:
    """이 테이블이 지오코딩해야 할 고유 키 전체 (빈 값 제외)."""
    keys: list[str] = []
    for level in KEY_LEVELS:
        col = f"key_{level}"
        if col in df.columns:
            keys += df.loc[df[col] != "", col].tolist()
    return list(dict.fromkeys(keys))


def resolve(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """정밀도 순으로 좌표를 채우고, 쓰인 단계를 geo_level 에 남긴다.

    coords: DataFrame[geo_key, lon, lat, matched]
    """
    lookup = (coords[coords["matched"]].drop_duplicates("geo_key")
                    .set_index("geo_key")[["lon", "lat"]])
    out = df.copy()
    out["lon"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["lat"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["geo_level"] = ""

    for level in KEY_LEVELS:
        col = f"key_{level}"
        if col not in out.columns:
            continue
        todo = out["lon"].isna() & (out[col] != "")
        if not todo.any():
            continue
        got = out.loc[todo, col].map(lookup["lon"])
        lat = out.loc[todo, col].map(lookup["lat"])
        hit = got.notna()
        idx = out.index[todo][hit.to_numpy()]
        out.loc[idx, "lon"] = got[hit].astype("Float64").to_numpy()
        out.loc[idx, "lat"] = lat[hit].astype("Float64").to_numpy()
        out.loc[idx, "geo_level"] = level
    return out


def level_breakdown(df: pd.DataFrame) -> dict[str, float]:
    """어느 정밀도로 좌표를 얻었는지의 비율. manifest 에 그대로 들어간다."""
    n = len(df)
    if n == 0 or "geo_level" not in df.columns:
        return {}
    counts = df["geo_level"].replace("", "none").value_counts()
    return {str(k): int(v) for k, v in counts.items()}
