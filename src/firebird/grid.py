"""500m 격자: 좌표 -> 격자 id, 격자 중심/경계, 이웃 격자.

경위도(EPSG:4326)를 UTM-K(EPSG:5179)로 옮긴 뒤 미터 단위로 자른다.
경위도에서 바로 자르면 위도에 따라 셀의 실제 크기가 달라져,
'500m 격자'라는 말이 지역마다 다른 뜻이 된다.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pyproj import Transformer


@lru_cache(maxsize=8)
def _to_metric(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def lonlat_to_xy(lon, lat, crs_geographic: str, crs_metric: str):
    tf = _to_metric(crs_geographic, crs_metric)
    return tf.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def xy_to_lonlat(x, y, crs_metric: str, crs_geographic: str):
    tf = _to_metric(crs_metric, crs_geographic)
    return tf.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def cell_index(x, y, size_m: int):
    """미터 좌표 -> 정수 격자 인덱스 (gx, gy). 결측은 pandas Int64 의 <NA>."""
    gx = pd.array(np.floor(np.asarray(x, dtype=float) / size_m), dtype="Int64")
    gy = pd.array(np.floor(np.asarray(y, dtype=float) / size_m), dtype="Int64")
    return gx, gy


def make_grid_id(gx, gy) -> pd.Series:
    """격자 id 문자열. NaN 좌표는 <NA> 로 남겨 뒤에서 걸러낸다."""
    gx = pd.Series(gx).astype("Int64")
    gy = pd.Series(gy).astype("Int64")
    out = gx.astype("string") + "_" + gy.astype("string")
    out[gx.isna() | gy.isna()] = pd.NA
    return out


def assign_grid(df: pd.DataFrame, cfg, lon_col: str = "lon", lat_col: str = "lat") -> pd.DataFrame:
    """좌표 컬럼이 있는 DataFrame 에 gx/gy/grid_id 를 붙인다(원본 불변)."""
    out = df.copy()
    ok = out[lon_col].notna() & out[lat_col].notna()
    out["gx"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["gy"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if ok.any():
        x, y = lonlat_to_xy(out.loc[ok, lon_col].to_numpy(),
                            out.loc[ok, lat_col].to_numpy(),
                            cfg.crs_geographic, cfg.crs_metric)
        gx, gy = cell_index(x, y, cfg.grid_size_m)
        out.loc[ok, "gx"] = gx
        out.loc[ok, "gy"] = gy
    out["grid_id"] = make_grid_id(out["gx"], out["gy"])
    return out


def grid_centers(grid_ids: pd.Series, cfg) -> pd.DataFrame:
    """격자 id -> 중심 경위도. 지도 시각화와 최근접 계산에 쓴다."""
    ids = pd.Series(grid_ids).dropna().astype(str).unique()
    if len(ids) == 0:
        return pd.DataFrame(columns=["grid_id", "gx", "gy", "cx", "cy", "lon", "lat"])
    gx = np.array([int(i.split("_")[0]) for i in ids])
    gy = np.array([int(i.split("_")[1]) for i in ids])
    size = cfg.grid_size_m
    cx = (gx + 0.5) * size
    cy = (gy + 0.5) * size
    lon, lat = xy_to_lonlat(cx, cy, cfg.crs_metric, cfg.crs_geographic)
    return pd.DataFrame({"grid_id": ids, "gx": gx, "gy": gy,
                         "cx": cx, "cy": cy, "lon": lon, "lat": lat})


def neighbor_ids(grid_id: str, ring: int = 1) -> list[str]:
    """자기 자신을 뺀 주변 격자 id 목록 (ring=1 이면 8-이웃)."""
    gx, gy = (int(v) for v in str(grid_id).split("_"))
    out = []
    for dx in range(-ring, ring + 1):
        for dy in range(-ring, ring + 1):
            if dx == 0 and dy == 0:
                continue
            out.append(f"{gx + dx}_{gy + dy}")
    return out


def cell_bounds(grid_id: str, cfg) -> list[tuple[float, float]]:
    """격자 폴리곤 꼭짓점(경위도). pydeck PolygonLayer 용."""
    gx, gy = (int(v) for v in str(grid_id).split("_"))
    s = cfg.grid_size_m
    xs = [gx * s, (gx + 1) * s, (gx + 1) * s, gx * s]
    ys = [gy * s, gy * s, (gy + 1) * s, (gy + 1) * s]
    lon, lat = xy_to_lonlat(np.array(xs), np.array(ys), cfg.crs_metric, cfg.crs_geographic)
    return list(zip(lon.tolist(), lat.tolist()))
