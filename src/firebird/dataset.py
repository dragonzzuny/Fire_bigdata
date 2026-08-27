"""원본 CSV -> 격자x연도 의사결정 테이블.

한 도시분 전체를 이 한 모듈이 책임진다:
    적재 -> 주소 정규화 -> 지오코딩 -> 격자 배정 -> 라벨/피처 -> 패널
좌표 확보율·결측 같은 '깎여나간 양'은 전부 manifest 에 기록한다.
숫자를 못 만든 것과 만들었는데 나쁜 것은 다른 문제이기 때문이다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import addresses, features as F, geocode, grid
from .io_utils import load_dataset

log = logging.getLogger(__name__)


@dataclass
class CityData:
    city: str
    panel: pd.DataFrame
    fires: pd.DataFrame
    targets: pd.DataFrame
    businesses: pd.DataFrame
    hydrants: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 연도 추출

def extract_year(series: pd.Series) -> pd.Series:
    """다양한 일시 표기에서 연도를 뽑는다. 실패하면 <NA>."""
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    year = parsed.dt.year
    # 파싱 실패분은 문자열 앞 4자리 숫자로 한 번 더 시도한다.
    fallback = pd.to_numeric(s.str.extract(r"(\d{4})")[0], errors="coerce")
    return year.fillna(fallback).astype("Int64")


def extract_hour_weekday(series: pd.Series) -> pd.DataFrame:
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce", format="mixed")
    return pd.DataFrame({"hour": parsed.dt.hour.astype("Int64"),
                         "weekday": parsed.dt.dayofweek.astype("Int64")})


# ---------------------------------------------------------------- 범주 정규화

def normalize_grade(series: pd.Series, grade_map: dict[str, str]) -> pd.Series:
    s = series.fillna("").astype(str).str.replace(r"\s", "", regex=True)
    out = pd.Series("기타", index=s.index, dtype=object)
    for raw, std in grade_map.items():
        out[s.str.contains(str(raw), na=False)] = std
    return out


def normalize_biz_type(series: pd.Series, keywords: dict[str, list[str]]) -> pd.Series:
    s = series.fillna("").astype(str).str.replace(r"\s", "", regex=True)
    out = pd.Series("기타", index=s.index, dtype=object)
    for std, kws in keywords.items():
        for kw in kws or []:
            out[s.str.contains(str(kw), na=False, case=False)] = std
    return out


# ---------------------------------------------------------------- 도시 적재

def _prepare(df: pd.DataFrame, cfg, city: str, kind: str) -> pd.DataFrame:
    """주소 -> geo_key 까지. 좌표 부여는 뒤에서 한꺼번에 한다."""
    default_sido = cfg.city(city)["label"]
    if "address" not in df.columns:
        log.warning("[%s/%s] 주소 컬럼이 없다 — 격자 배정 불가", city, kind)
        df = df.copy()
        df["address"] = ""
    return addresses.add_address_columns(df, "address", default_sido)


def load_city(cfg, city: str, *, use_api: bool = True) -> CityData:
    """한 도시의 원본을 전부 읽어 격자 패널까지 만든다."""
    manifest: dict[str, Any] = {"city": city, "grid_size_m": cfg.grid_size_m}

    fires = _prepare(load_dataset(cfg, city, "fire"), cfg, city, "fire")
    targets = _prepare(load_dataset(cfg, city, "target"), cfg, city, "target")
    businesses = _prepare(load_dataset(cfg, city, "business"), cfg, city, "business")
    try:
        hydrants = _prepare(load_dataset(cfg, city, "hydrant"), cfg, city, "hydrant")
    except FileNotFoundError:
        log.warning("[%s] 소방용수시설 데이터 없음 — 소화전 분석 비활성", city)
        hydrants = pd.DataFrame(columns=["address", "geo_key", "sgg", "road"])

    # --- 지오코딩: 모든 테이블의 키를 모아 한 번에 ---
    all_keys = pd.concat([t["geo_key"] for t in (fires, targets, businesses, hydrants)
                          if "geo_key" in t.columns], ignore_index=True)
    coords = geocode.geocode_keys(all_keys.tolist(), cfg, allow_network=use_api)
    manifest["geocode"] = {
        "unique_keys": int(coords.shape[0]),
        "matched_keys": int(coords["matched"].sum()),
        "key_match_rate": float(coords["matched"].mean()) if len(coords) else 0.0,
    }

    def to_grid(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if df.empty:
            return df
        # 소방용수시설처럼 원본에 좌표가 있으면 그것을 우선한다.
        if {"lat", "lon"}.issubset(df.columns) and df["lat"].notna().any():
            out = df.copy()
            out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
            out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
            need = out["lon"].isna() | out["lat"].isna()
            if need.any():
                filled = geocode.attach_coordinates(
                    out.loc[need].drop(columns=["lon", "lat"]), coords)
                out.loc[need, "lon"] = filled["lon"].to_numpy()
                out.loc[need, "lat"] = filled["lat"].to_numpy()
        else:
            out = geocode.attach_coordinates(df, coords)
        out = grid.assign_grid(out, cfg)
        cov = geocode.coverage(out)
        manifest.setdefault("coverage", {})[name] = cov
        log.info("[%s/%s] 좌표 확보 %d/%d (%.1f%%)", city, name,
                 cov["with_coords"], cov["rows"], cov["rate"] * 100)
        return out

    fires = to_grid(fires, "fire")
    targets = to_grid(targets, "target")
    businesses = to_grid(businesses, "business")
    hydrants = to_grid(hydrants, "hydrant")

    # --- 라벨: 연도 ---
    fires["year"] = extract_year(fires["occurred_at"])
    fires = fires.join(extract_hour_weekday(fires["occurred_at"]))
    yr = fires["year"]
    in_range = yr.between(cfg.year_min, cfg.year_max)
    manifest["fires"] = {
        "rows": int(len(fires)),
        "with_year": int(yr.notna().sum()),
        "in_year_range": int(in_range.sum()),
        "with_grid": int(fires["grid_id"].notna().sum()),
        "usable": int((in_range & fires["grid_id"].notna()).sum()),
        "year_range": [cfg.year_min, cfg.year_max],
    }
    fires_used = fires[in_range & fires["grid_id"].notna()].copy()

    panel = build_panel(cfg, fires_used, targets, businesses, hydrants)
    manifest["panel"] = {"rows": int(len(panel)),
                         "grids": int(panel["grid_id"].nunique()),
                         "years": sorted(int(y) for y in panel["year"].unique()),
                         "total_fires": float(panel["fires"].sum())}
    return CityData(city=city, panel=panel, fires=fires, targets=targets,
                    businesses=businesses, hydrants=hydrants, manifest=manifest)


def build_panel(cfg, fires: pd.DataFrame, targets: pd.DataFrame,
                businesses: pd.DataFrame, hydrants: pd.DataFrame) -> pd.DataFrame:
    """격자x연도 패널 조립."""
    years = list(range(cfg.year_min, cfg.year_max + 1))

    # 격자 모집단: 화재가 난 곳뿐 아니라 대상물·업소·소화전이 있는 곳도 포함.
    universe = pd.concat([t["grid_id"] for t in (fires, targets, businesses, hydrants)
                          if "grid_id" in t.columns], ignore_index=True).dropna()
    if universe.empty:
        raise ValueError("격자를 하나도 만들지 못했다 — 지오코딩이 전부 실패했을 가능성이 크다. "
                         "KAKAO_REST_API_KEY 를 확인하라.")

    counts = F.fire_counts_by_grid_year(fires)
    skeleton = F.build_panel_skeleton(universe, years)
    panel = F.add_fire_history(skeleton, counts, ring=int(cfg["grid"]["neighbor_ring"]))

    tables = []
    if not targets.empty and "grade" in targets.columns:
        t = targets.copy()
        t["grade"] = normalize_grade(t["grade"], cfg.schema.get("grade_map", {}))
        tables.append(F.counts_by_category(t, "grade", "target"))
    elif not targets.empty:
        tables.append(targets.dropna(subset=["grid_id"]).groupby("grid_id")
                      .size().rename("target_total").reset_index())

    if not businesses.empty and "biz_type" in businesses.columns:
        b = businesses.copy()
        kws = cfg.schema.get("biz_type_keywords", {})
        b["biz_type"] = normalize_biz_type(b["biz_type"], kws)
        tables.append(F.counts_by_category(b, "biz_type", "biz", categories=list(kws)))
    elif not businesses.empty:
        tables.append(businesses.dropna(subset=["grid_id"]).groupby("grid_id")
                      .size().rename("biz_total").reset_index())

    if not hydrants.empty and "grid_id" in hydrants.columns:
        tables.append(hydrants.dropna(subset=["grid_id"]).groupby("grid_id")
                      .size().rename("n_hydrant").reset_index())

    panel = F.attach_static(panel, *tables)
    panel = attach_grid_geometry(panel, cfg)
    panel = attach_sgg(panel, cfg, fires, targets, businesses)
    if "n_hydrant" in panel.columns:
        panel["dist_hydrant_m"] = nearest_hydrant_distance(panel, hydrants, cfg)
    F.assert_no_leakage(panel)
    return panel


def attach_grid_geometry(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """격자 중심 경위도만 붙인다.

    gx/gy 는 이미 패널에 있고, 여기서 다시 붙이면 suffix 컬럼(gx_c)이 생겨
    피처 목록에 격자 좌표가 섞여 들어간다. 그러면 모델이 '위험한 동네의
    좌표'를 외우고, 새 도시로 옮기는 순간 무너진다. 그래서 좌표는 표시용으로만 쓴다.
    """
    centers = grid.grid_centers(panel["grid_id"], cfg)[["grid_id", "lon", "lat"]]
    out = panel.drop(columns=[c for c in ("lon", "lat") if c in panel.columns])
    return out.merge(centers, on="grid_id", how="left")


def attach_sgg(panel: pd.DataFrame, cfg, *sources: pd.DataFrame) -> pd.DataFrame:
    """격자별 대표 시군구. LOGO 검증의 그룹 키가 된다."""
    frames = [s[["grid_id", "sgg"]] for s in sources
              if not s.empty and {"grid_id", "sgg"}.issubset(s.columns)]
    if not frames:
        panel = panel.copy()
        panel["sgg"] = ""
        return panel
    allsgg = pd.concat(frames, ignore_index=True).dropna(subset=["grid_id"])
    allsgg = allsgg[allsgg["sgg"].astype(str).str.strip() != ""]
    if allsgg.empty:
        panel = panel.copy()
        panel["sgg"] = ""
        return panel
    mode = (allsgg.groupby(["grid_id", "sgg"]).size().rename("n").reset_index()
                  .sort_values(["grid_id", "n"], ascending=[True, False])
                  .drop_duplicates("grid_id")[["grid_id", "sgg"]])
    out = panel.merge(mode, on="grid_id", how="left")
    out["sgg"] = out["sgg"].fillna("")
    return out


def nearest_hydrant_distance(panel: pd.DataFrame, hydrants: pd.DataFrame, cfg) -> pd.Series:
    """격자 중심에서 가장 가까운 소방용수시설까지의 거리(m).

    소화전이 하나도 없으면 전부 무한대로 둔다 — 0 으로 채우면
    '전부 소화전 옆'이라는 정반대 뜻이 된다.
    """
    n = len(panel)
    if hydrants.empty or "grid_id" not in hydrants.columns:
        return pd.Series(np.inf, index=panel.index)
    h = hydrants.dropna(subset=["lon", "lat"])
    if h.empty:
        return pd.Series(np.inf, index=panel.index)

    hx, hy = grid.lonlat_to_xy(h["lon"].to_numpy(), h["lat"].to_numpy(),
                               cfg.crs_geographic, cfg.crs_metric)
    centers = grid.grid_centers(panel["grid_id"], cfg).set_index("grid_id")
    cx = panel["grid_id"].map(centers["cx"]).to_numpy(dtype=float)
    cy = panel["grid_id"].map(centers["cy"]).to_numpy(dtype=float)

    # 격자 수 x 소화전 수가 커질 수 있어 블록으로 나눠 계산한다.
    out = np.full(n, np.inf)
    block = 2000
    hpts = np.column_stack([hx, hy])
    for start in range(0, n, block):
        end = min(start + block, n)
        d = np.sqrt(((np.column_stack([cx[start:end], cy[start:end]])[:, None, :]
                      - hpts[None, :, :]) ** 2).sum(axis=2))
        out[start:end] = d.min(axis=1)
    return pd.Series(out, index=panel.index)


def save_city(data: CityData, cfg) -> dict[str, str]:
    """패널과 manifest 를 저장한다. 재현 스크립트가 여기서부터 이어진다."""
    outdir = cfg.paths.processed
    outdir.mkdir(parents=True, exist_ok=True)
    panel_path = outdir / f"panel_{data.city}.parquet"
    data.panel.to_parquet(panel_path, index=False)
    man_path = outdir / f"manifest_{data.city}.json"
    man_path.write_text(json.dumps(data.manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    for name, df in (("fires", data.fires), ("hydrants", data.hydrants)):
        if not df.empty:
            df.to_parquet(outdir / f"{name}_{data.city}.parquet", index=False)
    return {"panel": str(panel_path), "manifest": str(man_path)}


def load_panel(cfg, city: str) -> pd.DataFrame:
    path = cfg.paths.processed / f"panel_{city}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. 먼저 scripts/03_build_dataset.py 를 돌려라.")
    return pd.read_parquet(path)
