"""원본 CSV -> 격자x연도 의사결정 테이블.

한 도시분 전체를 이 모듈이 책임진다:
    적재 -> 주소 조각 정규화 -> 계층 지오코딩 -> 500m 격자 -> 라벨/피처 -> 패널

깎여나간 양은 전부 manifest 에 기록한다. 좌표 확보율뿐 아니라 **어느 정밀도로**
얻었는지(도로 / 읍면동)까지 남긴다. 동 중심점으로 찍힌 화재를 도로 단위처럼
말하면 격자 해상도를 실제보다 좋게 주장하는 셈이 되기 때문이다.
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


# ---------------------------------------------------------------- 시각 처리

def parse_datetime(series: pd.Series) -> pd.Series:
    """'20210101143000' 형식을 우선 시도하고, 실패분만 일반 파서로 넘긴다."""
    s = series.astype(str).str.strip().str.replace(r"\D", "", regex=True)
    out = pd.to_datetime(s.str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    need = out.isna()
    if need.any():
        out.loc[need] = pd.to_datetime(s[need].str.slice(0, 8), format="%Y%m%d",
                                       errors="coerce")
    return out


def has_real_time(series: pd.Series) -> pd.Series:
    """시각이 000000 이 아닌 행. 플랫폼 _2021 파일은 시각이 전부 0 이다."""
    s = series.astype(str).str.strip().str.replace(r"\D", "", regex=True)
    return (s.str.len() >= 14) & (s.str.slice(8, 14) != "000000")


# ---------------------------------------------------------------- 범주 정규화

def normalize_by_keywords(series: pd.Series, keywords: dict[str, list[str]],
                          default: str = "기타") -> pd.Series:
    """부분일치 키워드로 원본 범주를 표준 범주에 접는다.

    나중에 선언된 키워드가 이긴다 — 더 구체적인 규칙을 아래에 두면 된다.
    """
    s = series.fillna("").astype(str).str.replace(r"\s", "", regex=True)
    out = pd.Series(default, index=s.index, dtype=object)
    for std, kws in (keywords or {}).items():
        for kw in kws or []:
            out[s.str.contains(str(kw).replace(" ", ""), na=False, case=False)] = std
    return out


# ---------------------------------------------------------------- 좌표 검증

def within_bbox(lon: pd.Series, lat: pd.Series, bbox: dict | None) -> pd.Series:
    """좌표가 그 도시 경계 안에 있는가. bbox 가 없으면 전부 통과."""
    lon = pd.to_numeric(lon, errors="coerce")
    lat = pd.to_numeric(lat, errors="coerce")
    if not bbox:
        return lon.notna() & lat.notna()
    return (lon.between(bbox["lon_min"], bbox["lon_max"])
            & lat.between(bbox["lat_min"], bbox["lat_max"]))


# ---------------------------------------------------------------- 도시 적재

def select_fire_source_per_year(fires: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """같은 연도가 여러 원본 파일에 들어 있으면 하나만 남긴다.

    플랫폼은 같은 데이터셋을 `_0000`(누적)과 `_2021`(최신) 두 벌로 준다.
    울산 화재는 두 파일의 기간이 겹치지 않아(2005~2020 / 2021) 합쳐야 하지만,
    세종 화재는 같은 사건이 양쪽에 들어 있고 `_2021` 쪽은 **일시가 뭉개져** 있다
    (시각 000000, 날짜도 월초로 밀림). 그냥 concat 하면 2012~2020 화재가
    거의 두 배로 세어진다 — 라벨이 두 배가 되면 그 뒤 모든 수치가 무의미하다.

    규칙: 연도마다 '시각이 살아있는 비율'이 높은 파일을 쓰고, 같으면 행이 많은 쪽.
    """
    if fires.empty or "_source_file" not in fires.columns:
        return fires, {}

    stat = (fires.assign(_t=fires["has_time"].astype(float))
                 .groupby(["year", "_source_file"], dropna=True)
                 .agg(n=("_t", "size"), time_rate=("_t", "mean"))
                 .reset_index())
    if stat.empty:
        return fires, {}

    chosen = (stat.sort_values(["year", "time_rate", "n"], ascending=[True, False, False])
                  .drop_duplicates("year")[["year", "_source_file"]])
    keep = fires.merge(chosen, on=["year", "_source_file"], how="inner")

    decisions = {}
    for year, grp in stat.groupby("year"):
        if len(grp) > 1:
            picked = chosen.loc[chosen["year"] == year, "_source_file"].iloc[0]
            decisions[int(year)] = {
                "picked": picked,
                "dropped": [{"file": r["_source_file"], "rows": int(r["n"])}
                            for _, r in grp.iterrows() if r["_source_file"] != picked],
            }
    return keep, decisions


def load_city(cfg, city: str, *, use_api: bool = True) -> CityData:
    """한 도시의 원본을 전부 읽어 격자 패널까지 만든다."""
    city_conf = cfg.city(city)
    label = city_conf["label"]
    bbox = city_conf.get("bbox")
    manifest: dict[str, Any] = {"city": city, "label": label,
                                "grid_size_m": cfg.grid_size_m}

    def prepare(kind: str, required: bool = True) -> pd.DataFrame:
        try:
            raw = load_dataset(cfg, city, kind)
        except FileNotFoundError:
            if required:
                raise
            log.warning("[%s] %s 데이터 없음 — 관련 피처 비활성", city, kind)
            return pd.DataFrame()
        return addresses.build_keys(raw, label)

    fires = prepare("fire")
    targets = prepare("target")
    businesses = prepare("business")
    hydrants = prepare("hydrant", required=False)

    # --- 지오코딩: 모든 테이블의 모든 정밀도 키를 모아 한 번에 ---
    keys: list[str] = []
    for t in (fires, targets, businesses, hydrants):
        if not t.empty:
            keys += addresses.all_keys(t)
    keys = list(dict.fromkeys(keys))
    coords = geocode.geocode_keys(keys, cfg, allow_network=use_api)
    manifest["geocode"] = {
        "unique_keys": int(len(coords)),
        "matched_keys": int(coords["matched"].sum()) if len(coords) else 0,
        "key_match_rate": float(coords["matched"].mean()) if len(coords) else 0.0,
    }

    def to_grid(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if df.empty:
            return df
        out = addresses.resolve(df, coords)

        # 원본 좌표가 있으면 경계 검증 후 우선 채택한다.
        if {"lat", "lon"}.issubset(df.columns):
            olon = pd.to_numeric(df["lon"], errors="coerce")
            olat = pd.to_numeric(df["lat"], errors="coerce")
            ok = within_bbox(olon, olat, bbox)
            n_have = int((olon.notna() & olat.notna()).sum())
            n_ok = int(ok.sum())
            if n_have and n_ok < n_have:
                log.warning("[%s/%s] 원본 좌표 %d건 중 %d건이 %s 경계 밖 — 버리고 "
                            "주소 지오코딩으로 대체", city, name, n_have, n_have - n_ok, label)
            out.loc[ok, "lon"] = olon[ok].astype("Float64")
            out.loc[ok, "lat"] = olat[ok].astype("Float64")
            out.loc[ok, "geo_level"] = "source_xy"
            manifest.setdefault("source_coords", {})[name] = {
                "present": n_have, "within_bbox": n_ok}

        out = grid.assign_grid(out, cfg)
        n = len(out)
        got = int(out["lon"].notna().sum())
        manifest.setdefault("coverage", {})[name] = {
            "rows": n, "with_coords": got, "rate": (got / n) if n else 0.0,
            "by_level": addresses.level_breakdown(out),
        }
        log.info("[%s/%s] 좌표 %d/%d (%.1f%%) · 단계별 %s",
                 city, name, got, n, (got / n * 100) if n else 0.0,
                 manifest["coverage"][name]["by_level"])
        return out

    fires = to_grid(fires, "fire")
    targets = to_grid(targets, "target")
    businesses = to_grid(businesses, "business")
    hydrants = to_grid(hydrants, "hydrant")

    # --- 라벨: 연도·시각 ---
    dt = parse_datetime(fires["occurred_at"])
    fires["occurred_dt"] = dt
    fires["year"] = dt.dt.year.astype("Int64")
    real_time = has_real_time(fires["occurred_at"])
    fires["has_time"] = real_time
    fires["hour"] = dt.dt.hour.where(real_time).astype("Int64")
    fires["weekday"] = dt.dt.dayofweek.astype("Int64")

    if "kind" in fires.columns:
        not_fire = fires["kind"].notna() & (fires["kind"] != "화재")
        if not_fire.any():
            log.info("[%s] 재난종별이 '화재'가 아닌 %d행 제외", city, int(not_fire.sum()))
            fires = fires[~not_fire].copy()

    fires, source_choice = select_fire_source_per_year(fires)
    if source_choice:
        manifest["fire_source_selection"] = source_choice
        for year, d in sorted(source_choice.items()):
            log.info("[%s] %d년: %s 채택, %s 제외", city, year,
                     Path(d["picked"]).name,
                     [x["file"] + f"({x['rows']}행)" for x in d["dropped"]])

    in_range = fires["year"].between(cfg.year_min, cfg.year_max)
    usable = in_range & fires["grid_id"].notna()
    manifest["fires"] = {
        "rows": int(len(fires)),
        "with_year": int(fires["year"].notna().sum()),
        "in_year_range": int(in_range.sum()),
        "with_grid": int(fires["grid_id"].notna().sum()),
        "usable": int(usable.sum()),
        "year_range": [cfg.year_min, cfg.year_max],
        "with_real_time": int(fires["has_time"].sum()),
        "time_note": ("플랫폼 _2021 파일은 시각이 000000 이라 시간대 분석에서 제외된다"),
        "by_year": {int(k): int(v) for k, v in
                    fires.loc[usable, "year"].value_counts().sort_index().items()},
    }
    fires_used = fires[usable].copy()

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
    schema = cfg.schema

    universe = pd.concat([t["grid_id"] for t in (fires, targets, businesses, hydrants)
                          if not t.empty and "grid_id" in t.columns],
                         ignore_index=True).dropna()
    if universe.empty:
        raise ValueError("격자를 하나도 만들지 못했다 — 지오코딩이 전부 실패했을 가능성이 크다. "
                         "KAKAO_REST_API_KEY 와 scripts/02_geocode.py 결과를 확인하라.")

    counts = F.fire_counts_by_grid_year(fires)
    skeleton = F.build_panel_skeleton(universe, years)
    panel = F.add_fire_history(skeleton, counts, ring=int(cfg["grid"]["neighbor_ring"]))

    tables: list[pd.DataFrame] = []

    if not targets.empty:
        t = targets.dropna(subset=["grid_id"]).copy()
        if "usage" in t.columns:
            t["usage_std"] = normalize_by_keywords(t["usage"], schema.get("usage_keywords", {}))
            tables.append(F.counts_by_category(t, "usage_std", "usage",
                                               categories=list(schema.get("usage_keywords", {}))))
        if "facility" in t.columns:
            t["facility_std"] = normalize_by_keywords(t["facility"],
                                                      schema.get("facility_keywords", {}))
            tables.append(F.counts_by_category(t, "facility_std", "fac",
                                               categories=list(schema.get("facility_keywords", {}))))
        tables.append(t.groupby("grid_id").size().rename("target_total").reset_index())

    if not businesses.empty:
        b = businesses.dropna(subset=["grid_id"]).copy()
        if "biz_type" in b.columns:
            b["biz_std"] = normalize_by_keywords(b["biz_type"],
                                                 schema.get("biz_type_keywords", {}))
            tables.append(F.counts_by_category(b, "biz_std", "biz",
                                               categories=list(schema.get("biz_type_keywords", {}))))
        else:
            tables.append(b.groupby("grid_id").size().rename("biz_total").reset_index())

    if not hydrants.empty and "grid_id" in hydrants.columns:
        h = hydrants.dropna(subset=["grid_id"]).copy()
        if "facility_type" in h.columns:
            h["hyd_std"] = normalize_by_keywords(h["facility_type"],
                                                 schema.get("hydrant_keywords", {}))
            # 대응취약의 분모는 '소화전'이다. 저수조·급수탑을 같이 세면
            # 소화전이 없는 격자가 있는 것처럼 보인다.
            hyd = h[h["hyd_std"] == "소화전"]
        else:
            hyd = h
        tables.append(hyd.groupby("grid_id").size().rename("n_hydrant").reset_index())

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
    """격자별 대표 시군구. LOGO 검증의 그룹 키가 된다.

    세종처럼 시군구가 없는 단층제는 읍면동으로 대신한다 —
    그래야 '한 지역을 빼고 학습' 검증을 아예 못 하는 상황을 피한다.
    """
    frames = []
    for s in sources:
        if s.empty or "grid_id" not in s.columns:
            continue
        col = None
        if "sgg" in s.columns and s["sgg"].astype(str).str.strip().ne("").any():
            col = "sgg"
        elif "emd" in s.columns:
            col = "emd"
        if col:
            frames.append(s[["grid_id", col]].rename(columns={col: "sgg"}))

    if not frames:
        out = panel.copy()
        out["sgg"] = ""
        return out

    allsgg = pd.concat(frames, ignore_index=True).dropna(subset=["grid_id"])
    allsgg["sgg"] = allsgg["sgg"].fillna("").astype(str).str.strip()
    allsgg = allsgg[allsgg["sgg"] != ""]
    if allsgg.empty:
        out = panel.copy()
        out["sgg"] = ""
        return out

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

    hx, hy = grid.lonlat_to_xy(h["lon"].astype(float).to_numpy(),
                              h["lat"].astype(float).to_numpy(),
                              cfg.crs_geographic, cfg.crs_metric)
    centers = grid.grid_centers(panel["grid_id"], cfg).set_index("grid_id")
    cx = panel["grid_id"].map(centers["cx"]).to_numpy(dtype=float)
    cy = panel["grid_id"].map(centers["cy"]).to_numpy(dtype=float)

    out = np.full(n, np.inf)
    hpts = np.column_stack([hx, hy])
    block = 2000
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
    man_path.write_text(json.dumps(data.manifest, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    for name, df in (("fires", data.fires), ("hydrants", data.hydrants)):
        if not df.empty:
            df.drop(columns=[c for c in ("occurred_dt",) if c in df.columns]) \
              .to_parquet(outdir / f"{name}_{data.city}.parquet", index=False)
    return {"panel": str(panel_path), "manifest": str(man_path)}


def load_panel(cfg, city: str) -> pd.DataFrame:
    path = cfg.paths.processed / f"panel_{city}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. 먼저 scripts/03_build_dataset.py 를 돌려라.")
    return pd.read_parquet(path)
