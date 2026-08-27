"""격자x연도 패널 피처.

누수 방지 규칙: 연도 t 의 피처는 t-1 까지의 화재만 쓴다.
`assert_no_leakage` 가 이 규칙을 코드로 강제한다 — 주석으로만 있는 규칙은
언젠가 깨지고, 깨진 것을 아무도 모른다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIRE_HISTORY_FEATURES = [
    "fires_lag1", "fires_lag2", "fires_cum", "fires_mean_prev",
    "neigh_fires_lag1", "neigh_fires_cum",
]


def _split_grid_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["grid_id"].astype(str).str.split("_", n=1, expand=True)
    out["gx"] = parts[0].astype(int)
    out["gy"] = parts[1].astype(int)
    return out


def fire_counts_by_grid_year(fires: pd.DataFrame) -> pd.DataFrame:
    """화재 원본 -> DataFrame[grid_id, year, fires]."""
    sub = fires.dropna(subset=["grid_id", "year"])
    out = (sub.groupby(["grid_id", "year"]).size()
              .rename("fires").reset_index())
    out["year"] = out["year"].astype(int)
    return out


def build_panel_skeleton(grid_ids: pd.Series, years: list[int]) -> pd.DataFrame:
    """모든 (격자, 연도) 조합. 화재가 0인 격자도 학습에 반드시 들어가야 한다."""
    ids = pd.Series(pd.unique(pd.Series(grid_ids).dropna().astype(str)), name="grid_id")
    return (pd.MultiIndex.from_product([ids, years], names=["grid_id", "year"])
              .to_frame(index=False))


def add_fire_history(panel: pd.DataFrame, counts: pd.DataFrame, *, ring: int = 1) -> pd.DataFrame:
    """과거 화재 피처(자기 격자 + 이웃 전파)를 붙인다.

    모든 값은 t-1 이하 정보만 사용한다.
    """
    df = panel.merge(counts, on=["grid_id", "year"], how="left")
    df["fires"] = df["fires"].fillna(0.0).astype(float)
    df = df.sort_values(["grid_id", "year"]).reset_index(drop=True)

    g = df.groupby("grid_id", sort=False)["fires"]
    df["fires_lag1"] = g.shift(1).fillna(0.0)
    df["fires_lag2"] = g.shift(2).fillna(0.0)
    # cumsum 을 한 칸 밀어 t-1 까지의 누적으로 만든다 (t 를 포함하면 라벨 누수).
    df["fires_cum"] = g.cumsum() - df["fires"]
    year_idx = df.groupby("grid_id", sort=False).cumcount()
    df["fires_mean_prev"] = np.where(year_idx > 0, df["fires_cum"] / year_idx.replace(0, 1), 0.0)

    # --- 이웃 격자 전파: 8-이웃의 lag1/cum 합 ---
    key = _split_grid_id(df[["grid_id", "year"]].drop_duplicates())
    base = df[["grid_id", "year", "fires_lag1", "fires_cum"]].merge(
        key, on=["grid_id", "year"], how="left")

    offsets = [(dx, dy)
               for dx in range(-ring, ring + 1)
               for dy in range(-ring, ring + 1)
               if not (dx == 0 and dy == 0)]

    acc_lag = np.zeros(len(base), dtype=float)
    acc_cum = np.zeros(len(base), dtype=float)
    lookup = base.set_index(["gx", "gy", "year"])[["fires_lag1", "fires_cum"]]
    for dx, dy in offsets:
        probe = pd.MultiIndex.from_arrays(
            [base["gx"] + dx, base["gy"] + dy, base["year"]])
        got = lookup.reindex(probe)
        acc_lag += got["fires_lag1"].to_numpy(dtype=float, na_value=0.0)
        acc_cum += got["fires_cum"].to_numpy(dtype=float, na_value=0.0)

    base["neigh_fires_lag1"] = acc_lag
    base["neigh_fires_cum"] = acc_cum
    return df.merge(base[["grid_id", "year", "gx", "gy",
                          "neigh_fires_lag1", "neigh_fires_cum"]],
                    on=["grid_id", "year"], how="left")


def counts_by_category(df: pd.DataFrame, cat_col: str, prefix: str,
                       categories: list[str] | None = None) -> pd.DataFrame:
    """격자별 범주 개수와 비율. 정적(스냅샷) 피처."""
    sub = df.dropna(subset=["grid_id"]).copy()
    sub[cat_col] = sub[cat_col].fillna("기타").astype(str)
    if categories:
        sub.loc[~sub[cat_col].isin(categories), cat_col] = "기타"
    wide = (sub.groupby(["grid_id", cat_col]).size().unstack(fill_value=0))
    wide.columns = [f"{prefix}_n_{c}" for c in wide.columns]
    total = wide.sum(axis=1)
    shares = wide.div(total.replace(0, np.nan), axis=0).fillna(0.0)
    shares.columns = [c.replace(f"{prefix}_n_", f"{prefix}_share_") for c in wide.columns]
    out = pd.concat([wide, shares], axis=1)
    out[f"{prefix}_total"] = total
    return out.reset_index()


def attach_static(panel: pd.DataFrame, *tables: pd.DataFrame) -> pd.DataFrame:
    """정적 피처 테이블들을 격자 기준으로 붙이고 결측은 0으로 채운다."""
    out = panel
    for t in tables:
        if t is None or t.empty:
            continue
        out = out.merge(t, on="grid_id", how="left")
    new_cols = [c for c in out.columns if c not in panel.columns]
    out[new_cols] = out[new_cols].fillna(0.0)
    return out


def assert_no_leakage(panel: pd.DataFrame) -> None:
    """t 년 라벨이 t 년 피처에 새어 들어가지 않았는지 실제로 확인한다.

    fires_cum 은 정의상 t-1 까지의 합이므로, 어떤 행에서도
    (누적 + 당해) 관계가 깨지면 안 된다.
    """
    df = panel.sort_values(["grid_id", "year"])
    g = df.groupby("grid_id", sort=False)
    expected_cum = g["fires"].cumsum() - df["fires"]
    bad = (df["fires_cum"] - expected_cum).abs() > 1e-9
    if bad.any():
        raise AssertionError(f"누수 감지: fires_cum 이 당해 화재를 포함한 행 {int(bad.sum())}개")

    expected_lag1 = g["fires"].shift(1).fillna(0.0)
    bad = (df["fires_lag1"] - expected_lag1).abs() > 1e-9
    if bad.any():
        raise AssertionError(f"누수 감지: fires_lag1 불일치 행 {int(bad.sum())}개")


#: 라벨, 식별자, 위치 좌표, 그리고 예측 단계에서 붙는 파생 컬럼.
#: 좌표(gx/gy/lon/lat)를 피처로 쓰면 모델이 '위험한 동네의 위치'를 외운다.
#: 그 모델은 같은 도시에서만 잘 맞고 새 도시로 옮기면 무너지므로, 여기서 막는다.
NON_FEATURE_COLUMNS = {
    "fires", "grid_id", "year", "city", "sgg", "sido", "road",
    "gx", "gy", "gx_c", "gy_c", "cx", "cy", "lon", "lat",
    "pred", "risk_score", "rank", "percentile",
}


#: 스냅샷(현재 시점) 피처의 접두사.
#:
#: 특정소방대상물·다중이용업소·소방용수시설 공개 데이터에는 '언제부터 존재했는가'가
#: 없다. 즉 2014년 행에도 현재 시점의 업소 수가 들어간다. 이는 미래 정보가 과거
#: 학습행에 섞이는 것이므로, 시간분할 검증을 '미래에 안전하다'고 말하려면
#: 이 피처들을 뺀 결과도 함께 봐야 한다. 데이터로는 못 없애는 한계이므로
#: 숨기는 대신 측정한다 (scripts/04_train_eval.py 의 [1b]).
SNAPSHOT_FEATURE_PREFIXES = ("target_", "biz_")
SNAPSHOT_FEATURE_NAMES = {"n_hydrant", "dist_hydrant_m"}


def is_snapshot_feature(name: str) -> bool:
    return (name in SNAPSHOT_FEATURE_NAMES
            or str(name).startswith(SNAPSHOT_FEATURE_PREFIXES))


def history_only_columns(panel: pd.DataFrame) -> list[str]:
    """과거 화재 이력만으로 이루어진 피처. 시점이 확실한 것만 남는다."""
    return [c for c in feature_columns(panel) if not is_snapshot_feature(c)]


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """모델 입력 컬럼. 라벨·식별자·좌표는 제외한다."""
    return [c for c in panel.columns
            if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(panel[c])]
