"""해상도 정직성 검사.

화재의 3분의 2는 도로명이 없어 **읍면동 중심점**으로 격자에 배정된다.
그러면 같은 동의 화재 수십 건이 한 격자에 쌓인다. 그 격자는 자연히
'화재가 많은 격자'가 되고, 모델은 그걸 쉽게 맞힌다.

즉 **읍면동 뭉침은 포착률을 부풀린다.** 얼마나 부풀리는지 모르면
'상위 20%로 화재 73% 포착'이라는 문장의 뜻을 아무도 모른다.
이 모듈이 그 값을 잰다:

  1) 뭉침 정도      동 중심점 격자가 화재의 몇 %를 담고 있는가
  2) 도로명 한정 평가  도로명이 있는 화재만으로 다시 평가 (해상도가 진짜인 구간)
  3) 뭉침 보정      동 중심점 화재를 동 내부 격자에 대상물 밀도 비례로 흩뿌린 뒤 재평가
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E


def concentration(fires: pd.DataFrame) -> dict:
    """격자별 화재 집중도와, 그 집중이 어느 지오코딩 단계에서 왔는지."""
    if fires.empty or "grid_id" not in fires.columns:
        return {}
    used = fires.dropna(subset=["grid_id"])
    per_grid = used.groupby("grid_id").size().sort_values(ascending=False)
    total = int(per_grid.sum())
    out = {
        "n_fires": total,
        "n_grids": int(len(per_grid)),
        "top1_share": float(per_grid.iloc[0] / total) if total else 0.0,
        "top10_share": float(per_grid.head(10).sum() / total) if total else 0.0,
        "gini": E.gini(per_grid.to_numpy()),
    }
    if "geo_level" in used.columns:
        by_level = used["geo_level"].replace("", "none").value_counts()
        out["by_level"] = {str(k): int(v) for k, v in by_level.items()}
        out["share_from_emd_centroid"] = float(
            by_level.get("emd", 0) + by_level.get("sido_emd", 0)) / max(total, 1)
        # 동 중심점 격자 하나에 몇 건이 쌓였는가
        emd_only = used[used["geo_level"].isin(["emd", "sido_emd"])]
        if not emd_only.empty:
            stack = emd_only.groupby("grid_id").size()
            out["emd_centroid_grids"] = int(len(stack))
            out["max_fires_in_one_centroid_grid"] = int(stack.max())
            out["median_fires_in_centroid_grid"] = float(stack.median())
    return out


def road_only_panel(panel: pd.DataFrame, fires: pd.DataFrame, cfg) -> pd.DataFrame:
    """도로명으로 좌표를 얻은 화재만 라벨로 삼는 패널.

    격자 모집단(피처)은 그대로 두고 라벨만 좁힌다. 해상도가 실제로
    도로 단위인 구간에서 모델이 어떤 성능인지 보기 위해서다.
    """
    from . import features as F

    if "geo_level" not in fires.columns:
        return panel.copy()
    road = fires[fires["geo_level"].isin(["road", "source_xy"])]
    counts = F.fire_counts_by_grid_year(road)

    out = panel.drop(columns=[c for c in F.FIRE_HISTORY_FEATURES + ["fires"]
                              if c in panel.columns])
    skeleton = panel[["grid_id", "year"]].drop_duplicates()
    hist = F.add_fire_history(skeleton, counts, ring=int(cfg["grid"]["neighbor_ring"]))
    merged = hist.merge(out, on=["grid_id", "year"], how="left")
    return merged


def spread_centroid_fires(fires: pd.DataFrame, weights: pd.DataFrame,
                          seed: int = 42) -> pd.DataFrame:
    """동 중심점에 뭉친 화재를 그 동의 격자들에 밀도 비례로 흩뿌린다.

    weights: DataFrame[emd, grid_id, weight]  (보통 대상물 수)
    화재가 실제로 어디서 났는지는 모른다. 다만 **한 점에 전부 몰아두는 것보다는
    건물이 있는 곳에 비례해 나누는 편이 덜 틀린다.** 이건 추정이므로
    보정 전/후를 항상 함께 보고한다.
    """
    if fires.empty or "geo_level" not in fires.columns:
        return fires.copy()
    rng = np.random.default_rng(seed)
    out = fires.copy()
    target = out["geo_level"].isin(["emd", "sido_emd"]) & out["emd"].notna()
    if not target.any() or weights.empty:
        return out

    table = {}
    for emd, grp in weights.groupby("emd"):
        w = grp["weight"].to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        table[emd] = (grp["grid_id"].to_numpy(), w / w.sum())

    idx = out.index[target]
    new_ids = out.loc[idx, "grid_id"].tolist()
    for pos, i in enumerate(idx):
        emd = out.at[i, "emd"]
        if emd in table:
            ids, probs = table[emd]
            new_ids[pos] = str(rng.choice(ids, p=probs))
    out.loc[idx, "grid_id"] = new_ids
    out.loc[idx, "geo_level"] = "emd_spread"
    return out


def build_emd_weights(targets: pd.DataFrame, businesses: pd.DataFrame) -> pd.DataFrame:
    """동별로 '건물이 있는 격자'와 그 밀도. 흩뿌리기의 근거가 된다."""
    frames = []
    for df in (targets, businesses):
        if df is None or df.empty:
            continue
        if not {"emd", "grid_id"}.issubset(df.columns):
            continue
        frames.append(df.dropna(subset=["grid_id", "emd"])[["emd", "grid_id"]])
    if not frames:
        return pd.DataFrame(columns=["emd", "grid_id", "weight"])
    allrows = pd.concat(frames, ignore_index=True)
    return (allrows.groupby(["emd", "grid_id"]).size()
                   .rename("weight").reset_index())


def compare_resolutions(results: dict[str, dict], headline_k: int = 20) -> pd.DataFrame:
    """해상도 시나리오별 핵심 수치를 한 표로.

    '원본(동 뭉침 포함)' 대비 '도로명 한정' / '뭉침 보정' 이 얼마나 떨어지는지가
    곧 뭉침이 부풀린 양이다.
    """
    rows = []
    key = f"top{int(headline_k)}"
    for name, r in results.items():
        if not r:
            continue
        h = r.get("headline", {})
        rows.append({
            "시나리오": name,
            "격자수": r.get("model", {}).get("n_grids"),
            "화재수": r.get("model", {}).get("total_fires"),
            f"모델포착@{headline_k}%": h.get("model_capture"),
            f"베이스라인@{headline_k}%": h.get("baseline_capture"),
            "차이(%p)": h.get("delta_pp"),
            "PAI": r.get("model", {}).get("pai", {}).get(key),
            "PEI": r.get("model", {}).get("pei", {}).get(key),
        })
    df = pd.DataFrame(rows)
    if len(df) > 1 and f"모델포착@{headline_k}%" in df.columns:
        base = df.iloc[0][f"모델포착@{headline_k}%"]
        if base and base == base:
            df["원본대비(%p)"] = (df[f"모델포착@{headline_k}%"] - base) * 100
    return df
