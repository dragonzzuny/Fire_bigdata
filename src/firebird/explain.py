"""SHAP: 이 격자의 위험을 '무엇이' 끌어올렸는가.

점수만 주면 현장은 움직이지 않는다. 새로 부임한 대원이 '왜 여기부터인가'를
설명할 수 있어야 그 순위가 실제 점검 계획이 된다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap

log = logging.getLogger(__name__)

# 피처 이름 -> 현장에서 쓰는 말
FEATURE_LABELS = {
    "fires_lag1": "작년 화재 건수",
    "fires_lag2": "재작년 화재 건수",
    "fires_cum": "누적 화재 건수",
    "fires_mean_prev": "과거 연평균 화재",
    "neigh_fires_lag1": "주변 격자 작년 화재(전파)",
    "neigh_fires_cum": "주변 격자 누적 화재(전파)",
    "target_total": "특정소방대상물 수",
    "usage_total": "특정소방대상물 수",
    "fac_total": "소방시설 설치대상 수",
    "biz_total": "다중이용업소 수",
    "n_hydrant": "소화전 수",
    "dist_hydrant_m": "최근접 소화전 거리",
}


def label_of(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix, kor in (("usage_n_", "대상물 용도 "), ("usage_share_", "대상물 용도비율 "),
                        ("fac_n_", "소방시설 "), ("fac_share_", "소방시설 비율 "),
                        ("biz_n_", "다중이용업소 "), ("biz_share_", "업소 비율 ")):
        if feature.startswith(prefix):
            return kor + feature[len(prefix):]
    return feature


def compute_shap(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model.estimator)
    values = explainer.shap_values(X[model.feature_cols])
    return np.asarray(values, dtype=float)


def top_drivers(shap_row: np.ndarray, feature_cols: list[str], values: pd.Series,
                top_n: int = 5, positive_only: bool = True) -> list[dict]:
    """한 격자의 위험 상승 요인 상위 N개."""
    order = np.argsort(-shap_row)
    out = []
    for i in order:
        contrib = float(shap_row[i])
        if positive_only and contrib <= 0:
            break
        f = feature_cols[i]
        out.append({"feature": f, "label": label_of(f),
                    "value": float(values.get(f, np.nan)), "contribution": contrib})
        if len(out) >= top_n:
            break
    return out


def explain_grids(model, panel_slice: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """격자별 상위 기여요인 표. 점검계획서의 '위험을 높인 요인'이 된다."""
    if panel_slice.empty:
        return pd.DataFrame(columns=["grid_id", "drivers"])
    X = panel_slice.copy()
    for c in model.feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    sv = compute_shap(model, X)
    rows = []
    for pos, (_, row) in enumerate(X.iterrows()):
        rows.append({"grid_id": row["grid_id"],
                     "drivers": top_drivers(sv[pos], model.feature_cols,
                                            row[model.feature_cols], top_n)})
    return pd.DataFrame(rows)


def global_importance(model, X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """전역 중요도(|SHAP| 평균). 모델이 무엇을 보고 있는지 한 장으로."""
    sv = compute_shap(model, X)
    imp = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({"feature": model.feature_cols,
                          "label": [label_of(f) for f in model.feature_cols],
                          "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n).reset_index(drop=True))
