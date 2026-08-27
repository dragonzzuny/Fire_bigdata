"""LightGBM 위험 예측 모델과 검증 프로토콜.

세 가지 검증을 모두 돌린다. 하나만으로는 답이 안 나오기 때문이다.
  1) 시간분할  — 미래를 맞히는가 (2020까지 학습 -> 2021 예측)
  2) LOGO      — 학습에서 통째로 뺀 구·군을 맞히는가
  3) 도시 이식 — 울산에서 배운 모델이 세종에서도 되는가
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from . import evaluate as E

log = logging.getLogger(__name__)

BASELINE_FEATURE = "fires_lag1"   # 베이스라인: '작년에 화재가 많았던 순'


@dataclass
class TrainedModel:
    estimator: LGBMRegressor
    feature_cols: list[str]
    train_years: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            # 도시 이식 시 한쪽에만 있는 업종/등급 컬럼은 0으로 채운다.
            df = df.copy()
            for c in missing:
                df[c] = 0.0
            log.info("예측 대상에 없는 피처 %d개를 0으로 채움: %s", len(missing), missing[:8])
        return np.asarray(self.estimator.predict(df[self.feature_cols]), dtype=float)


def make_estimator(cfg) -> LGBMRegressor:
    p = dict(cfg["model"])
    return LGBMRegressor(verbose=-1, **p)


def fit(panel: pd.DataFrame, feature_cols: list[str], cfg,
        train_years: list[int]) -> TrainedModel:
    tr = panel[panel["year"].isin(train_years)]
    if tr.empty:
        raise ValueError(f"학습 데이터가 비었다. years={train_years}")
    est = make_estimator(cfg)
    est.fit(tr[feature_cols], tr["fires"])
    return TrainedModel(estimator=est, feature_cols=list(feature_cols),
                        train_years=list(train_years),
                        meta={"n_train_rows": int(len(tr))})


def temporal_validation(panel: pd.DataFrame, feature_cols: list[str], cfg) -> dict:
    """2020년까지 학습 -> 한 번도 보지 않은 2021년을 예측한다."""
    train_years = [y for y in sorted(panel["year"].unique()) if y <= cfg.split_year]
    test_year = cfg.holdout_year
    test = panel[panel["year"] == test_year]
    if test.empty:
        raise ValueError(f"홀드아웃 연도 {test_year} 데이터가 없다")

    model = fit(panel, feature_cols, cfg, train_years)
    pred = model.predict(test)
    ks = cfg["evaluation"]["top_k_percents"]
    result = E.compare_with_baseline(
        test["fires"], pred, test[BASELINE_FEATURE], ks, cfg.headline_k,
        cfg["evaluation"]["n_deciles"])
    result["protocol"] = "temporal_holdout"
    result["train_years"] = train_years
    result["test_year"] = int(test_year)

    # 기획서에 없던 것들. '73%가 잘한 건가'에 답하려면 이만큼이 더 필요하다.
    E.extend_with_standard_indices(result["model"], test["fires"], pred, ks)
    E.extend_with_standard_indices(result["baseline"], test["fires"],
                                   test[BASELINE_FEATURE], ks)
    E.add_confidence_intervals(result, test["fires"], pred, test[BASELINE_FEATURE], ks,
                               n_boot=int(cfg["evaluation"].get("n_bootstrap", 1000)))
    result["calibration"] = E.calibration(test["fires"], pred)
    if "sgg" in test.columns:
        eq = E.equity(test["fires"], pred, test["sgg"], cfg.headline_k)
        result["equity"] = eq.to_dict(orient="records")
        result["equity_summary"] = E.equity_summary(eq)
    prev = panel[panel["year"] == test_year - 1]
    if not prev.empty:
        merged = test[["grid_id", "fires"]].merge(
            prev[["grid_id", "fires"]], on="grid_id", how="left",
            suffixes=("_curr", "_prev"))
        result["recapture"] = E.recapture_rate(
            merged["fires_prev"].fillna(0), merged["fires_curr"], pred, cfg.headline_k)

    return {"result": result, "model": model,
            "predictions": test[["grid_id", "year", "fires"]].assign(pred=pred)}


def logo_validation(panel: pd.DataFrame, feature_cols: list[str], cfg,
                    group_col: str | None = None) -> dict:
    """구·군을 하나씩 통째로 빼고 학습해, 뺀 지역을 맞힌다.

    한 지역의 패턴을 외운 것인지, 옮길 수 있는 규칙을 배운 것인지 가른다.
    """
    group_col = group_col or cfg["evaluation"]["logo_column"]
    if group_col not in panel.columns:
        raise KeyError(f"LOGO 그룹 컬럼 '{group_col}' 이 패널에 없다")

    groups = [g for g in sorted(panel[group_col].dropna().unique()) if str(g).strip()]
    per_group = []
    for g in groups:
        tr = panel[panel[group_col] != g]
        te = panel[panel[group_col] == g]
        if te.empty or te["fires"].sum() <= 0 or tr.empty:
            log.info("LOGO 건너뜀(화재 0 또는 데이터 없음): %s", g)
            continue
        est = make_estimator(cfg)
        est.fit(tr[feature_cols], tr["fires"])
        pred = est.predict(te[feature_cols])
        per_group.append({
            "group": str(g),
            "n_grid_years": int(len(te)),
            "total_fires": float(te["fires"].sum()),
            "capture": E.capture_at_k(te["fires"], pred, cfg.headline_k),
            "baseline_capture": E.capture_at_k(te["fires"], te[BASELINE_FEATURE], cfg.headline_k),
        })

    if not per_group:
        raise ValueError(
            f"LOGO 검증에 쓸 수 있는 폴드가 하나도 없다 (그룹 컬럼 '{group_col}'). "
            f"시군구가 전부 비어 있거나 어느 그룹에도 화재가 없다. "
            f"조용히 NaN 을 돌려주면 '검증했는데 결과가 없다'와 '검증을 못 했다'가 "
            f"구분되지 않으므로 여기서 멈춘다.")

    caps = [r["capture"] for r in per_group if r["capture"] == r["capture"]]
    return {
        "protocol": "leave_one_group_out",
        "group_col": group_col,
        "headline_k": cfg.headline_k,
        "per_group": per_group,
        "capture_min": float(np.min(caps)) if caps else float("nan"),
        "capture_max": float(np.max(caps)) if caps else float("nan"),
        "capture_mean": float(np.mean(caps)) if caps else float("nan"),
    }


def transfer_validation(source_panel: pd.DataFrame, target_panel: pd.DataFrame,
                        feature_cols: list[str], cfg,
                        target_years: list[int] | None = None) -> dict:
    """원본 도시에서 학습한 모델을 그대로 다른 도시에 적용한다.

    타깃 도시 데이터는 학습에 단 한 행도 쓰지 않는다.
    """
    train_years = sorted(source_panel["year"].unique().tolist())
    model = fit(source_panel, feature_cols, cfg, train_years)

    te = target_panel
    if target_years:
        te = te[te["year"].isin(target_years)]
    if te.empty or te["fires"].sum() <= 0:
        raise ValueError("이식 검증 대상에 화재가 없다")

    pred = model.predict(te)
    result = E.compare_with_baseline(
        te["fires"], pred, te[BASELINE_FEATURE],
        cfg["evaluation"]["top_k_percents"], cfg.headline_k,
        cfg["evaluation"]["n_deciles"])
    # 이식 검증은 표본이 작은 경우가 많다(세종 2021년 화재 104건).
    # 점추정만 말하면 그 불확실성이 숨는다.
    E.add_confidence_intervals(result, te["fires"], pred, te[BASELINE_FEATURE],
                               cfg["evaluation"]["top_k_percents"],
                               n_boot=int(cfg["evaluation"].get("n_bootstrap", 1000)))
    result["protocol"] = "cross_city_transfer"
    result["source_train_years"] = train_years
    result["target_years"] = sorted(te["year"].unique().tolist())
    return result


def single_feature_probe(panel: pd.DataFrame, cfg,
                         candidates: list[str] | None = None) -> pd.DataFrame:
    """'대상물 수 하나만 봐도 되는 것 아닌가'를 실제로 확인한다.

    기획서가 스스로 인정한 한계다. 숫자로 남겨두면 심사에서
    지적받기 전에 우리가 먼저 답을 갖고 있게 된다.
    """
    test = panel[panel["year"] == cfg.holdout_year]
    if test.empty:
        return pd.DataFrame()
    candidates = candidates or [c for c in ["fires_lag1", "fires_cum", "neigh_fires_lag1",
                                            "target_total", "biz_total", "n_hydrant"]
                                if c in panel.columns]
    rows = []
    for c in candidates:
        rows.append({
            "feature": c,
            f"capture_top{cfg.headline_k}": E.capture_at_k(test["fires"], test[c], cfg.headline_k),
        })
    return pd.DataFrame(rows).sort_values(f"capture_top{cfg.headline_k}", ascending=False)


def ranking_comparison(panel: pd.DataFrame, feature_cols: list[str], cfg) -> dict:
    """같은 프로토콜에서 회귀(Poisson) vs 순위학습(LambdaRank)을 비교한다.

    이 도구의 출력은 '건수'가 아니라 '점검 순서'다. 그렇다면 순위를 직접
    학습하는 편이 목적함수와 맞다 — 라는 가설을 실제로 확인한다.
    기준선(LightGBM Poisson)은 그대로 두고 후보를 옆에 세우는 방식이라,
    기획서 수치와의 연속성이 끊기지 않는다.
    """
    from lightgbm import LGBMRanker

    train_years = [y for y in sorted(panel["year"].unique()) if y <= cfg.split_year]
    test = panel[panel["year"] == cfg.holdout_year]
    tr = panel[panel["year"].isin(train_years)]
    if test.empty or tr.empty:
        return {}

    ks = cfg["evaluation"]["top_k_percents"]
    out: dict = {"protocol": "temporal_holdout", "test_year": int(cfg.holdout_year)}

    # --- 후보 1: Poisson 회귀 (기준선) ---
    base = fit(panel, feature_cols, cfg, train_years)
    out["poisson"] = _score_block(test, base.predict(test), ks, cfg)

    # --- 후보 2: LambdaRank ---
    # 연도를 쿼리 그룹으로 둔다. '그 해 안에서의 순서'가 우리가 원하는 것이다.
    tr_sorted = tr.sort_values("year")
    groups = tr_sorted.groupby("year").size().tolist()
    # 관련도 라벨은 정수여야 한다. 화재 건수를 0~4 로 자른다.
    labels = np.clip(tr_sorted["fires"].to_numpy(), 0, 4).astype(int)
    params = {k: v for k, v in cfg["model"].items() if k != "objective"}
    try:
        ranker = LGBMRanker(objective="lambdarank", verbose=-1,
                            label_gain=list(range(5)), **params)
        ranker.fit(tr_sorted[feature_cols], labels, group=groups)
        out["lambdarank"] = _score_block(test, ranker.predict(test[feature_cols]), ks, cfg)
    except Exception as exc:                                   # noqa: BLE001
        log.warning("LambdaRank 학습 실패: %s", exc)
        out["lambdarank"] = {"error": str(exc)}

    # --- 후보 3: 베이스라인 (작년 화재 순) ---
    out["baseline_last_year"] = _score_block(test, test[BASELINE_FEATURE], ks, cfg)

    key = f"top{int(cfg.headline_k)}"
    ranked = {k: v["capture"][key] for k, v in out.items()
              if isinstance(v, dict) and "capture" in v}
    if ranked:
        out["winner"] = max(ranked, key=ranked.get)
        out["capture_by_model"] = ranked
    return out


def _score_block(test: pd.DataFrame, score, ks: list[float], cfg) -> dict:
    r = E.evaluate_ranking(test["fires"], score, ks, cfg["evaluation"]["n_deciles"])
    E.extend_with_standard_indices(r, test["fires"], score, ks)
    return r
