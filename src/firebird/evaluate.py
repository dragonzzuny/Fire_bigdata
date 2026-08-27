"""평가 지표.

핵심 질문은 하나다: "위험 상위 k% 격자에 점검·순찰을 집중하면
그해 실제 화재의 몇 %가 그 안에서 발생했는가."

동점 처리에 주의한다. 베이스라인('작년 화재 순')은 화재 0인 격자가
수천 개라 동점 덩어리가 거대하다. 정렬이 입력 순서에 의존하면
베이스라인 점수가 데이터 정렬 방식에 따라 흔들린다 — 그건 성능이 아니라
우연이다. 그래서 동점은 고정 시드 난수로 깬다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _order_desc(score: np.ndarray, seed: int = 42) -> np.ndarray:
    """점수 내림차순 정렬 인덱스. 동점은 고정 시드로 무작위 배치."""
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(score))
    return np.lexsort((jitter, -np.asarray(score, dtype=float)))


def capture_at_k(y_true, score, k_percent: float, seed: int = 42) -> float:
    """상위 k% 격자가 담은 실제 화재의 비율."""
    y = np.asarray(y_true, dtype=float)
    total = y.sum()
    if total <= 0:
        return float("nan")
    n_top = max(1, int(round(len(y) * k_percent / 100.0)))
    order = _order_desc(score, seed)[:n_top]
    return float(y[order].sum() / total)


def lift_at_k(y_true, score, k_percent: float, seed: int = 42) -> float:
    """무작위 대비 몇 배인가. capture / (k/100)."""
    cap = capture_at_k(y_true, score, k_percent, seed)
    return float(cap / (k_percent / 100.0)) if cap == cap else float("nan")


def decile_table(y_true, score, n_deciles: int = 10, seed: int = 42) -> pd.DataFrame:
    """위험 등급별 실제 화재 평균. 1등급=최저위험 ... n등급=최고위험.

    등급이 올라갈수록 실제 화재가 단조 증가하면 '등급이 실제와 맞는다'.
    """
    y = np.asarray(y_true, dtype=float)
    order = _order_desc(score, seed)
    ranks = np.empty(len(y), dtype=int)
    ranks[order] = np.arange(len(y))          # 0 = 최고위험
    # 최고위험이 마지막 등급이 되도록 뒤집는다.
    grade = n_deciles - np.floor(ranks / len(y) * n_deciles).astype(int)
    grade = np.clip(grade, 1, n_deciles)
    df = pd.DataFrame({"grade": grade, "fires": y, "score": np.asarray(score, dtype=float)})
    out = (df.groupby("grade")
             .agg(n_grids=("fires", "size"),
                  mean_fires=("fires", "mean"),
                  total_fires=("fires", "sum"),
                  mean_score=("score", "mean"))
             .reset_index()
             .sort_values("grade"))
    out["share_of_fires"] = out["total_fires"] / max(y.sum(), 1e-9)
    return out


def is_monotonic(decile: pd.DataFrame) -> bool:
    """등급 1->N 으로 갈수록 평균 화재가 (약하게) 증가하는가."""
    v = decile.sort_values("grade")["mean_fires"].to_numpy()
    return bool(np.all(np.diff(v) >= -1e-12))


def evaluate_ranking(y_true, score, k_percents: list[float],
                     n_deciles: int = 10, seed: int = 42) -> dict:
    """한 예측에 대한 전체 지표 묶음."""
    dec = decile_table(y_true, score, n_deciles, seed)
    return {
        "n_grids": int(len(np.asarray(y_true))),
        "total_fires": float(np.asarray(y_true, dtype=float).sum()),
        "capture": {f"top{int(k)}": capture_at_k(y_true, score, k, seed) for k in k_percents},
        "lift": {f"top{int(k)}": lift_at_k(y_true, score, k, seed) for k in k_percents},
        "decile": dec.to_dict(orient="records"),
        "decile_monotonic": is_monotonic(dec),
    }


def compare_with_baseline(y_true, model_score, baseline_score,
                          k_percents: list[float], headline_k: int = 20,
                          n_deciles: int = 10, seed: int = 42) -> dict:
    """모델 vs 베이스라인('작년 화재가 많았던 순').

    베이스라인이 없으면 '73% 포착'이 잘한 건지 알 수 없다.
    도시 대부분의 화재는 원래 소수 격자에 몰려 있기 때문이다.
    """
    model = evaluate_ranking(y_true, model_score, k_percents, n_deciles, seed)
    base = evaluate_ranking(y_true, baseline_score, k_percents, n_deciles, seed)
    key = f"top{int(headline_k)}"
    return {
        "model": model,
        "baseline": base,
        "headline_k": int(headline_k),
        "headline": {
            "model_capture": model["capture"][key],
            "baseline_capture": base["capture"][key],
            "delta_pp": (model["capture"][key] - base["capture"][key]) * 100.0,
            "model_lift": model["lift"][key],
        },
    }


def format_report(result: dict) -> str:
    """사람이 읽는 요약. 보고서와 콘솔에 같은 문장이 나가게 한다."""
    h = result["headline"]
    m, b = result["model"], result["baseline"]
    lines = [
        f"격자 {m['n_grids']:,}개 · 실제 화재 {m['total_fires']:.0f}건",
        "",
        f"{'상위%':>6} {'모델포착':>9} {'베이스라인':>11} {'차이(%p)':>9} {'리프트':>7}",
    ]
    for key in m["capture"]:
        mk, bk = m["capture"][key], b["capture"][key]
        lines.append(f"{key[3:]+'%':>6} {mk:>8.1%} {bk:>10.1%} "
                     f"{(mk-bk)*100:>+9.1f} {m['lift'][key]:>6.2f}x")
    lines += [
        "",
        f"핵심(상위 {result['headline_k']}%): 모델 {h['model_capture']:.1%} vs "
        f"베이스라인 {h['baseline_capture']:.1%} ({h['delta_pp']:+.1f}%p), "
        f"무작위 대비 {h['model_lift']:.2f}배",
        f"등급 단조성: {'통과' if m['decile_monotonic'] else '깨짐'}",
    ]
    return "\n".join(lines)


# ===========================================================================
# 예측치안(predictive policing) 표준 지표
#
# 기획서는 '상위 20% 포착률' 하나로 성능을 말한다. 그 수치만으로는
# "73%가 잘한 건가"에 답할 수 없다. 화재가 원래 몇 개 격자에 몰려 있으면
# 아무 모델이나 높은 포착률을 낸다. 아래 두 지표가 그 질문에 답한다.
#
#   PAI  = 포착률 / 면적비율        무작위 대비 몇 배인가 (=lift)
#   PEI  = PAI / PAI_최대           **이론상 최선 대비 몇 %인가**
#
# PEI 가 핵심이다. 같은 도시에서 아무리 좋은 모델도 넘을 수 없는 상한이 있고,
# PAI 만 보면 그 상한이 낮은 도시에서 나온 낮은 PAI 를 성능 부족으로 오해한다.
# Chainey et al.(2008) 의 PAI, Hunt(2016) 의 PEI 정의를 따른다.
# ===========================================================================

def pai(y_true, score, k_percent: float, seed: int = 42) -> float:
    """Predictive Accuracy Index. 상위 k% 면적이 담은 화재 비율 / (k/100)."""
    return lift_at_k(y_true, score, k_percent, seed)


def pai_max(y_true, k_percent: float) -> float:
    """이 데이터에서 도달 가능한 PAI 의 상한.

    실제 화재 건수 자체로 줄을 세운 '사후 최적' 순위의 PAI.
    미래를 아는 예언자도 이보다 잘할 수 없다.
    """
    y = np.asarray(y_true, dtype=float)
    if y.sum() <= 0:
        return float("nan")
    n_top = max(1, int(round(len(y) * k_percent / 100.0)))
    best = np.sort(y)[::-1][:n_top].sum() / y.sum()
    return float(best / (k_percent / 100.0))


def pei(y_true, score, k_percent: float, seed: int = 42) -> float:
    """Predictive Efficiency Index = PAI / PAI_max. 0~1.

    '이론상 최선의 몇 %인가'. 도시가 달라도 비교할 수 있는 유일한 축이다.
    """
    top = pai_max(y_true, k_percent)
    if not top or top != top or top == 0:
        return float("nan")
    return float(pai(y_true, score, k_percent, seed) / top)


def recapture_rate(y_prev, y_curr, score, k_percent: float, seed: int = 42) -> dict:
    """작년 화재 격자가 올해도 나는가 (RRI).

    베이스라인('작년 화재 순')이 왜 그렇게 센지를 설명하는 수치다.
    재발률이 높은 도시에서는 단순 베이스라인이 강하고, 모델이 이길 여지가 좁다.
    """
    prev = np.asarray(y_prev, dtype=float)
    curr = np.asarray(y_curr, dtype=float)
    had = prev > 0
    return {
        "grids_with_prev_fire": int(had.sum()),
        "share_of_grids": float(had.mean()) if len(prev) else 0.0,
        "repeat_rate": float((curr[had] > 0).mean()) if had.any() else float("nan"),
        "share_of_curr_fires_in_prev_grids": (float(curr[had].sum() / curr.sum())
                                              if curr.sum() > 0 else float("nan")),
    }


# ---------------------------------------------------------------- 형평성

def equity(y_true, score, groups, k_percent: float, seed: int = 42) -> pd.DataFrame:
    """구·군별로 '점검 배분'과 '실제 위험 배분'이 얼마나 어긋나는가.

    기획서는 '효율·형평성'을 내세우면서 형평성을 재지 않는다.
    특정 구에만 점검이 몰리는데 그 구의 화재 비중이 그만큼이 아니라면,
    그건 효율이 아니라 편중이다. 그 차이를 여기서 숫자로 만든다.
    """
    y = np.asarray(y_true, dtype=float)
    g = pd.Series(np.asarray(groups)).fillna("").astype(str).to_numpy()
    n_top = max(1, int(round(len(y) * k_percent / 100.0)))
    top_idx = _order_desc(score, seed)[:n_top]

    selected = np.zeros(len(y), dtype=bool)
    selected[top_idx] = True

    df = pd.DataFrame({"group": g, "fires": y, "selected": selected})
    agg = (df.groupby("group")
             .agg(n_grids=("fires", "size"),
                  fires=("fires", "sum"),
                  selected=("selected", "sum"))
             .reset_index())
    agg["share_of_grids"] = agg["n_grids"] / len(y)
    agg["share_of_fires"] = agg["fires"] / max(y.sum(), 1e-9)
    agg["share_of_inspections"] = agg["selected"] / max(n_top, 1)
    # 1보다 크면 위험 비중보다 점검이 더 갔다는 뜻.
    agg["inspection_vs_risk"] = (agg["share_of_inspections"]
                                 / agg["share_of_fires"].replace(0, np.nan))
    agg["coverage_within_group"] = (agg["selected"] / agg["n_grids"])
    return agg.sort_values("share_of_fires", ascending=False).reset_index(drop=True)


def gini(values) -> float:
    """불균형 정도. 0 = 완전 균등, 1 = 한 곳에 몰림."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def equity_summary(eq: pd.DataFrame) -> dict:
    """형평성 한 줄 요약. 1.0 에 가까울수록 위험 비중대로 점검이 갔다는 뜻."""
    ratio = eq["inspection_vs_risk"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n_groups": int(len(eq)),
        "inspection_vs_risk_min": float(ratio.min()) if len(ratio) else float("nan"),
        "inspection_vs_risk_max": float(ratio.max()) if len(ratio) else float("nan"),
        "gini_of_inspections": gini(eq["share_of_inspections"]),
        "gini_of_fires": gini(eq["share_of_fires"]),
        "underserved": eq.loc[ratio.index[ratio < 0.7], "group"].tolist() if len(ratio) else [],
    }


# ---------------------------------------------------------------- 캘리브레이션

def calibration(y_true, pred) -> dict:
    """예측 '건수'가 실제 건수와 자릿수가 맞는가.

    순위만 맞고 값이 엉뚱하면 '이 격자는 연 3건 예상'이라는 말을 못 한다.
    운영에서 인력을 배분하려면 순위뿐 아니라 값도 맞아야 한다.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(pred, dtype=float), 1e-9, None)
    # Poisson deviance: 카운트 예측의 표준 손실
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / p), 0.0)
    dev = float(2.0 * np.sum(term - (y - p)) / max(len(y), 1))
    return {
        "total_actual": float(y.sum()),
        "total_predicted": float(p.sum()),
        "total_ratio": float(p.sum() / y.sum()) if y.sum() > 0 else float("nan"),
        "mean_poisson_deviance": dev,
        "mae": float(np.mean(np.abs(y - p))),
    }


def extend_with_standard_indices(result: dict, y_true, score, k_percents: list[float],
                                 seed: int = 42) -> dict:
    """evaluate_ranking 결과에 PAI/PEI 를 덧붙인다."""
    result["pai"] = {f"top{int(k)}": pai(y_true, score, k, seed) for k in k_percents}
    result["pai_max"] = {f"top{int(k)}": pai_max(y_true, k) for k in k_percents}
    result["pei"] = {f"top{int(k)}": pei(y_true, score, k, seed) for k in k_percents}
    return result
