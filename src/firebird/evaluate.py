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
