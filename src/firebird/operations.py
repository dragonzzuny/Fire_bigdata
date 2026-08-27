"""운영 최적화: 실제 제약 아래에서 어디를 점검할 것인가.

기획서는 '위험 상위 20%'를 쓴다. 그런데 20% 는 아무 데서도 나오지 않은 숫자다.
현장의 제약은 다르게 생겼다 — **점검 인력 N명, 1인당 1일 M건, 기간 D일.**
그 제약 아래에서 기대 화재 포착을 최대로 만드는 격자 조합이 진짜 답이다.

격자마다 점검 '비용'도 다르다. 대상물 40개소가 있는 격자와 3개소인 격자를
같은 한 칸으로 세면 계획이 현장에서 깨진다. 그래서 배낭문제로 푼다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Capacity:
    """점검 역량. 단위는 '대상물 1개소 점검'."""
    inspectors: int
    per_day: int
    days: int

    @property
    def total_visits(self) -> int:
        return int(self.inspectors * self.per_day * self.days)

    def describe(self) -> str:
        return (f"점검관 {self.inspectors}명 × 1일 {self.per_day}건 × {self.days}일 "
                f"= 총 {self.total_visits:,}건")


def inspection_cost(panel_year: pd.DataFrame, *, min_cost: float = 1.0) -> pd.Series:
    """격자 하나를 점검하는 데 드는 방문 건수.

    대상물 + 다중이용업소 수가 곧 돌아야 할 집의 수다.
    자료가 없으면 1로 둔다(적어도 한 번은 가야 한다).
    """
    cost = pd.Series(0.0, index=panel_year.index)
    for col in ("target_total", "biz_total"):
        if col in panel_year.columns:
            cost = cost + panel_year[col].fillna(0).astype(float)
    if (cost <= 0).all():
        cost = pd.Series(min_cost, index=panel_year.index)
    return cost.clip(lower=min_cost)


def allocate(panel_year: pd.DataFrame, risk, capacity: Capacity,
             *, cost: pd.Series | None = None) -> pd.DataFrame:
    """제약 아래 기대 화재 포착 최대화 (분수 배낭 + 정수 보정).

    각 격자의 '효율' = 기대 화재 / 점검 비용 순으로 담는다.
    분수 배낭의 탐욕 해는 정수 배낭의 최적해에 가장 비싼 한 칸 이내로 붙는다 —
    격자가 수천 개인 이 문제에서는 실질적으로 최적이다.

    반환: 선택된 격자 + 누적 비용/기대포착. 선택 순서가 곧 점검 순서다.
    """
    df = panel_year.copy()
    df["expected_fires"] = np.asarray(risk, dtype=float)
    df["cost"] = np.asarray(cost if cost is not None else inspection_cost(df), dtype=float)
    df["efficiency"] = df["expected_fires"] / df["cost"].replace(0, np.nan)

    df = df.sort_values(["efficiency", "expected_fires"], ascending=False)
    budget = capacity.total_visits
    cum = df["cost"].cumsum()
    df["cum_cost"] = cum
    chosen = df[cum <= budget].copy()

    # 남은 예산으로 다음 격자를 넣을 수 있으면 넣는다.
    rest = df[cum > budget]
    if not rest.empty:
        left = budget - (chosen["cost"].sum() if len(chosen) else 0.0)
        nxt = rest.iloc[0]
        if left >= nxt["cost"]:
            chosen = pd.concat([chosen, rest.iloc[[0]]])

    chosen = chosen.reset_index(drop=True)
    chosen.insert(0, "점검순서", range(1, len(chosen) + 1))
    chosen["누적비용"] = chosen["cost"].cumsum()
    chosen["누적기대화재"] = chosen["expected_fires"].cumsum()
    return chosen


def compare_to_topk(panel_year: pd.DataFrame, risk, capacity: Capacity,
                    k_percent: float = 20.0) -> dict:
    """'상위 k%' 방식과 '제약 하 최적 배분'을 같은 예산으로 비교한다.

    같은 인력으로 몇 건을 더 잡는지가 이 모듈이 존재하는 이유다.
    """
    df = panel_year.copy()
    df["expected_fires"] = np.asarray(risk, dtype=float)
    df["cost"] = inspection_cost(df)

    alloc = allocate(df, df["expected_fires"], capacity, cost=df["cost"])

    # 상위 k%: 위험 순으로 자르되, 예산을 넘으면 거기서 멈춘다(현실 제약).
    topk = df.sort_values("expected_fires", ascending=False)
    n_top = max(1, int(round(len(df) * k_percent / 100.0)))
    topk = topk.head(n_top)
    within = topk[topk["cost"].cumsum() <= capacity.total_visits]

    actual_col = "fires" if "fires" in df.columns else None
    out = {
        "capacity": capacity.describe(),
        "budget_visits": capacity.total_visits,
        "optimized": {
            "n_grids": int(len(alloc)),
            "cost_used": float(alloc["cost"].sum()),
            "expected_fires": float(alloc["expected_fires"].sum()),
        },
        "top_k_percent": {
            "k": k_percent,
            "n_grids_selected": int(len(topk)),
            "n_grids_affordable": int(len(within)),
            "cost_if_all": float(topk["cost"].sum()),
            "cost_used": float(within["cost"].sum()),
            "expected_fires": float(within["expected_fires"].sum()),
        },
    }
    if actual_col:
        out["optimized"]["actual_fires_captured"] = float(alloc[actual_col].sum())
        out["top_k_percent"]["actual_fires_captured"] = float(within[actual_col].sum())
        total = float(df[actual_col].sum())
        if total > 0:
            out["optimized"]["actual_capture_rate"] = out["optimized"]["actual_fires_captured"] / total
            out["top_k_percent"]["actual_capture_rate"] = out["top_k_percent"]["actual_fires_captured"] / total
            out["gain_pp"] = ((out["optimized"]["actual_capture_rate"]
                               - out["top_k_percent"]["actual_capture_rate"]) * 100)
    return out


def format_allocation_report(cmp: dict) -> str:
    """사람이 읽는 비교 요약."""
    o, t = cmp["optimized"], cmp["top_k_percent"]
    lines = [
        f"역량: {cmp['capacity']}",
        "",
        f"{'방식':<22}{'격자':>7}{'사용예산':>10}{'기대화재':>10}",
        f"{'제약 하 최적 배분':<20}{o['n_grids']:>8,}{o['cost_used']:>10,.0f}{o['expected_fires']:>10.1f}",
        f"{f'상위 {t[chr(107)]:.0f}% 방식':<20}{t['n_grids_affordable']:>8,}"
        f"{t['cost_used']:>10,.0f}{t['expected_fires']:>10.1f}",
    ]
    if t["n_grids_affordable"] < t["n_grids_selected"]:
        lines.append(f"  * 상위 {t['k']:.0f}% 는 {t['n_grids_selected']:,}격자를 지목하지만 "
                     f"예산으로는 {t['n_grids_affordable']:,}격자까지만 갈 수 있다 "
                     f"(필요 {t['cost_if_all']:,.0f}건 vs 가용 {cmp['budget_visits']:,}건)")
    if "gain_pp" in cmp:
        lines += ["",
                  f"같은 인력으로 실제 화재 포착: "
                  f"최적 {o['actual_capture_rate']:.1%} vs 상위% {t['actual_capture_rate']:.1%} "
                  f"({cmp['gain_pp']:+.1f}%p)"]
    return "\n".join(lines)


# ---------------------------------------------------------------- 순찰 경로 개선

def two_opt(points: np.ndarray, order: list[int], *, max_rounds: int = 40) -> list[int]:
    """2-opt 개선. 최근접 이웃 경로의 교차를 풀어 거리를 줄인다.

    도로망을 쓰지 않으므로 여전히 직선거리 기준이지만, 최근접 이웃만
    쓰면 마지막에 먼 격자로 되돌아가는 구간이 남는다. 그걸 없앤다.
    """
    def dist(a: int, b: int) -> float:
        return float(np.hypot(*(points[a] - points[b])))

    best = list(order)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b, c, d = best[i - 1], best[i], best[j], best[j + 1]
                if dist(a, b) + dist(c, d) > dist(a, c) + dist(b, d) + 1e-9:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
    return best
