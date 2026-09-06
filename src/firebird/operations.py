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


#: 점검 1건의 상대 소요. 특급 대상물 한 곳과 일반 근린생활 한 곳을 같은 1건으로
#: 세면 계획이 현장에서 깨진다. 소방시설이 많을수록 확인할 설비가 늘어난다.
#: (현장 실측치가 아니라 설비 구성에 따른 상대 가중치다 — 서별 실측이 있으면 교체하라.)
COST_WEIGHTS = {
    "fac_n_스프링클러": 2.5,      # 밸브·헤드·알람밸브 시험까지
    "fac_n_고층": 2.5,            # 제연설비·피난안전구역
    "fac_n_대형연면적": 2.0,      # 방화구획 관통부 전수 확인
    "fac_n_옥내소화전": 1.6,      # 방수압 측정
    "fac_n_자동화재탐지": 1.3,    # 수신기·감지기
    "fac_n_일반대상물": 1.0,
}
BIZ_COST = 0.8                    # 다중이용업소는 대상물보다 점검 항목이 적다


def inspection_cost(panel_year: pd.DataFrame, *, min_cost: float = 1.0,
                    weighted: bool = True) -> pd.Series:
    """격자 하나를 점검하는 데 드는 상대 소요(방문 건수 환산).

    설비 구성별 가중치를 쓴다. 가중치를 매길 컬럼이 없으면 단순 개수로 물러난다.
    자료가 아예 없으면 1로 둔다 — 적어도 한 번은 가야 한다.
    """
    cost = pd.Series(0.0, index=panel_year.index)
    used_weights = False

    if weighted:
        for col, w in COST_WEIGHTS.items():
            if col in panel_year.columns:
                cost = cost + panel_year[col].fillna(0).astype(float) * w
                used_weights = True

    if not used_weights and "target_total" in panel_year.columns:
        cost = cost + panel_year["target_total"].fillna(0).astype(float)
    elif not used_weights and "usage_total" in panel_year.columns:
        cost = cost + panel_year["usage_total"].fillna(0).astype(float)

    if "biz_total" in panel_year.columns:
        cost = cost + panel_year["biz_total"].fillna(0).astype(float) * (
            BIZ_COST if used_weights else 1.0)

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
    budget = float(capacity.total_visits)

    # 효율 순으로 훑되 '들어가면 담는다'. 누적합이 예산을 넘는 순간 뒤를 통째로
    # 버리면, 비싼 구역 하나 때문에 뒤의 싼 구역들을 못 담는다. 무차별 대입
    # 최적해와 맞대어 보니 그 방식은 평균 0.92, 최악 0.20 이었다.
    spent, take = 0.0, []
    for pos, c in enumerate(df["cost"].to_numpy(dtype=float)):
        if spent + c <= budget + 1e-9:
            take.append(pos)
            spent += c
    chosen = df.iloc[take].copy()

    # 예산에 들어가는 단일 최대 격자가 더 나으면 그것을 쓴다.
    # 이 한 줄이 탐욕의 최악을 최적의 1/2 아래로 떨어지지 않게 붙든다.
    fits = df[df["cost"] <= budget + 1e-9]
    if len(fits) and fits["expected_fires"].max() > chosen["expected_fires"].sum():
        chosen = fits.nlargest(1, "expected_fires").copy()

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
            "targets_if_all": float(topk.get("target_total", pd.Series(dtype=float)).sum()),
            "biz_if_all": float(topk.get("biz_total", pd.Series(dtype=float)).sum()),
            "cost_used": float(within["cost"].sum()),
            "expected_fires": float(within["expected_fires"].sum()),
        },
    }
    # '위험한 순서대로 예산이 떨어질 때까지' 갔을 때의 결과.
    # 발표에서 배분의 대비값으로 쓰는 숫자라, 손으로 역산하지 않고 여기서 낸다.
    #   whole   — 온전히 끝낸 구역만 인정. 1위 구역 소요가 예산보다 크면 0이다.
    #   partial — 마지막 구역을 간 만큼 비례로 쳐 준다. '쪼개서 가면?' 에 대한 답.
    order = df.sort_values("expected_fires", ascending=False)
    c = order["cost"].to_numpy(dtype=float)
    cum = c.cumsum()
    n_whole = int((cum <= capacity.total_visits + 1e-9).sum())
    spent = float(cum[n_whole - 1]) if n_whole else 0.0
    frac = ((capacity.total_visits - spent) / c[n_whole]
            if n_whole < len(c) and c[n_whole] > 0 else 0.0)
    out["risk_order"] = {
        "n_grids_whole": n_whole,
        "cost_used": spent,
        "next_grid_progress": float(frac),
        "top1_cost": float(c[0]) if len(c) else 0.0,
    }

    if actual_col:
        f = order[actual_col].to_numpy(dtype=float)
        whole_fires = float(f[:n_whole].sum())
        part_fires = float(f[n_whole]) * frac if n_whole < len(f) else 0.0
        out["risk_order"]["fires_whole"] = whole_fires
        out["risk_order"]["fires_partial"] = whole_fires + part_fires
        out["optimized"]["actual_fires_captured"] = float(alloc[actual_col].sum())
        out["top_k_percent"]["actual_fires_captured"] = float(within[actual_col].sum())
        total = float(df[actual_col].sum())
        if total > 0:
            out["optimized"]["actual_capture_rate"] = out["optimized"]["actual_fires_captured"] / total
            out["top_k_percent"]["actual_capture_rate"] = out["top_k_percent"]["actual_fires_captured"] / total
            out["gain_pp"] = ((out["optimized"]["actual_capture_rate"]
                               - out["top_k_percent"]["actual_capture_rate"]) * 100)
            out["risk_order"]["capture_rate_whole"] = (
                out["risk_order"]["fires_whole"] / total)
            out["risk_order"]["capture_rate_partial"] = (
                out["risk_order"]["fires_partial"] / total)
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


# ---------------------------------------------------------------- 형평성 제약

def capture_rate(allocation: pd.DataFrame, panel_year: pd.DataFrame,
                 *, label: str = "fires") -> float | None:
    """배분한 구역들이 그해 실제 화재의 몇 할을 품고 있었는가.

    화면·장표·보고서가 저마다 이 값을 다시 세다가, 표에는 형평성을 적용한
    배분이 뜨고 지표에는 효율만 적용한 값이 뜬 적이 있다. 세는 자리를 하나로
    둔다. 셀 수 없으면 0 으로 얼버무리지 않고 None 을 돌려준다 — 부를 쪽이
    무엇을 대신 쓸지 정하게 한다.
    """
    if label not in allocation.columns or label not in panel_year.columns:
        return None
    total = float(panel_year[label].fillna(0).sum())
    if total <= 0:
        return None
    return float(allocation[label].fillna(0).sum()) / total


def allocate_with_equity(panel_year: pd.DataFrame, risk, capacity: Capacity,
                         *, group_col: str = "sgg", min_share: float = 0.5,
                         cost: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """관할별 최소 배분을 보장하면서 배분한다.

    효율만 보고 담으면 특정 구에 점검이 쏠린다. 실제로 울산 배분에서 북구가
    화재 비중 대비 0.68 배만 배정됐다. 그걸 찾아 놓고 배분에서 무시하면
    '형평성'은 주장으로만 남는다.

    min_share: 각 관할이 최소한 '화재 비중 x min_share' 만큼의 예산은 받는다.
               1.0 이면 화재 비중에 정확히 비례, 0 이면 제약 없음(순수 효율).
    """
    df = panel_year.copy()
    df["expected_fires"] = np.asarray(risk, dtype=float)
    df["cost"] = np.asarray(cost if cost is not None else inspection_cost(df), dtype=float)
    df["efficiency"] = df["expected_fires"] / df["cost"].replace(0, np.nan)

    budget = float(capacity.total_visits)
    if group_col not in df.columns or min_share <= 0:
        alloc = allocate(df, df["expected_fires"], capacity, cost=df["cost"])
        return alloc, {"equity_constrained": False}

    groups = df[group_col].fillna("").astype(str)
    df[group_col] = groups
    valid = [g for g in groups.unique() if g.strip()]
    if len(valid) < 2:
        alloc = allocate(df, df["expected_fires"], capacity, cost=df["cost"])
        return alloc, {"equity_constrained": False}

    # 관할별 위험 비중 -> 최소 예산
    risk_share = (df.groupby(group_col)["expected_fires"].sum()
                  / max(df["expected_fires"].sum(), 1e-9))
    floors = {g: budget * float(risk_share.get(g, 0.0)) * min_share for g in valid}

    chosen_idx: list = []
    spent = 0.0
    # 1단계: 관할별 최소분을 각 관할 내 효율 순으로 채운다.
    #
    # 두 가지에 걸려 넘어지기 쉬운 자리다.
    #  (1) 첫 격자가 최소분보다 비싸면 그 관할이 통째로 배제된다.
    #      아무 데도 안 가는 관할이 생기면 그건 형평성이 아니라 배제다.
    #      -> 관할마다 최소 한 격자는 보장한다.
    #  (2) 최소분에 안 맞는 격자를 만났을 때 멈추면, 그 뒤의 **저렴한 격자를 못 본다**.
    #      울산 동구가 그랬다. 효율 순위 7번이 130건짜리라 거기서 멈췄고,
    #      뒤에 있던 한 자릿수 비용 격자들이 통째로 버려져 배분 비율이 0.25 였다.
    #      -> 멈추지 말고 건너뛰며 계속 담는다.
    for g in valid:
        sub = df[df[group_col] == g].sort_values(
            ["efficiency", "expected_fires"], ascending=False)
        used = 0.0
        for idx, row in sub.iterrows():
            if spent + row["cost"] > budget:
                continue
            if used > 0.0 and used + row["cost"] > floors[g]:
                continue
            chosen_idx.append(idx)
            used += row["cost"]
            spent += row["cost"]
            if used >= floors[g]:
                break

    # 2단계: 남은 예산을 전체 효율 순으로 채운다
    rest = df.drop(index=chosen_idx).sort_values(
        ["efficiency", "expected_fires"], ascending=False)
    for idx, row in rest.iterrows():
        if spent + row["cost"] > budget:
            continue
        chosen_idx.append(idx)
        spent += row["cost"]

    alloc = df.loc[chosen_idx].sort_values(
        ["efficiency", "expected_fires"], ascending=False).reset_index(drop=True)
    alloc.insert(0, "점검순서", range(1, len(alloc) + 1))
    alloc["누적비용"] = alloc["cost"].cumsum()
    alloc["누적기대화재"] = alloc["expected_fires"].cumsum()

    got = alloc.groupby(group_col)["cost"].sum()
    info = {
        "equity_constrained": True,
        "min_share": min_share,
        "budget": budget,
        "spent": float(spent),
        "by_group": {str(g): {"risk_share": float(risk_share.get(g, 0.0)),
                              "budget_share": float(got.get(g, 0.0) / max(spent, 1e-9)),
                              "floor": float(floors[g]),
                              "allocated": float(got.get(g, 0.0))}
                     for g in valid},
    }
    ratios = [v["budget_share"] / v["risk_share"]
              for v in info["by_group"].values() if v["risk_share"] > 0]
    if ratios:
        info["ratio_min"] = float(min(ratios))
        info["ratio_max"] = float(max(ratios))
    return alloc, info
