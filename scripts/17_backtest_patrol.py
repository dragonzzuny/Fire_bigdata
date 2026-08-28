#!/usr/bin/env python
"""회고 검증: 그해 이전 자료만으로 순찰 계획을 세웠다면 화재를 몇 건 만났는가.

'화재가 몇 건 줄어드느냐'는 질문에 답하려면 두 가지가 필요하다.
  (1) 순찰 구역 안에서 그해 실제로 몇 건이 났는가  ← 여기서 측정한다
  (2) 순찰이 화재를 몇 % 막는가                    ← 이 저장소로는 측정할 수 없다

(1)만 재고 (2)는 가정으로 남긴다. (2)를 아는 척하면 나머지 숫자도 같이
믿을 수 없게 된다. 대신 담당자가 자기 값을 넣을 수 있도록 시나리오 표를
함께 낸다.

비교 기준은 '작년에 화재가 많았던 순'(fires_lag1)이다. 지금 현장이 경험으로
하는 선택에 가장 가깝고, 별도 학습이 필요 없다.

알려진 한계: 대상물·업소 피처는 스냅숏이라 과거 연도 행에도 현재 값이 들어
있다(src/firebird/features.py 참조). 화재 라벨에는 누수가 없지만, 이 회고가
'2019년에 실제로 알 수 있었던 것'과 완전히 같지는 않다. 라벨 기준 회고이지
운영 시뮬레이션이 아니다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from firebird import dataset as D, evaluate as E, features as F, \
    model as M  # noqa: E402
from firebird.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("backtest")

#: 순찰 규모. 관서가 실제로 돌 수 있는 구역 수를 모르므로 한 점을 고르지 않고
#: 구간으로 낸다. 60 = 관서 6개 × 구역 10개를 상정한 운영점이다.
PATROL_SIZES = (15, 30, 60, 100)
OPERATING_POINT = 60

#: 순찰의 예방효과. 측정한 값이 아니라 담당자가 넣을 자리다.
EFFECT_SCENARIOS = (0.05, 0.10, 0.15)


def _rank(score) -> np.ndarray:
    """내림차순 정렬 인덱스. 동점은 고정 시드로 흩는다.

    비교 기준인 '작년 화재 순'은 대부분의 구역이 0이라 동점 덩어리가 크다.
    행 순서대로 자르면 앞쪽 구역이 공짜로 뽑혀 비교가 불공정해진다.
    평가기(evaluate._order_desc)와 같은 방식을 쓴다.
    """
    s = np.asarray(score, dtype=float)
    if np.isnan(s).all():
        raise ValueError("점수가 전부 NaN 이다 — 순위를 매길 수 없다")
    s = np.where(np.isnan(s), -np.inf, s)      # NaN 은 맨 뒤로, 말없이 앞에 두지 않는다
    return E._order_desc(s)


def _capture(test: pd.DataFrame, score, n: int) -> dict:
    """점수 상위 n개 구역이 그해 실제 화재를 몇 건 담고 있었는가."""
    picked = test.iloc[_rank(score)[:n]]
    return {"n_grids": int(len(picked)),
            "fires": float(picked["fires"].sum()),
            "grid_ids": [str(g) for g in picked["grid_id"].head(5)]}


def _paired_ci(test: pd.DataFrame, model_score, base_score, n: int, *,
               n_boot: int = 1000, seed: int = 42) -> dict:
    """'우리 계획 − 작년 화재 순' 차이의 부트스트랩 신뢰구간(화재 건수).

    점추정만 적어 두면 +39건이 우연인지 아닌지 말할 수 없다. 구역을 복원
    추출해 같은 절차를 다시 밟는다.
    """
    y = test["fires"].to_numpy(dtype=float)
    ms = np.asarray(model_score, dtype=float)
    bs = np.asarray(base_score, dtype=float)
    rng = np.random.default_rng(seed)
    rows = len(test)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, rows, rows)
        sub = pd.DataFrame({"fires": y[idx]})
        mo = _rank(ms[idx])[:n]
        bo = _rank(bs[idx])[:n]
        diffs[b] = sub["fires"].to_numpy()[mo].sum() - sub["fires"].to_numpy()[bo].sum()
    return {"lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "n_boot": int(n_boot)}


def backtest_year(panel: pd.DataFrame, feats: list[str], cfg, year: int) -> dict:
    """year 이전 자료만 학습해 year 를 예측하고, 순찰 규모별로 포착을 센다."""
    train_years = [int(y) for y in sorted(panel["year"].unique()) if y < year]
    test = panel[panel["year"] == year].reset_index(drop=True)
    if test.empty or not train_years:
        raise ValueError(f"{year}년 회고 검증 불가 (학습 {train_years}, 검증 {len(test)}행)")

    model = M.fit(panel, feats, cfg, train_years)
    pred = pd.Series(model.predict(test), index=test.index)
    prev = test[M.BASELINE_FEATURE].fillna(0)

    total_fires = float(test["fires"].sum())
    total_grids = int(len(test))
    rows = []
    for n in PATROL_SIZES:
        mine = _capture(test, pred, n)
        base = _capture(test, prev, n)
        rand = total_fires * n / total_grids
        rows.append({
            "patrol_grids": n,
            "share_of_city": n / total_grids,
            "model_fires": mine["fires"],
            "capture_share": (mine["fires"] / total_fires) if total_fires else 0.0,
            "baseline_fires": base["fires"],
            "random_fires": round(rand, 1),
            "gain_over_baseline": mine["fires"] - base["fires"],
            "lift_over_random": (mine["fires"] / rand) if rand else float("nan"),
            "gain_ci": _paired_ci(test, pred, prev, n),
            "top_grids": mine["grid_ids"],
        })

    return {"year": int(year), "train_years": train_years,
            "n_grids": total_grids, "total_fires": total_fires,
            "by_patrol_size": rows}


def scenario_table(row: dict) -> list[dict]:
    """포착 화재에 예방효과를 곱한다. 곱하는 값은 가정이라고 못 박는다."""
    out = []
    for e in EFFECT_SCENARIOS:
        out.append({
            "assumed_effect": e,
            "prevented_if_patrolled": round(row["model_fires"] * e, 1),
            "prevented_vs_current_practice": round(row["gain_over_baseline"] * e, 1),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="ulsan")
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021])
    args = ap.parse_args()

    cfg = load_config()
    panel = D.load_panel(cfg, args.city)
    feats = F.feature_columns(panel)
    F.assert_no_leakage(panel)

    report = {"city": args.city, "patrol_sizes": list(PATROL_SIZES),
              "operating_point": OPERATING_POINT,
              "effect_scenarios": list(EFFECT_SCENARIOS), "years": []}

    for year in args.years:
        r = backtest_year(panel, feats, cfg, year)
        report["years"].append(r)
        print(f"\n{'='*78}")
        print(f"[{year}년] ~{max(r['train_years'])}년까지만 학습 → {year}년 순찰 계획")
        print(f"  전체 {r['n_grids']:,}개 구역 · 그해 실제 화재 {r['total_fires']:,.0f}건")
        print(f"\n  {'순찰구역':>8} {'관내비중':>7} {'우리계획':>8} {'작년화재순':>9} "
              f"{'무작위':>7} {'차이':>7} {'배수':>6}")
        for row in r["by_patrol_size"]:
            mark = " ←" if row["patrol_grids"] == OPERATING_POINT else ""
            print(f"  {row['patrol_grids']:>7}개 {row['share_of_city']:>6.1%} "
                  f"{row['model_fires']:>7.0f}건 {row['baseline_fires']:>8.0f}건 "
                  f"{row['random_fires']:>6.1f}건 {row['gain_over_baseline']:>+6.0f}건 "
                  f"{row['lift_over_random']:>5.2f}배 ({row['capture_share']:.1%}){mark}")

    # 운영점에서의 시나리오. 발표에 쓰는 자리라 별도로 뽑아 둔다.
    head = report["years"][0]
    op_row = next(r for r in head["by_patrol_size"]
                  if r["patrol_grids"] == OPERATING_POINT)
    report["headline"] = {
        "year": head["year"], "train_upto": max(head["train_years"]),
        "patrol_grids": OPERATING_POINT,
        "share_of_city": op_row["share_of_city"],
        "total_fires": head["total_fires"],
        "model_fires": op_row["model_fires"],
        "capture_share": op_row["capture_share"],
        "baseline_fires": op_row["baseline_fires"],
        "gain_over_baseline": op_row["gain_over_baseline"],
        "gain_ci": op_row["gain_ci"],
        "lift_over_random": op_row["lift_over_random"],
        "scenarios": scenario_table(op_row),
    }

    h = report["headline"]
    print(f"\n{'='*78}\n[운영점] {h['train_upto']}년까지 학습 → {h['year']}년 "
          f"{h['patrol_grids']}개 구역 순찰")
    g = h["gain_ci"]
    print(f"  그 안에서 난 화재 {h['model_fires']:.0f}건 "
          f"= 그해 전체의 {h['capture_share']:.1%} "
          f"(작년화재순 {h['baseline_fires']:.0f}건, 차이 "
          f"{h['gain_over_baseline']:+.0f}건, 95% CI "
          f"{g['lo']:+.0f} ~ {g['hi']:+.0f}건)")
    if g["lo"] <= 0 <= g["hi"]:
        print("  ※ 차이의 신뢰구간이 0을 포함한다. '작년 화재 순보다 낫다'고"
              "\n     단정할 수 없다 — 포착 건수 자체를 근거로 말할 것.")
    print("\n  순찰이 화재를 e 만큼 막는다고 가정하면:")
    for s in h["scenarios"]:
        print(f"    e={s['assumed_effect']:.0%}  순찰구역 전체 기준 "
              f"{s['prevented_if_patrolled']:.1f}건 · "
              f"작년화재순 대비 추가로 {s['prevented_vs_current_practice']:.1f}건")
    print("\n  e 는 이 저장소에서 측정할 수 없다. 위 표는 가정이며,"
          "\n  측정한 것은 '순찰 구역 안에서 난 화재 건수'뿐이다.")

    out = cfg.paths.outputs / f"backtest_patrol_{args.city}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
