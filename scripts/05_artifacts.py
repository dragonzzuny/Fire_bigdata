#!/usr/bin/env python
"""현장 산출물 생성: 위험 우선순위표 · AI 점검계획서 · 소화전 사각지대 · 순찰 계획.

예측에서 끝내지 않고 현장에서 바로 쓰는 물건까지 만드는 단계다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from firebird import dataset as D, explain as X, features as F, hydrant as H, \
    llm as L, model as M, operations as OP, patrol as P, rules as R  # noqa: E402
from firebird.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("artifacts")


def risk_score_0_100(pred: np.ndarray) -> np.ndarray:
    """예측값을 0~100 점으로. 백분위 기반이라 도시가 달라도 뜻이 같다."""
    s = pd.Series(pred)
    return (s.rank(pct=True) * 100).to_numpy()


def load_model(cfg, city: str) -> M.TrainedModel:
    path = cfg.paths.outputs / f"model_{city}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. 먼저 scripts/04_train_eval.py 를 돌려라.")
    blob = joblib.load(path)
    return M.TrainedModel(estimator=blob["estimator"], feature_cols=blob["feature_cols"],
                          train_years=blob["train_years"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="ulsan")
    ap.add_argument("--year", type=int, default=None, help="기본: 홀드아웃 연도")
    ap.add_argument("--plans", type=int, default=10, help="점검계획서를 만들 상위 격자 수")
    ap.add_argument("--patrol-top", type=int, default=15, help="순찰 동선에 넣을 상위 격자 수")
    ap.add_argument("--no-llm", action="store_true", help="LLM 없이 규칙기반 초안만")
    ap.add_argument("--inspectors", type=int, default=4, help="점검 가능 인원")
    ap.add_argument("--per-day", type=int, default=8, help="1인 1일 점검 건수(대상물 기준)")
    ap.add_argument("--days", type=int, default=20, help="점검 기간(일)")
    args = ap.parse_args()

    cfg = load_config()
    panel = D.load_panel(cfg, args.city)
    year = args.year or cfg.holdout_year
    cur = panel[panel["year"] == year].copy()
    if cur.empty:
        raise SystemExit(f"{year}년 패널이 비었다")

    model = load_model(cfg, args.city)
    cur["pred"] = model.predict(cur)
    cur["risk_score"] = risk_score_0_100(cur["pred"])
    cur = cur.sort_values("pred", ascending=False).reset_index(drop=True)
    cur["rank"] = range(1, len(cur) + 1)
    cur["percentile"] = cur["rank"] / len(cur) * 100

    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 화면1: 위험 지도·우선순위표 ----------
    cols = [c for c in ["rank", "grid_id", "sgg", "lon", "lat", "risk_score", "pred",
                        "fires_lag1", "target_total", "biz_total", "n_hydrant",
                        "dist_hydrant_m", "fires"] if c in cur.columns]
    priority = cur[cols].rename(columns={
        "rank": "순위", "sgg": "시군구", "risk_score": "위험점수", "pred": "예측화재건수",
        "fires_lag1": "작년화재", "target_total": "대상물수", "biz_total": "업소수",
        "n_hydrant": "소화전수", "dist_hydrant_m": "최근접소화전_m", "fires": "실제화재"})
    priority.to_csv(out / f"priority_{args.city}_{year}.csv", index=False, encoding="utf-8-sig")
    print(f"[화면1] 위험 우선순위 {len(priority):,}격자 -> priority_{args.city}_{year}.csv")
    print(priority.head(10).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # ---------- 화면2: AI 점검계획서 ----------
    topn = cur.head(args.plans)
    drivers_tbl = X.explain_grids(model, topn).set_index("grid_id")["drivers"].to_dict()
    use_llm = (not args.no_llm) and L.is_available(cfg)
    print(f"\n[화면2] 점검계획서 {len(topn)}건 생성 "
          f"({'로컬 LLM' if use_llm else '규칙기반 초안 — Ollama 미가동'})")

    plans = []
    plan_dir = out / "inspection_plans"
    plan_dir.mkdir(exist_ok=True)
    for _, row in topn.iterrows():
        checklist = R.checklist_for_grid(row, cfg)
        risk = {"score_0_100": float(row["risk_score"]), "rank": int(row["rank"]),
                "percentile": float(row["percentile"])}
        plan = L.inspection_plan(cfg, str(row["grid_id"]), risk,
                                 drivers_tbl.get(row["grid_id"], []), checklist,
                                 use_llm=use_llm)
        plan["checklist"] = checklist
        plan["risk"] = risk
        plans.append(plan)
        (plan_dir / f"{args.city}_{year}_{row['grid_id']}.txt").write_text(
            plan["text"], encoding="utf-8")
    (out / f"inspection_plans_{args.city}_{year}.json").write_text(
        json.dumps(plans, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  -> {plan_dir}/ ({len(plans)}개 파일)")
    print("  예시:\n" + "\n".join("    " + l for l in plans[0]["text"].splitlines()[:12]))

    # ---------- 화면1b: 제약 하 점검 배분 ----------
    # '상위 20%'는 아무 데서도 나오지 않은 숫자다. 현장 제약은
    # '점검관 N명 × 1일 M건 × D일'로 생겼고, 격자마다 돌아야 할 집의 수도 다르다.
    cap = OP.Capacity(inspectors=args.inspectors, per_day=args.per_day, days=args.days)
    alloc_cmp = OP.compare_to_topk(cur, cur["pred"], cap, cfg.headline_k)
    print(f"\n[화면1b] 제약 하 점검 배분")
    print("  " + OP.format_allocation_report(alloc_cmp).replace("\n", "\n  "))
    alloc = OP.allocate(cur, cur["pred"], cap)
    alloc_cols = [c for c in ["점검순서", "grid_id", "sgg", "lon", "lat", "위험점수",
                              "expected_fires", "cost", "누적비용", "누적기대화재", "fires"]
                  if c in alloc.columns]
    alloc[alloc_cols].rename(columns={
        "grid_id": "격자", "sgg": "시군구", "expected_fires": "기대화재",
        "cost": "점검대상수", "fires": "실제화재"}).to_csv(
        out / f"allocation_{args.city}_{year}.csv", index=False, encoding="utf-8-sig")

    # ---------- 화면3: 순찰 시간대·요일·동선 ----------
    fires_path = cfg.paths.processed / f"fires_{args.city}.parquet"
    patrol_info: dict = {}
    if fires_path.exists():
        fires = pd.read_parquet(fires_path)
        hours = P.hour_profile(fires)
        wdays = P.weekday_profile(fires)
        peaks = P.peak_windows(fires, top_n=3)
        hours.to_csv(out / f"patrol_hours_{args.city}.csv", index=False, encoding="utf-8-sig")
        wdays.to_csv(out / f"patrol_weekdays_{args.city}.csv", index=False, encoding="utf-8-sig")
        patrol_info = {"peak_windows": peaks,
                       "top_weekday": wdays.sort_values("n", ascending=False)
                                           .head(2)[["요일", "n"]].to_dict("records")}
        print(f"\n[화면3] 화재 집중 시간대: " +
              ", ".join(f"{p['start_hour']:02d}~{p['end_hour']:02d}시({p['share']:.0%})"
                        for p in peaks))
    else:
        log.warning("fires parquet 없음 — 시간대 분석 생략")

    route = P.patrol_route(cur.head(args.patrol_top))
    route_cols = [c for c in ["순번", "grid_id", "sgg", "lon", "lat", "risk_score",
                              "이동거리_m", "누적거리_m"] if c in route.columns]
    route[route_cols].to_csv(out / f"patrol_route_{args.city}_{year}.csv",
                             index=False, encoding="utf-8-sig")
    if not route.empty:
        print(f"  순찰 동선 {len(route)}개 격자 · 총 이동 {route['누적거리_m'].iloc[-1]:,.0f}m "
              f"(직선거리 기준)")

    # ---------- 화면4: 대응취약·소화전 ----------
    cov = H.hydrant_coverage(cur)
    blind = H.blind_spots(cur, cur["pred"], cfg)
    surge = H.surge_alert(panel, year)
    if cov.get("available"):
        print(f"\n[화면4] 소화전 없는 격자 {cov['grids_without_hydrant']:,}/"
              f"{cov['n_grids']:,} ({cov['share_without_hydrant']:.1%})")
        print(f"  고위험(상위 {cfg['hydrant']['high_risk_percentile']}%) + 소화전 사각: "
              f"{len(blind)}개 격자")
        blind.to_csv(out / f"hydrant_blindspots_{args.city}_{year}.csv",
                     index=False, encoding="utf-8-sig")
    if not surge.empty:
        surge.to_csv(out / f"fire_surge_{args.city}_{year}.csv",
                     index=False, encoding="utf-8-sig")
        print(f"  화재 급증 경보 격자: {len(surge)}개")

    summary = {
        "city": args.city, "year": int(year), "n_grids": int(len(cur)),
        "allocation": alloc_cmp,
        "hydrant_coverage": cov,
        "n_blind_spots": int(len(blind)),
        "n_surge_alerts": int(len(surge)),
        "patrol": patrol_info,
        "plan_source": plans[0]["source"] if plans else None,
    }
    (out / f"artifacts_summary_{args.city}_{year}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n저장 완료 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
