#!/usr/bin/env python
"""순찰 조건 변경 전후 동선 지도 — 발표자료용.

화면을 캡처하지 않는다. 캡처는 스크롤 위치·창 크기에 따라 지도가 잘리고
배율도 그때그때 달라져, 같은 그림을 두 번 얻을 수 없다. 여기서는 계획을
직접 계산해 그림을 그리므로 언제 돌려도 같은 결과가 나온다.

두 지도는 같은 범위·같은 배율이다. 축척이 다르면 '동선이 짧아졌다'가
그림에서 거짓말이 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib                                     # noqa: E402
import pandas as pd                              # noqa: E402

from firebird import dataset as D                # noqa: E402
from firebird import mapviz as MV                # noqa: E402
from firebird import model as M                  # noqa: E402
from firebird import patrol_modes as PM          # noqa: E402
from firebird import routing as RT               # noqa: E402
from firebird import stations as ST              # noqa: E402
from firebird.config import load_config          # noqa: E402

#: 비교할 두 조건. 목적이 바뀌면 가는 곳도 시간대도 달라진다는 것을 보여 준다.
BEFORE, AFTER = "general", "night_business"


def _plan(cur, cfg, mode, stations, k):
    targets = PM.select_targets(cur, mode, k)
    targets = ST.assign_dispatch(targets, stations, level="center")
    plan = RT.plan_from_stations(targets, stations, level="center",
                                 use_road=True, budget_min=60.0)
    return plan, targets


def main() -> int:
    cfg = load_config()
    city = "ulsan"
    year = int(cfg.holdout_year)

    panel = D.load_panel(cfg, city)
    cur = panel[panel["year"] == year].copy()
    if cur.empty:
        print("해당 연도 자료가 없습니다.")
        return 1

    mp = cfg.paths.outputs / f"model_{city}.joblib"
    if not mp.exists():
        print("학습된 모델이 없습니다. scripts/04_train_eval.py 를 먼저 실행하십시오.")
        return 1
    b = joblib.load(mp)
    model = M.TrainedModel(estimator=b["estimator"], feature_cols=b["feature_cols"],
                           train_years=b["train_years"])
    cur["pred"] = model.predict(cur)

    stations = ST.station_table(cur, cfg, level="center",
                                city_label=cfg.city(city)["label"])
    a_mode, b_mode = PM.MODES[BEFORE], PM.MODES[AFTER]
    a_plan, a_tg = _plan(cur, cfg, a_mode, stations, a_mode.default_k)
    b_plan, b_tg = _plan(cur, cfg, b_mode, stations, b_mode.default_k)

    fig = MV.route_compare(
        cur, a_plan["routes"], b_plan["routes"], cfg,
        before_title=f"{a_mode.label}  ·  {len(a_tg)}개 구역 "
                     f"{a_plan['total_km']:.1f} km",
        after_title=f"{b_mode.label}  ·  {len(b_tg)}개 구역 "
                    f"{b_plan['total_km']:.1f} km")
    if fig is None:
        print("그릴 동선이 없습니다.")
        return 1
    out = cfg.paths.figures / "fig_route_compare.png"
    MV.save(fig, out)

    same = len(set(a_tg["grid_id"]) & set(b_tg["grid_id"]))
    print(f"  {a_mode.label}: {len(a_tg)}구역 {a_plan['total_km']:.1f} km "
          f"({a_plan['distance_source']})")
    print(f"  {b_mode.label}: {len(b_tg)}구역 {b_plan['total_km']:.1f} km "
          f"({b_plan['distance_source']})")
    print(f"  겹치는 구역 {same}개 · 총 이동 "
          f"{b_plan['total_km'] - a_plan['total_km']:+.1f} km")
    print(f"저장 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
