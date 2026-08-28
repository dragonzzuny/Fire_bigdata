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

#: 한 관서 관할만 그린다. 시 전체를 그리면 동선이 점처럼 작아져,
#: 정작 보여 줘야 할 '어디로 옮겨 갔는가'가 안 보인다.
#: 관서는 손으로 고르지 않는다 — 아래 규칙으로 정하고 그 이유를 함께 찍는다.
MIN_GRIDS = 100          # 이보다 작으면 관할이 좁아 비교가 무의미하다
K = 12                   # 관할 하나에서 도는 구역 수


def pick_station(cur: pd.DataFrame) -> tuple[str, str]:
    """야간 업소 순찰 이야기가 실제로 성립하는 관할을 고른다.

    다중이용업소가 가장 많은 관할이라야 '일반순찰과 야간순찰이 다른 곳으로
    간다'가 데이터로 참이다. 업소가 거의 없는 관할에서는 두 계획이 같아지고,
    그건 서비스가 아니라 표본의 성질이다.
    """
    g = (cur.groupby("station")
           .agg(격자=("grid_id", "nunique"), 업소=("biz_total", "sum"),
                화재=("fires", "sum"))
           .query("격자 >= @MIN_GRIDS")
           .sort_values("업소", ascending=False))
    if g.empty:
        return "", ""
    name = str(g.index[0])
    r = g.iloc[0]
    return name, (f"{name} 관할 — 다중이용업소 {r['업소']:,.0f}개소로 관내 최다"
                  f" (격자 {r['격자']:,.0f}개, {int(r['화재'])}건)")


def _hours(mode) -> str:
    """권장 시간대. 목적에 따라 시간까지 달라진다는 것이 이 그림의 절반이다."""
    h = getattr(mode, "hours", None)
    return f"   {h[0]:02d}–{h[1]:02d}시" if h else ""


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

    station, why = pick_station(cur)
    if station:
        cur = cur[cur["station"] == station].copy()
        print(f"  대상: {why}")

    stations = ST.station_table(cur, cfg, level="center",
                                city_label=cfg.city(city)["label"])
    a_mode, b_mode = PM.MODES[BEFORE], PM.MODES[AFTER]
    a_plan, a_tg = _plan(cur, cfg, a_mode, stations, K)
    b_plan, b_tg = _plan(cur, cfg, b_mode, stations, K)
    same = len(set(a_tg["grid_id"]) & set(b_tg["grid_id"]))

    fig = MV.route_compare(
        cur, a_plan["routes"], b_plan["routes"], cfg,
        before_title=f"{a_mode.label}{_hours(a_mode)}"
                     f"   ·   {len(a_tg)}개 구역 {a_plan['total_km']:.1f} km",
        after_title=f"{b_mode.label}{_hours(b_mode)}"
                    f"   ·   {len(b_tg)}개 구역 {b_plan['total_km']:.1f} km",
        pad_m=900.0)
    if fig is None:
        print("그릴 동선이 없습니다.")
        return 1
    out = cfg.paths.figures / "fig_route_compare.png"
    MV.save(fig, out)

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
