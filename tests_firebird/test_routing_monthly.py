"""도로 경로·팀 분할·순찰 유형·월 위험계수. 네트워크 없이 검증한다."""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from firebird import monthly as MO, patrol_modes as PM, routing as RT  # noqa: E402


def grid_df(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "grid_id": [f"g{i}" for i in range(n)],
        "lon": 129.30 + rng.random(n) * 0.15,
        "lat": 35.50 + rng.random(n) * 0.12,
        "sgg": rng.choice(["남구", "중구", "북구"], n),
        "pred": rng.gamma(1.0, 1.0, n),
    })


class TestDistance(unittest.TestCase):
    def test_haversine_is_symmetric_with_zero_diagonal(self):
        d = grid_df(6)
        M = RT.haversine_matrix(d["lon"], d["lat"])
        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(np.diag(M), 0))

    def test_known_distance_is_right_order_of_magnitude(self):
        """위도 1도는 약 111km."""
        M = RT.haversine_matrix([129.0, 129.0], [35.0, 36.0])
        self.assertAlmostEqual(M[0, 1] / 1000, 111, delta=2)

    def test_falls_back_to_straight_line_when_osrm_fails(self):
        d = grid_df(5)
        with mock.patch.object(RT, "_osrm_table", return_value=None):
            dist, dur, src = RT.road_distance_matrix(d["lon"], d["lat"])
        self.assertEqual(src, "straight_line_x_detour")
        straight = RT.haversine_matrix(d["lon"], d["lat"])
        self.assertTrue(np.allclose(dist, straight * RT.DETOUR_FACTOR))

    def test_use_road_false_skips_network(self):
        d = grid_df(5)
        with mock.patch.object(RT, "_osrm_table") as called:
            _, _, src = RT.road_distance_matrix(d["lon"], d["lat"], use_road=False)
        called.assert_not_called()
        self.assertEqual(src, "straight_line_x_detour")


class TestPartition(unittest.TestCase):
    def test_every_grid_assigned_exactly_once(self):
        d = grid_df(20)
        labels = RT.partition_teams(d, 4)
        self.assertEqual(len(labels), len(d))
        self.assertTrue(set(labels).issubset({0, 1, 2, 3}))

    def test_teams_are_balanced(self):
        """한 팀이 15격자, 다른 팀이 1격자인 계획은 쓸 수 없다."""
        d = grid_df(24, seed=3)
        labels = RT.partition_teams(d, 4)
        counts = pd.Series(labels).value_counts()
        self.assertLessEqual(counts.max() - counts.min(), 4)

    def test_single_team_returns_all_zeros(self):
        d = grid_df(8)
        self.assertTrue((RT.partition_teams(d, 1) == 0).all())

    def test_more_teams_than_grids_is_safe(self):
        d = grid_df(3)
        labels = RT.partition_teams(d, 10)
        self.assertEqual(len(labels), 3)


class TestRoute(unittest.TestCase):
    def test_route_visits_every_node_once(self):
        d = grid_df(9)
        M = RT.haversine_matrix(d["lon"], d["lat"])
        order = RT.solve_route(M)
        self.assertEqual(sorted(order), list(range(9)))

    def test_two_opt_does_not_lengthen(self):
        d = grid_df(10, seed=5)
        M = RT.haversine_matrix(d["lon"], d["lat"])
        naive = list(range(10))
        self.assertLessEqual(RT.route_length(M, RT.solve_route(M)),
                             RT.route_length(M, naive) + 1e-6)


class TestPlanPatrol(unittest.TestCase):
    def setUp(self):
        self.d = grid_df(15, seed=7)

    def plan(self, **kw):
        with mock.patch.object(RT, "_osrm_table", return_value=None):
            return RT.plan_patrol(self.d, **kw)

    def test_teams_do_not_overlap(self):
        """같은 격자를 두 팀이 가면 인력 낭비다."""
        plan = self.plan(n_teams=3)
        seen = [g for r in plan["routes"] for g in r["grid_id"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(self.d["grid_id"]))

    def test_summary_matches_routes(self):
        plan = self.plan(n_teams=3)
        self.assertEqual(len(plan["summary"]), len(plan["routes"]))
        for row, r in zip(plan["summary"].to_dict("records"), plan["routes"]):
            self.assertEqual(row["격자수"], len(r))

    def test_cumulative_distance_is_consistent(self):
        plan = self.plan(n_teams=2)
        for r in plan["routes"]:
            self.assertAlmostEqual(r["누적거리_m"].iloc[-1], r["이동거리_m"].sum(), places=3)

    def test_respect_groups_keeps_teams_within_one_sgg(self):
        """팀 수가 관할 수 이상이면 한 팀이 두 관할에 걸치지 않아야 한다."""
        plan = self.plan(n_teams=3, respect_groups=True)
        for r in plan["routes"]:
            self.assertEqual(r["sgg"].nunique(), 1,
                             f"팀이 여러 관할에 걸쳤다: {set(r['sgg'])}")

    def test_empty_input_is_safe(self):
        plan = RT.plan_patrol(pd.DataFrame(columns=["lon", "lat"]), 2)
        self.assertEqual(plan["routes"], [])


class TestPatrolModes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        n = 40
        self.df = pd.DataFrame({
            "grid_id": [f"g{i}" for i in range(n)],
            "sgg": rng.choice(["남구", "중구"], n),
            "pred": rng.gamma(1.0, 1.0, n),
            "biz_n_유흥주점": rng.poisson(1, n).astype(float),
            "usage_n_노유자": rng.poisson(0.5, n).astype(float),
            "usage_n_판매영업": rng.poisson(1, n).astype(float),
            "dist_hydrant_m": rng.gamma(2, 300, n),
        })

    def test_every_mode_produces_targets(self):
        for key, mode in PM.MODES.items():
            t = PM.select_targets(self.df, mode, 8)
            self.assertEqual(len(t), 8, f"{key} 가 대상을 못 골랐다")
            self.assertIn("순찰점수", t.columns)

    def test_modes_select_different_places(self):
        """목적이 다르면 가는 곳도 달라야 한다. 같다면 유형을 나눈 의미가 없다."""
        night = set(PM.select_targets(self.df, PM.MODES["night_business"], 8)["grid_id"])
        water = set(PM.select_targets(self.df, PM.MODES["water_supply"], 8)["grid_id"])
        self.assertLess(len(night & water), 8)

    def test_region_filter_restricts_targets(self):
        t = PM.select_targets(self.df, PM.MODES["general"], 10, sgg=["남구"])
        self.assertTrue((t["sgg"] == "남구").all())

    def test_night_mode_has_night_hours(self):
        self.assertEqual(PM.recommended_hours(PM.MODES["night_business"]), (22, 2))

    def test_hydrant_mode_prefers_far_from_hydrant(self):
        t = PM.select_targets(self.df, PM.MODES["water_supply"], 10)
        self.assertGreater(t["dist_hydrant_m"].mean(), self.df["dist_hydrant_m"].mean())


class TestMonthly(unittest.TestCase):
    def make_fires(self, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for year in range(2014, 2022):
            for month in range(1, 13):
                # 겨울에 많고 가을에 적게
                lam = 100 * (1.2 if month in (1, 12) else 0.85 if month in (9, 10) else 1.0)
                for _ in range(rng.poisson(lam)):
                    rows.append({"grid_id": "g1",
                                 "occurred_at": f"{year}{month:02d}15120000"})
        return pd.DataFrame(rows)

    def test_seasonal_index_centred_near_one(self):
        mf = MO.fires_by_month(self.make_fires(), 2014, 2021)
        s = MO.seasonal_index(mf)
        self.assertEqual(len(s), 12)
        self.assertAlmostEqual(s["seasonal_index"].mean(), 1.0, delta=0.05)

    def test_seasonal_index_finds_the_winter_peak(self):
        mf = MO.fires_by_month(self.make_fires(), 2014, 2021)
        s = MO.seasonal_index(mf).set_index("month")
        self.assertGreater(s.loc[1, "seasonal_index"], s.loc[9, "seasonal_index"])

    def test_year_totals_do_not_leak_into_seasonality(self):
        """연도별 총량 차이가 '그 달이 위험하다'로 잘못 읽히면 안 된다."""
        rows = []
        for year in range(2014, 2022):
            n_year = 200 if year < 2018 else 50      # 앞 4년이 4배 많다
            for month in range(1, 13):
                for _ in range(n_year):
                    rows.append({"grid_id": "g1",
                                 "occurred_at": f"{year}{month:02d}15120000"})
        mf = MO.fires_by_month(pd.DataFrame(rows), 2014, 2021)
        s = MO.seasonal_index(mf)
        self.assertLess(s["seasonal_index"].std(), 0.01)   # 모든 달이 같아야 한다

    def test_fit_without_weather_still_works(self):
        mf = MO.fires_by_month(self.make_fires(), 2014, 2021)
        fit = MO.fit_month_risk(mf, None)
        self.assertFalse(fit["uses_weather"])
        self.assertEqual(len(fit["seasonal"]), 12)

    def test_multiplier_defaults_to_one_without_data(self):
        self.assertEqual(MO.month_multiplier({}, 5), 1.0)

    def test_monthly_plan_has_twelve_rows(self):
        mf = MO.fires_by_month(self.make_fires(), 2014, 2021)
        fit = MO.fit_month_risk(mf, None)
        panel = pd.DataFrame({"pred": [1.0] * 10})
        plan = MO.monthly_plan(panel, "pred", fit)
        self.assertEqual(len(plan), 12)
        self.assertIn("위험계수", plan.columns)
        self.assertIn("등급", plan.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
