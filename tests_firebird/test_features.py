"""피처 생성의 산술을 손으로 검산한다. 데이터 없이 돌아간다."""
import sys, unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firebird import features as F  # noqa: E402


class TestFireHistory(unittest.TestCase):
    def setUp(self):
        # 격자 A(10_10)와 그 이웃 B(11_10), 그리고 멀리 떨어진 C(50_50).
        self.counts = pd.DataFrame([
            {"grid_id": "10_10", "year": 2018, "fires": 2},
            {"grid_id": "10_10", "year": 2019, "fires": 3},
            {"grid_id": "11_10", "year": 2018, "fires": 5},
            {"grid_id": "50_50", "year": 2020, "fires": 1},
        ])
        self.years = [2018, 2019, 2020]
        skel = F.build_panel_skeleton(
            pd.Series(["10_10", "11_10", "50_50"]), self.years)
        self.panel = F.add_fire_history(skel, self.counts, ring=1)

    def row(self, gid, year):
        r = self.panel[(self.panel.grid_id == gid) & (self.panel.year == year)]
        self.assertEqual(len(r), 1, f"{gid}/{year} 행이 하나여야 한다")
        return r.iloc[0]

    def test_zero_fire_grids_are_kept(self):
        """화재 0인 격자가 빠지면 상위 k% 포착률의 분모가 틀어진다."""
        self.assertEqual(len(self.panel), 3 * 3)

    def test_lag1_is_previous_year_only(self):
        self.assertEqual(self.row("10_10", 2018)["fires_lag1"], 0.0)   # 이전 연도 없음
        self.assertEqual(self.row("10_10", 2019)["fires_lag1"], 2.0)
        self.assertEqual(self.row("10_10", 2020)["fires_lag1"], 3.0)

    def test_cum_excludes_current_year(self):
        self.assertEqual(self.row("10_10", 2019)["fires_cum"], 2.0)     # 2018 만
        self.assertEqual(self.row("10_10", 2020)["fires_cum"], 5.0)     # 2018+2019
        self.assertEqual(self.row("10_10", 2020)["fires"], 0.0)

    def test_mean_prev(self):
        self.assertAlmostEqual(self.row("10_10", 2020)["fires_mean_prev"], 2.5)

    def test_neighbor_propagation(self):
        """A 의 2019년 이웃 lag1 은 B 의 2018년 화재 5건이어야 한다."""
        self.assertEqual(self.row("10_10", 2019)["neigh_fires_lag1"], 5.0)
        self.assertEqual(self.row("11_10", 2019)["neigh_fires_lag1"], 2.0)

    def test_distant_grid_gets_no_neighbor_signal(self):
        self.assertEqual(self.row("50_50", 2020)["neigh_fires_lag1"], 0.0)

    def test_leakage_guard_passes_on_clean_panel(self):
        F.assert_no_leakage(self.panel)

    def test_leakage_guard_catches_contamination(self):
        dirty = self.panel.copy()
        dirty.loc[dirty.index[0], "fires_cum"] += 1.0
        with self.assertRaises(AssertionError):
            F.assert_no_leakage(dirty)


class TestCategoryCounts(unittest.TestCase):
    def test_counts_and_shares(self):
        df = pd.DataFrame({
            "grid_id": ["10_10", "10_10", "10_10", "11_10"],
            "biz_type": ["일반음식점", "일반음식점", "노래연습장", "PC방"],
        })
        out = F.counts_by_category(df, "biz_type", "biz").set_index("grid_id")
        self.assertEqual(out.loc["10_10", "biz_n_일반음식점"], 2)
        self.assertEqual(out.loc["10_10", "biz_total"], 3)
        self.assertAlmostEqual(out.loc["10_10", "biz_share_일반음식점"], 2 / 3)
        self.assertEqual(out.loc["11_10", "biz_n_일반음식점"], 0)

    def test_unlisted_categories_fold_into_기타(self):
        df = pd.DataFrame({"grid_id": ["1_1", "1_1"], "t": ["일반음식점", "듣보업종"]})
        out = F.counts_by_category(df, "t", "biz", categories=["일반음식점"]).set_index("grid_id")
        self.assertEqual(out.loc["1_1", "biz_n_기타"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestFeatureSelection(unittest.TestCase):
    """좌표가 피처로 새어 들어가면 모델이 '위치'를 외운다. 그걸 막는다."""

    def test_coordinates_are_never_features(self):
        df = pd.DataFrame({
            "grid_id": ["1_1"], "year": [2020], "fires": [1.0],
            "gx": [1], "gy": [1], "gx_c": [1], "gy_c": [1],
            "lon": [129.3], "lat": [35.5], "cx": [1.0], "cy": [2.0],
            "fires_lag1": [0.0], "biz_total": [3.0],
        })
        feats = F.feature_columns(df)
        for banned in ("gx", "gy", "gx_c", "gy_c", "lon", "lat", "cx", "cy", "fires"):
            self.assertNotIn(banned, feats, f"{banned} 가 피처에 포함되면 안 된다")
        self.assertEqual(sorted(feats), ["biz_total", "fires_lag1"])

    def test_prediction_artifacts_are_not_features(self):
        df = pd.DataFrame({"fires": [1.0], "pred": [0.5], "risk_score": [90.0],
                           "rank": [1], "percentile": [2.0], "fires_lag1": [0.0]})
        self.assertEqual(F.feature_columns(df), ["fires_lag1"])


class TestFireSourceSelection(unittest.TestCase):
    """같은 연도가 두 원본 파일에 있으면 하나만 남아야 한다.

    라벨이 두 배가 되면 그 뒤의 모든 수치가 무의미해진다.
    """

    def setUp(self):
        from firebird import dataset as D
        self.D = D

    def test_overlapping_years_are_collapsed(self):
        fires = pd.DataFrame({
            "year": [2019, 2019, 2019, 2019],
            "_source_file": ["a_0000.csv", "a_0000.csv", "b_2021.csv", "b_2021.csv"],
            "has_time": [True, True, False, False],
        })
        kept, decisions = self.D.select_fire_source_per_year(fires)
        self.assertEqual(len(kept), 2)
        self.assertEqual(set(kept["_source_file"]), {"a_0000.csv"})   # 시각 있는 쪽
        self.assertIn(2019, decisions)
        self.assertEqual(decisions[2019]["picked"], "a_0000.csv")

    def test_non_overlapping_years_are_both_kept(self):
        """울산처럼 기간이 겹치지 않으면 두 파일 다 살아야 한다."""
        fires = pd.DataFrame({
            "year": [2020, 2020, 2021, 2021],
            "_source_file": ["a_0000.csv", "a_0000.csv", "b_2021.csv", "b_2021.csv"],
            "has_time": [True, True, False, False],
        })
        kept, decisions = self.D.select_fire_source_per_year(fires)
        self.assertEqual(len(kept), 4)
        self.assertEqual(decisions, {})

    def test_tie_on_time_rate_prefers_more_rows(self):
        fires = pd.DataFrame({
            "year": [2018] * 5,
            "_source_file": ["a.csv", "a.csv", "a.csv", "b.csv", "b.csv"],
            "has_time": [False] * 5,
        })
        kept, _ = self.D.select_fire_source_per_year(fires)
        self.assertEqual(set(kept["_source_file"]), {"a.csv"})
        self.assertEqual(len(kept), 3)
