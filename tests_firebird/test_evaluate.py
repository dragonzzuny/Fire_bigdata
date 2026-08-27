"""평가 지표를 손으로 계산 가능한 예제로 검산한다."""
import sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firebird import evaluate as E  # noqa: E402


class TestCapture(unittest.TestCase):
    def test_perfect_ranking_captures_everything(self):
        y = [0]*90 + [1]*10
        score = list(range(100))          # 화재난 격자가 정확히 상위 10%
        self.assertAlmostEqual(E.capture_at_k(y, score, 10), 1.0)

    def test_worst_ranking_captures_nothing(self):
        y = [0]*90 + [1]*10
        score = list(range(100, 0, -1))   # 완전히 뒤집힌 순위
        self.assertAlmostEqual(E.capture_at_k(y, score, 10), 0.0)

    def test_random_ranking_is_near_k(self):
        rng = np.random.default_rng(0)
        y = rng.poisson(0.3, 20000)
        cap = E.capture_at_k(y, rng.random(20000), 20)
        self.assertAlmostEqual(cap, 0.20, delta=0.02)

    def test_lift_is_capture_over_k(self):
        y = [0]*90 + [1]*10
        self.assertAlmostEqual(E.lift_at_k(y, list(range(100)), 10), 10.0)

    def test_all_zero_labels_gives_nan_not_crash(self):
        self.assertNotEqual(E.capture_at_k([0, 0, 0], [1, 2, 3], 50),
                            E.capture_at_k([0, 0, 0], [1, 2, 3], 50))  # nan != nan

    def test_ties_are_broken_deterministically(self):
        y = [1]*50 + [0]*50
        flat = [7.0]*100
        a = E.capture_at_k(y, flat, 20, seed=1)
        b = E.capture_at_k(y, flat, 20, seed=1)
        self.assertEqual(a, b)

    def test_tie_breaking_is_not_input_order(self):
        """동점을 입력 순서로 깨면 베이스라인 점수가 부풀려진다."""
        y = [1]*20 + [0]*80
        flat = [0.0]*100
        cap = E.capture_at_k(y, flat, 20, seed=3)
        self.assertLess(cap, 0.9)   # 1.0 이 나오면 앞에서부터 집은 것


class TestDecile(unittest.TestCase):
    def test_top_grade_holds_the_most_fires(self):
        y = list(range(100))                      # 점수와 화재가 완전 일치
        d = E.decile_table(y, y, 10)
        self.assertEqual(d["grade"].tolist(), list(range(1, 11)))
        self.assertGreater(d[d.grade == 10]["mean_fires"].iloc[0],
                           d[d.grade == 1]["mean_fires"].iloc[0])
        self.assertTrue(E.is_monotonic(d))

    def test_monotonic_detects_broken_ordering(self):
        rng = np.random.default_rng(5)
        y = rng.poisson(0.3, 5000)
        d = E.decile_table(y, rng.random(5000), 10)   # 무작위 점수
        self.assertFalse(E.is_monotonic(d))


class TestComparison(unittest.TestCase):
    def test_headline_delta_sign(self):
        y = [0]*80 + [1]*20
        good = list(range(100))
        bad = list(range(100, 0, -1))
        r = E.compare_with_baseline(y, good, bad, [10, 20], headline_k=20)
        self.assertGreater(r["headline"]["delta_pp"], 0)
        self.assertIn("상위 20%", E.format_report(r))


class TestStandardIndices(unittest.TestCase):
    """PAI/PEI 는 손으로 계산 가능한 예제로 검산한다."""

    def test_pai_equals_lift(self):
        y = [0]*90 + [1]*10
        s = list(range(100))
        self.assertAlmostEqual(E.pai(y, s, 10), E.lift_at_k(y, s, 10))

    def test_pai_max_is_the_oracle(self):
        """화재가 10개 격자에 1건씩 있으면 상위 10%로 전부 잡는 게 최선."""
        y = [0]*90 + [1]*10
        self.assertAlmostEqual(E.pai_max(y, 10), 10.0)

    def test_pai_max_bounded_when_fires_are_spread(self):
        """화재가 모든 격자에 고르게 있으면 예언자도 상위 10%로 10%만 잡는다."""
        y = [1]*100
        self.assertAlmostEqual(E.pai_max(y, 10), 1.0)

    def test_pei_is_one_for_perfect_model(self):
        y = [0]*90 + [1]*10
        self.assertAlmostEqual(E.pei(y, list(range(100)), 10), 1.0)

    def test_pei_never_exceeds_one(self):
        rng = np.random.default_rng(3)
        y = rng.poisson(0.4, 3000)
        for k in (5, 10, 20):
            v = E.pei(y, y + rng.normal(0, 0.1, 3000), k)   # 거의 완벽한 점수
            self.assertLessEqual(v, 1.0 + 1e-9, f"top{k} 에서 PEI 가 1 을 넘었다")

    def test_pei_distinguishes_hard_city_from_bad_model(self):
        """화재가 퍼진 도시에서는 PAI 가 낮아도 PEI 는 높을 수 있다."""
        y = [1]*100                       # 완전히 균등 -> PAI 상한이 1
        good = list(range(100))
        self.assertAlmostEqual(E.pai(y, good, 20), 1.0, places=6)
        self.assertAlmostEqual(E.pei(y, good, 20), 1.0, places=6)


class TestEquity(unittest.TestCase):
    def test_inspection_matching_risk_gives_ratio_near_one(self):
        y = [1]*50 + [1]*50
        groups = ["A"]*50 + ["B"]*50
        score = ([1.0]*25 + [0.0]*25) + ([1.0]*25 + [0.0]*25)
        eq = E.equity(y, score, groups, 50)
        for r in eq["inspection_vs_risk"]:
            self.assertAlmostEqual(r, 1.0, delta=0.2)

    def test_detects_concentration_on_one_group(self):
        """A 에 화재가 절반인데 점검이 전부 A 로 가면 B 는 방치된다."""
        y = [1]*50 + [1]*50
        groups = ["A"]*50 + ["B"]*50
        score = [1.0]*50 + [0.0]*50
        eq = E.equity(y, score, groups, 50).set_index("group")
        self.assertAlmostEqual(eq.loc["A", "share_of_inspections"], 1.0)
        self.assertAlmostEqual(eq.loc["B", "share_of_inspections"], 0.0)
        summary = E.equity_summary(eq.reset_index())
        self.assertIn("B", summary["underserved"])

    def test_gini_bounds(self):
        self.assertAlmostEqual(E.gini([1, 1, 1, 1]), 0.0, places=6)
        self.assertGreater(E.gini([0, 0, 0, 4]), 0.6)


class TestCalibration(unittest.TestCase):
    def test_perfect_prediction_has_ratio_one(self):
        y = [0, 1, 2, 3]
        c = E.calibration(y, y)
        self.assertAlmostEqual(c["total_ratio"], 1.0)
        self.assertAlmostEqual(c["mae"], 0.0, places=6)

    def test_systematic_overprediction_is_visible(self):
        y = [1, 1, 1, 1]
        c = E.calibration(y, [2, 2, 2, 2])
        self.assertAlmostEqual(c["total_ratio"], 2.0)
        self.assertGreater(c["mean_poisson_deviance"], 0)


class TestRecapture(unittest.TestCase):
    def test_repeat_rate(self):
        prev = [1, 1, 0, 0]
        curr = [1, 0, 0, 1]
        r = E.recapture_rate(prev, curr, curr, 50)
        self.assertEqual(r["grids_with_prev_fire"], 2)
        self.assertAlmostEqual(r["repeat_rate"], 0.5)
        self.assertAlmostEqual(r["share_of_curr_fires_in_prev_grids"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
