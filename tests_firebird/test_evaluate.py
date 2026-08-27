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


if __name__ == "__main__":
    unittest.main(verbosity=2)
