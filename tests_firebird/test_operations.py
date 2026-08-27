"""운영 최적화: 제약 하 배분과 경로 개선."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from firebird import operations as O, patrol as P  # noqa: E402


class TestCapacity(unittest.TestCase):
    def test_total_visits(self):
        self.assertEqual(O.Capacity(4, 8, 20).total_visits, 640)

    def test_describe_mentions_every_factor(self):
        d = O.Capacity(2, 5, 10).describe()
        for token in ("2명", "5건", "10일", "100"):
            self.assertIn(token, d)


class TestCost(unittest.TestCase):
    def test_cost_is_number_of_places_to_visit(self):
        df = pd.DataFrame({"target_total": [10.0, 0.0], "biz_total": [5.0, 0.0]})
        c = O.inspection_cost(df)
        self.assertEqual(c.iloc[0], 15.0)

    def test_empty_grid_still_costs_one_visit(self):
        """대상물이 0이어도 적어도 한 번은 가야 한다 — 0 이면 무한히 담긴다."""
        df = pd.DataFrame({"target_total": [0.0], "biz_total": [0.0]})
        self.assertEqual(O.inspection_cost(df).iloc[0], 1.0)


class TestAllocation(unittest.TestCase):
    def setUp(self):
        # 격자 A: 기대 2건 / 비용 1  (효율 2.0)
        # 격자 B: 기대 9건 / 비용 9  (효율 1.0) — 위험은 최고지만 비싸다
        # 격자 C: 기대 1건 / 비용 1  (효율 1.0)
        self.df = pd.DataFrame({
            "grid_id": ["A", "B", "C"],
            "target_total": [1.0, 9.0, 1.0],
            "biz_total": [0.0, 0.0, 0.0],
            "fires": [2.0, 9.0, 1.0],
        })
        self.risk = [2.0, 9.0, 1.0]

    def test_budget_is_respected(self):
        cap = O.Capacity(1, 2, 1)          # 2건
        alloc = O.allocate(self.df, self.risk, cap)
        self.assertLessEqual(alloc["cost"].sum(), cap.total_visits)

    def test_efficiency_beats_raw_risk_under_tight_budget(self):
        """예산이 빠듯하면 '가장 위험한 격자'가 정답이 아니다."""
        cap = O.Capacity(1, 2, 1)          # 2건 — B(9건 필요)는 못 간다
        alloc = O.allocate(self.df, self.risk, cap)
        self.assertIn("A", alloc["grid_id"].tolist())
        self.assertNotIn("B", alloc["grid_id"].tolist())

    def test_large_budget_takes_everything(self):
        alloc = O.allocate(self.df, self.risk, O.Capacity(10, 10, 10))
        self.assertEqual(len(alloc), 3)

    def test_order_is_the_inspection_order(self):
        alloc = O.allocate(self.df, self.risk, O.Capacity(10, 10, 10))
        self.assertEqual(alloc["점검순서"].tolist(), [1, 2, 3])
        self.assertTrue(alloc["누적비용"].is_monotonic_increasing)


class TestComparisonWithTopK(unittest.TestCase):
    def test_optimized_never_uses_more_than_budget(self):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({"target_total": rng.poisson(6, n).astype(float),
                           "biz_total": rng.poisson(2, n).astype(float)})
        risk = rng.gamma(0.6, 1.0, n)
        df["fires"] = rng.poisson(risk)
        cap = O.Capacity(3, 8, 15)
        cmp = O.compare_to_topk(df, risk, cap, 20)
        self.assertLessEqual(cmp["optimized"]["cost_used"], cap.total_visits)
        self.assertLessEqual(cmp["top_k_percent"]["cost_used"], cap.total_visits)

    def test_report_flags_unaffordable_top_k(self):
        """'상위 20%' 가 예산을 넘으면 그 사실이 보고서에 드러나야 한다."""
        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame({"target_total": rng.poisson(20, n).astype(float),
                           "biz_total": rng.poisson(5, n).astype(float)})
        risk = rng.gamma(0.6, 1.0, n)
        df["fires"] = rng.poisson(risk)
        cmp = O.compare_to_topk(df, risk, O.Capacity(1, 5, 10), 20)   # 50건뿐
        text = O.format_allocation_report(cmp)
        self.assertIn("예산으로는", text)


class TestTwoOpt(unittest.TestCase):
    def path_len(self, pts, order):
        return sum(float(np.hypot(*(pts[order[i]] - pts[order[i - 1]])))
                   for i in range(1, len(order)))

    def test_two_opt_never_lengthens(self):
        rng = np.random.default_rng(7)
        for seed in range(5):
            pts = rng.random((12, 2)) * 1000
            order = list(range(12))
            self.assertLessEqual(self.path_len(pts, O.two_opt(pts, order)),
                                 self.path_len(pts, order) + 1e-6)

    def test_two_opt_fixes_a_crossing(self):
        """교차하는 경로는 반드시 짧아져야 한다."""
        pts = np.array([[0., 0.], [10., 10.], [0., 10.], [10., 0.]])
        order = [0, 1, 2, 3]
        self.assertLess(self.path_len(pts, O.two_opt(pts, order)),
                        self.path_len(pts, order))

    def test_patrol_route_uses_two_opt(self):
        rng = np.random.default_rng(5)
        n = 12
        df = pd.DataFrame({"grid_id": [f"g{i}" for i in range(n)],
                           "lon": 129.3 + rng.random(n) * 0.1,
                           "lat": 35.5 + rng.random(n) * 0.1})
        route = P.patrol_route(df)
        self.assertEqual(len(route), n)
        self.assertEqual(route["순번"].tolist(), list(range(1, n + 1)))
        # 누적거리가 각 구간 합과 일치해야 한다 (2-opt 후 재계산이 맞물렸는지)
        self.assertAlmostEqual(route["누적거리_m"].iloc[-1],
                               route["이동거리_m"].sum(), places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
