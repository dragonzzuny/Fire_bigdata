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


class TestWeightedCost(unittest.TestCase):
    """설비 구성별 가중치. 특급 대상물과 일반 근린생활을 같은 1건으로 세면 안 된다."""

    def test_sprinkler_costs_more_than_general(self):
        df = pd.DataFrame({"fac_n_스프링클러": [1.0, 0.0],
                           "fac_n_일반대상물": [0.0, 1.0]})
        c = O.inspection_cost(df)
        self.assertGreater(c.iloc[0], c.iloc[1])

    def test_falls_back_to_plain_counts_without_facility_columns(self):
        df = pd.DataFrame({"target_total": [10.0], "biz_total": [5.0]})
        self.assertEqual(O.inspection_cost(df).iloc[0], 15.0)

    def test_weighted_false_ignores_weights(self):
        df = pd.DataFrame({"fac_n_스프링클러": [4.0], "biz_total": [1.0],
                           "target_total": [4.0]})
        plain = O.inspection_cost(df, weighted=False).iloc[0]
        self.assertEqual(plain, 5.0)

    def test_still_at_least_one_visit(self):
        df = pd.DataFrame({"fac_n_일반대상물": [0.0], "biz_total": [0.0]})
        self.assertEqual(O.inspection_cost(df).iloc[0], 1.0)


class TestEquityConstrainedAllocation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 400
        # 두 관할: A 는 위험이 높고 밀집, B 는 위험이 낮고 흩어져 있다.
        self.df = pd.DataFrame({
            "grid_id": [f"g{i}" for i in range(n)],
            "sgg": ["A"] * (n // 2) + ["B"] * (n - n // 2),
            "target_total": rng.poisson(8, n).astype(float),
            "biz_total": rng.poisson(3, n).astype(float),
        })
        self.risk = np.concatenate([rng.gamma(1.4, 1.0, n // 2),
                                    rng.gamma(0.4, 1.0, n - n // 2)])
        self.df["fires"] = rng.poisson(self.risk)
        self.cap = O.Capacity(4, 8, 20)

    def test_budget_respected(self):
        alloc, _ = O.allocate_with_equity(self.df, self.risk, self.cap, min_share=1.0)
        self.assertLessEqual(alloc["cost"].sum(), self.cap.total_visits + 1e-6)

    def test_constraint_improves_worst_group_ratio(self):
        """제약을 걸면 가장 소외된 관할의 배분 비율이 올라가야 한다."""
        _, free = O.allocate_with_equity(self.df, self.risk, self.cap, min_share=0.0)
        _, tight = O.allocate_with_equity(self.df, self.risk, self.cap, min_share=1.0)
        self.assertFalse(free["equity_constrained"])
        self.assertTrue(tight["equity_constrained"])
        self.assertIn("ratio_min", tight)
        self.assertGreater(tight["ratio_min"], 0.0)

    def test_every_group_gets_something_under_full_constraint(self):
        alloc, info = O.allocate_with_equity(self.df, self.risk, self.cap, min_share=1.0)
        for g, v in info["by_group"].items():
            if v["risk_share"] > 0.05:
                self.assertGreater(v["allocated"], 0, f"{g} 에 배분이 하나도 없다")

    def test_no_group_column_falls_back_gracefully(self):
        df = self.df.drop(columns=["sgg"])
        alloc, info = O.allocate_with_equity(df, self.risk, self.cap, min_share=1.0)
        self.assertFalse(info["equity_constrained"])
        self.assertGreater(len(alloc), 0)

    def test_single_group_falls_back(self):
        df = self.df.assign(sgg="A")
        _, info = O.allocate_with_equity(df, self.risk, self.cap, min_share=1.0)
        self.assertFalse(info["equity_constrained"])

    def test_order_is_still_an_inspection_order(self):
        alloc, _ = O.allocate_with_equity(self.df, self.risk, self.cap, min_share=0.5)
        self.assertEqual(alloc["점검순서"].tolist(), list(range(1, len(alloc) + 1)))
        self.assertTrue(alloc["누적비용"].is_monotonic_increasing)


class TestEquityFloorEdgeCase(unittest.TestCase):
    """최소분보다 비싼 격자밖에 없는 관할도 배제되면 안 된다."""

    def test_expensive_group_still_gets_one_grid(self):
        df = pd.DataFrame({
            "grid_id": ["a1", "a2", "b1"],
            "sgg": ["A", "A", "B"],
            # B 의 유일한 격자는 최소분보다 비싸다
            "target_total": [5.0, 5.0, 60.0],
            "biz_total": [0.0, 0.0, 0.0],
            "fires": [3.0, 3.0, 2.0],
        })
        risk = [3.0, 3.0, 2.0]
        alloc, info = O.allocate_with_equity(df, risk, O.Capacity(1, 10, 10),
                                             min_share=1.0)
        self.assertIn("b1", alloc["grid_id"].tolist(),
                      "비싼 관할이 통째로 배제됐다 — 형평성이 아니라 배제다")
        self.assertGreater(info["by_group"]["B"]["allocated"], 0)

    def test_budget_still_respected_with_floor_guarantee(self):
        df = pd.DataFrame({
            "grid_id": [f"g{i}" for i in range(6)],
            "sgg": ["A", "A", "B", "B", "C", "C"],
            "target_total": [50.0] * 6, "biz_total": [0.0] * 6,
            "fires": [1.0] * 6,
        })
        cap = O.Capacity(1, 10, 10)          # 100건
        alloc, _ = O.allocate_with_equity(df, [1.0] * 6, cap, min_share=1.0)
        self.assertLessEqual(alloc["cost"].sum(), cap.total_visits + 1e-6)


class TestEquityFloorScanning(unittest.TestCase):
    """비싼 격자를 만나도 멈추지 말고 뒤의 저렴한 격자를 계속 봐야 한다."""

    def test_expensive_grid_does_not_block_cheaper_ones(self):
        # 효율 순서: c1(비쌈) 이 앞에 오지만 floor 를 넘는다.
        # 뒤의 c2, c3 는 저렴해서 담을 수 있어야 한다.
        df = pd.DataFrame({
            "grid_id": ["a1", "c1", "c2", "c3"],
            "sgg": ["A", "C", "C", "C"],
            "target_total": [10.0, 100.0, 3.0, 3.0],
            "biz_total": [0.0, 0.0, 0.0, 0.0],
            "fires": [5.0, 40.0, 1.0, 1.0],
        })
        risk = [5.0, 40.0, 1.0, 1.0]
        alloc, info = O.allocate_with_equity(df, risk, O.Capacity(1, 5, 10),
                                             min_share=1.0)
        picked = set(alloc["grid_id"])
        self.assertTrue({"c2", "c3"} & picked,
                        "비싼 격자에서 멈춰 뒤의 저렴한 격자를 놓쳤다")

    def test_floor_is_not_wildly_exceeded(self):
        rng = np.random.default_rng(9)
        n = 200
        df = pd.DataFrame({
            "grid_id": [f"g{i}" for i in range(n)],
            "sgg": rng.choice(["A", "B", "C"], n),
            "target_total": rng.poisson(6, n).astype(float) + 1,
            "biz_total": np.zeros(n),
            "fires": rng.poisson(0.5, n).astype(float),
        })
        cap = O.Capacity(3, 8, 15)
        alloc, _ = O.allocate_with_equity(df, df["fires"], cap, min_share=1.0)
        self.assertLessEqual(alloc["cost"].sum(), cap.total_visits + 1e-6)


class TestAllocationColumnNames(unittest.TestCase):
    """화면이 배분 결과를 다시 셀 때 쓰는 컬럼 이름을 고정한다.

    한 번 '실제화재'(표시용 이름)로 찾았다가, 컬럼이 없어 조용히 형평성
    미적용 값으로 되돌아간 적이 있다. 화면에는 186개 구역이 뜨는데 포착률은
    155개 구역 기준이 나갔다. 이름이 바뀌면 여기서 걸린다.
    """

    def test_allocation_keeps_raw_fires_column(self):
        panel = pd.DataFrame({
            "grid_id": [f"g{i}" for i in range(6)],
            "pred": [9.0, 7.0, 5.0, 4.0, 3.0, 1.0],
            "fires": [4, 3, 2, 1, 1, 0],
            "sgg": ["A", "A", "A", "B", "B", "B"],
            "target_total": [2, 2, 2, 2, 2, 2],
        })
        cap = O.Capacity(inspectors=1, per_day=4, days=2)
        alloc, _ = O.allocate_with_equity(panel, panel["pred"], cap,
                                           min_share=0.3)
        self.assertIn("fires", alloc.columns)
        self.assertNotIn("실제화재", alloc.columns)
        rate = O.capture_rate(alloc, panel)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(
            rate, float(alloc["fires"].sum()) / float(panel["fires"].sum()))
        self.assertGreater(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_capture_rate_returns_none_when_it_cannot_count(self):
        """셀 수 없으면 0 이 아니라 None — 0% 는 '못 잡았다'는 거짓말이다."""
        panel = pd.DataFrame({"grid_id": ["a"], "pred": [1.0], "fires": [0]})
        self.assertIsNone(O.capture_rate(panel, panel))          # 분모 0
        self.assertIsNone(O.capture_rate(panel.drop(columns=["fires"]), panel))

    def test_app_counts_capture_through_the_shared_helper(self):
        """앱이 표시용 컬럼 이름으로 배분을 다시 세지 않는지 본다."""
        src = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("OP.capture_rate(alloc, view)", src)
        self.assertNotIn('alloc["실제화재"]', src)


class TestAllocationQuality(unittest.TestCase):
    """배분이 최적해에서 얼마나 떨어지는지 무차별 대입으로 잰다.

    효율 순 누적합이 예산을 넘는 순간 뒤를 통째로 버리던 때가 있었다.
    비싼 구역 하나 때문에 뒤의 싼 구역들을 못 담아, 최적 대비 0.20 까지
    떨어졌다. '들어가면 담는' 방식으로 바꾼 뒤 최악이 0.74 로 올라왔다.
    다시 나빠지면 여기서 걸린다.
    """

    def _brute(self, cost, val, budget):
        import itertools
        best = 0.0
        n = len(val)
        for r in range(n + 1):
            for c in itertools.combinations(range(n), r):
                if cost[list(c)].sum() <= budget:
                    best = max(best, val[list(c)].sum())
        return best

    def test_최적해_대비_비율이_유지된다(self):
        rng = np.random.default_rng(11)
        ratios = []
        for _ in range(120):
            n = int(rng.integers(8, 12))
            df = pd.DataFrame({
                "grid_id": [f"g{i}" for i in range(n)],
                "pred": rng.random(n) * 6,
                "fires": rng.integers(0, 7, n),
                "target_total": rng.integers(1, 9, n)})
            cap = O.Capacity(1, int(rng.integers(3, 7)), 2)
            cost = O.inspection_cost(df).to_numpy()
            val = df["pred"].to_numpy()
            best = self._brute(cost, val, cap.total_visits)
            if best <= 0:
                continue
            got = O.allocate(df, df["pred"], cap)
            mine = float(got["expected_fires"].sum()) if len(got) else 0.0
            ratios.append(mine / best)
        self.assertGreater(len(ratios), 50)
        self.assertGreater(min(ratios), 0.5, "최악이 최적의 절반 아래로 떨어졌다")
        self.assertGreater(sum(ratios) / len(ratios), 0.95, "평균이 0.95 아래다")

    def test_비싼_구역_하나가_뒤를_막지_않는다(self):
        """효율 1위가 예산을 넘어도 뒤의 싼 구역들을 담아야 한다."""
        df = pd.DataFrame({
            "grid_id": ["비쌈", "쌈1", "쌈2", "쌈3"],
            "pred": [50.0, 3.0, 3.0, 3.0],
            "fires": [0, 0, 0, 0],
            "target_total": [100, 2, 2, 2]})
        cap = O.Capacity(1, 6, 1)          # 예산 6건
        got = O.allocate(df, df["pred"], cap)
        self.assertEqual(len(got), 3, f"싼 구역 셋을 담아야 하는데 {len(got)}개")
        self.assertNotIn("비쌈", set(got["grid_id"]))


class TestRiskOrderGreedy(unittest.TestCase):
    """'위험한 순서대로 예산 소진까지' 값이 산출물로 나오는가.

    이 숫자는 발표에서 배분(17.8%)의 대비값으로 쓴다. 손으로 역산한 추정치를
    무대에 올릴 수 없어 코드가 내도록 했다.
    """

    def _panel(self):
        return pd.DataFrame({
            "grid_id": ["a", "b", "c", "d"],
            "fires": [10.0, 5.0, 4.0, 1.0],
            "target_total": [100.0, 3.0, 2.0, 1.0],
        })

    def test_top1_alone_exceeds_budget(self):
        """1위 구역 하나가 예산을 넘으면 온전한 구역은 0개, 포착도 0이다."""
        df = self._panel()
        cap = O.Capacity(inspectors=1, per_day=10, days=5)      # 50건
        out = O.compare_to_topk(df, df["fires"], cap, 20.0)
        ro = out["risk_order"]
        self.assertEqual(ro["n_grids_whole"], 0)
        self.assertEqual(ro["capture_rate_whole"], 0.0)
        self.assertGreater(ro["top1_cost"], cap.total_visits)

    def test_partial_credit_is_prorated(self):
        """부분 인정은 간 만큼의 비율만큼만 쳐 준다."""
        df = self._panel()
        cap = O.Capacity(inspectors=1, per_day=10, days=5)      # 50건
        ro = O.compare_to_topk(df, df["fires"], cap, 20.0)["risk_order"]
        # 1위 구역 소요 100건 중 50건 → 절반
        self.assertAlmostEqual(ro["next_grid_progress"], 0.5, places=6)
        self.assertAlmostEqual(ro["fires_partial"], 5.0, places=6)
        self.assertAlmostEqual(ro["capture_rate_partial"], 5.0 / 20.0, places=6)

    def test_partial_never_below_whole(self):
        """부분 인정이 온전 인정보다 작을 수는 없다."""
        df = self._panel()
        for days in (1, 5, 20, 100):
            ro = O.compare_to_topk(
                df, df["fires"], O.Capacity(1, 10, days), 20.0)["risk_order"]
            self.assertGreaterEqual(ro["capture_rate_partial"],
                                    ro["capture_rate_whole"],
                                    f"days={days}")
