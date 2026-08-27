"""알고리즘 정확성 감사.

이 파일은 '돌아가는가'가 아니라 '맞는가'를 본다.
휴리스틱은 정답을 아는 작은 문제에서 완전탐색과 맞춰 보고,
지표는 손으로 계산한 값과 맞춰 본다. 성질(순열인가, 예산을 넘지 않는가,
개선 단계에서 값이 나빠지지 않는가)은 무작위 입력 수십 개로 확인한다.

여기서 실패하면 화면에 뜨는 숫자를 믿을 수 없다.
"""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firebird import evaluate as E, monthly as MO, operations as OP, routing as RT  # noqa: E402
from firebird import patrol_modes as PM  # noqa: E402


def _rand_points(rng, n, lon0=129.3, lat0=35.55, spread=0.08):
    return (lon0 + rng.normal(0, spread, n), lat0 + rng.normal(0, spread, n))


# ==================================================================== 거리

class TestDistance(unittest.TestCase):

    def test_haversine_알려진_거리(self):
        """서울시청 ↔ 부산시청 대권거리는 약 325 km 로 알려져 있다."""
        d = RT.haversine_matrix([126.9780, 129.0756], [37.5665, 35.1796])
        self.assertAlmostEqual(d[0, 1] / 1000.0, 325.0, delta=3.0)

    def test_haversine_성질(self):
        """대칭이고, 대각선은 0이며, 삼각부등식을 만족해야 한다."""
        rng = np.random.default_rng(0)
        lon, lat = _rand_points(rng, 12)
        d = RT.haversine_matrix(lon, lat)
        np.testing.assert_allclose(d, d.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-6)
        for i, j, k in itertools.permutations(range(6), 3):
            self.assertLessEqual(d[i, j], d[i, k] + d[k, j] + 1e-6)

    def test_우회계수_적용(self):
        """도로망을 못 쓸 때는 직선거리에 우회계수를 곱한 값이어야 한다."""
        rng = np.random.default_rng(1)
        lon, lat = _rand_points(rng, 8)
        straight = RT.haversine_matrix(lon, lat)
        d, t, src = RT.road_distance_matrix(lon, lat, use_road=False)
        self.assertEqual(src, "straight_line_x_detour")
        np.testing.assert_allclose(d, straight * RT.DETOUR_FACTOR, rtol=1e-9)
        self.assertTrue((t[straight > 0] > 0).all())


# ==================================================================== 경로

def _brute_force_route(dist, start=0, closed=False):
    """출발점 고정 최단 경로를 완전탐색으로 구한다. closed=True 면 왕복."""
    n = len(dist)
    others = [i for i in range(n) if i != start]
    best, best_len = None, float("inf")
    for perm in itertools.permutations(others):
        order = [start] + list(perm)
        L = sum(dist[order[i - 1], order[i]] for i in range(1, n))
        if closed:
            L += dist[order[-1], start]
        if L < best_len:
            best, best_len = order, L
    return best, best_len


class TestRoute(unittest.TestCase):
    """순찰 동선.

    실제 순찰은 **왕복**이다(관서에서 나가 관서로 복귀). 편도만 최적화하고
    왕복 거리를 보고하면 목적함수와 보고값이 어긋난다. 그래서 왕복(closed=True)이
    본 검사 대상이고, 편도는 보조로 함께 본다.
    """

    def test_왕복_완전탐색과_비교(self):
        """지점 8개까지는 완전탐색으로 정답을 안다. 휴리스틱이 그 정답에 닿는가."""
        gaps = []
        for s in range(30):
            rng = np.random.default_rng(100 + s)
            n = int(rng.integers(4, 9))
            lon, lat = _rand_points(rng, n)
            dist = RT.haversine_matrix(lon, lat)
            _, opt = _brute_force_route(dist, 0, closed=True)
            got = RT.solve_route(dist, start=0, restarts=4, closed=True)
            self.assertEqual(got[0], 0, "출발점(관서)이 맨 앞이어야 한다")
            self.assertEqual(sorted(got), list(range(n)), "모든 지점을 한 번씩")
            L = RT.route_length(dist, got, closed=True)
            gaps.append((L - opt) / max(opt, 1e-9))
            self.assertLessEqual(L, opt + 1e-6,
                                 f"seed={s}: 휴리스틱 {L:.1f} > 최적 {opt:.1f}")
        self.assertAlmostEqual(max(gaps), 0.0, delta=1e-6)

    def test_왕복_중간규모(self):
        """지점 9개(완전탐색 40,320가지)에서도 최적해와 붙어 있는가."""
        worst = 0.0
        for s in range(8):
            rng = np.random.default_rng(300 + s)
            lon, lat = _rand_points(rng, 9)
            dist = RT.haversine_matrix(lon, lat)
            _, opt = _brute_force_route(dist, 0, closed=True)
            L = RT.route_length(dist, RT.solve_route(dist, 0, restarts=6, closed=True),
                                closed=True)
            worst = max(worst, (L - opt) / opt)
        self.assertLessEqual(worst, 0.01, f"최적 대비 최악 {worst:.2%}")

    def test_편도_완전탐색과_비교(self):
        """편도(관서 복귀 없음)도 최적해 근처여야 한다."""
        worst = 0.0
        for s in range(30):
            rng = np.random.default_rng(400 + s)
            n = int(rng.integers(4, 9))
            lon, lat = _rand_points(rng, n)
            dist = RT.haversine_matrix(lon, lat)
            _, opt = _brute_force_route(dist, 0)
            L = RT.route_length(dist, RT.solve_route(dist, 0, restarts=4))
            worst = max(worst, (L - opt) / opt)
        self.assertLessEqual(worst, 0.05, f"편도 최적 대비 최악 {worst:.2%}")

    def test_꼬리_뒤집기_수를_놓치지_않는다(self):
        """열린 경로에서 마지막 구간을 통째로 뒤집어야 짧아지는 배치가 있다.
        이 수를 빼면 완전탐색 대비 20% 넘게 벌어지는 사례가 실제로 나왔다."""
        rng = np.random.default_rng(305)
        lon, lat = _rand_points(rng, 9)
        dist = RT.haversine_matrix(lon, lat)
        _, opt = _brute_force_route(dist, 0)
        L = RT.route_length(dist, RT.solve_route(dist, 0, restarts=6))
        self.assertLessEqual((L - opt) / opt, 0.01)

    def test_개선단계는_길이를_늘리지_않는다(self):
        """2-opt·Or-opt 는 단조 개선이어야 한다. 늘어나면 구현이 틀린 것이다."""
        for closed in (False, True):
            for s in range(25):
                rng = np.random.default_rng(500 + s)
                n = int(rng.integers(5, 16))
                lon, lat = _rand_points(rng, n)
                dist = RT.haversine_matrix(lon, lat)
                start = RT.nearest_neighbor(dist, 0)
                before = RT.route_length(dist, start, closed=closed)
                after2 = RT._two_opt_matrix(dist, list(start), closed=closed)
                l2 = RT.route_length(dist, after2, closed=closed)
                self.assertLessEqual(l2, before + 1e-6)
                afteror = RT._or_opt_matrix(dist, list(after2), closed=closed)
                self.assertLessEqual(RT.route_length(dist, afteror, closed=closed),
                                     l2 + 1e-6)
                self.assertEqual(afteror[0], 0)
                self.assertEqual(sorted(afteror), list(range(n)))

    def test_2opt만으로는_못_고치는_것을_oropt가_고친다(self):
        """둘을 함께 쓸 근거. 하나라도 개선 사례가 있어야 한다."""
        improved = 0
        for s in range(40):
            rng = np.random.default_rng(700 + s)
            n = int(rng.integers(7, 13))
            lon, lat = _rand_points(rng, n)
            dist = RT.haversine_matrix(lon, lat)
            nn = RT.nearest_neighbor(dist, 0)
            only2 = RT._improve(dist, nn, or_opt=False, closed=True)
            both = RT._improve(dist, nn, or_opt=True, closed=True)
            if (RT.route_length(dist, both, closed=True)
                    < RT.route_length(dist, only2, closed=True) - 1e-6):
                improved += 1
        self.assertGreater(improved, 0, "Or-opt 가 한 번도 기여하지 않으면 뺄 이유가 있다")

    def test_비대칭_거리행렬도_다룬다(self):
        """도로거리는 일방통행 때문에 A→B 와 B→A 가 다르다."""
        worst = 0.0
        for s in range(10):
            rng = np.random.default_rng(11 + s)
            n = 7
            d = rng.uniform(500, 5000, (n, n))
            np.fill_diagonal(d, 0.0)
            got = RT.solve_route(d, start=0, restarts=4, closed=True)
            self.assertEqual(sorted(got), list(range(n)))
            _, opt = _brute_force_route(d, 0, closed=True)
            worst = max(worst, (RT.route_length(d, got, closed=True) - opt) / opt)
        self.assertLessEqual(worst, 0.05, f"비대칭 최악 {worst:.2%}")


# ==================================================================== 분할

class TestPartition(unittest.TestCase):

    def test_모든_격자가_정확히_한_팀에(self):
        for s in range(20):
            rng = np.random.default_rng(900 + s)
            n = int(rng.integers(6, 60))
            k = int(rng.integers(2, 7))
            lon, lat = _rand_points(rng, n)
            df = pd.DataFrame({"lon": lon, "lat": lat})
            lab = RT.partition_teams(df, k)
            self.assertEqual(len(lab), n)
            self.assertTrue((lab >= 0).all())
            self.assertLessEqual(len(set(lab)), min(k, n))
            # 겹침 없음: 라벨이 하나씩만 붙는다(배열 구조상 자명하나 명시적으로 본다)
            self.assertEqual(sum((lab == c).sum() for c in set(lab)), n)

    def test_한_팀에_몰리지_않는다(self):
        """그냥 k-means 를 쓰면 30 대 3 이 나온다. 용량 제한이 실제로 걸리는가."""
        rng = np.random.default_rng(3)
        n, k = 48, 4
        lon = np.r_[rng.normal(129.30, 0.005, 40), rng.normal(129.45, 0.02, 8)]
        lat = np.r_[rng.normal(35.54, 0.005, 40), rng.normal(35.62, 0.02, 8)]
        lab = RT.partition_teams(pd.DataFrame({"lon": lon, "lat": lat}), k)
        counts = np.bincount(lab, minlength=k)
        self.assertLessEqual(counts.max(), np.ceil(n / k * 1.15) + 1,
                             f"팀별 격자 수 {counts.tolist()} — 용량 제한이 안 걸렸다")
        self.assertTrue((counts > 0).all(), f"빈 팀 발생: {counts.tolist()}")


# ==================================================================== 회차 분할

class TestBudgetSplit(unittest.TestCase):

    def test_모든_회차가_시간을_지키고_빠짐없이_돈다(self):
        for s in range(25):
            rng = np.random.default_rng(1100 + s)
            n = int(rng.integers(4, 20))
            lon, lat = _rand_points(rng, n)
            dist, dur, _ = RT.road_distance_matrix(lon, lat, use_road=False)
            order = RT.solve_route(dist, 0, restarts=2)
            budget, stay = 60.0, 5.0
            trips = RT.split_by_budget(order, dur, dist, budget, stay_min=stay)

            visited = [x for t in trips for x in t[1:]]
            self.assertEqual(sorted(visited), sorted(order[1:]),
                             "회차를 합치면 원래 방문지와 같아야 한다")
            self.assertEqual(len(visited), len(set(visited)), "같은 격자를 두 번 가면 안 된다")
            for t in trips:
                self.assertEqual(t[0], order[0], "회차는 관서에서 출발한다")
                if len(t) <= 2:
                    continue          # 한 곳만 가는데도 넘으면 쪼갤 방법이 없다
                total = sum(dur[t[i - 1], t[i]] / 60.0 + stay for i in range(1, len(t)))
                total += dur[t[-1], t[0]] / 60.0
                self.assertLessEqual(total, budget * 1.5,
                                     f"seed={s} 회차 {total:.0f}분 (예산 {budget:.0f}분)")

    def test_예산이_없으면_쪼개지_않는다(self):
        rng = np.random.default_rng(7)
        lon, lat = _rand_points(rng, 10)
        dist, dur, _ = RT.road_distance_matrix(lon, lat, use_road=False)
        order = RT.solve_route(dist, 0)
        self.assertEqual(RT.split_by_budget(order, dur, dist, 0.0), [order])


# ==================================================================== 배분

def _knapsack_exact(values, weights, cap):
    """0/1 배낭 최적값 (정수 무게 완전탐색). 정답을 알기 위한 참조 구현."""
    n = len(values)
    best = 0.0
    for mask in range(1 << n):
        w = sum(weights[i] for i in range(n) if mask >> i & 1)
        if w <= cap:
            best = max(best, sum(values[i] for i in range(n) if mask >> i & 1))
    return best


class TestAllocation(unittest.TestCase):

    def test_예산을_절대_넘지_않는다(self):
        for s in range(20):
            rng = np.random.default_rng(1300 + s)
            n = int(rng.integers(10, 200))
            df = pd.DataFrame({
                "grid_id": [f"g{i}" for i in range(n)],
                "target_total": rng.integers(0, 40, n),
                "biz_total": rng.integers(0, 15, n),
            })
            risk = rng.gamma(1.2, 0.8, n)
            cap = OP.Capacity(inspectors=int(rng.integers(1, 6)), per_day=4,
                              days=int(rng.integers(5, 30)))
            got = OP.allocate(df, risk, cap)
            self.assertLessEqual(got["cost"].sum(), cap.total_visits + 1e-6)
            self.assertEqual(got["grid_id"].nunique(), len(got), "같은 격자 중복 배정")
            self.assertEqual(list(got["점검순서"]), list(range(1, len(got) + 1)))
            if len(got) > 1:      # 효율 내림차순이어야 한다
                eff = got["expected_fires"] / got["cost"]
                self.assertTrue((np.diff(eff.to_numpy()) <= 1e-9).all())

    def test_완전탐색_배낭과_비교(self):
        """탐욕해가 최적해에 얼마나 붙는가. 이론상 '가장 비싼 한 칸' 이내다."""
        ratios = []
        for s in range(30):
            rng = np.random.default_rng(1500 + s)
            n = 14
            cost = rng.integers(1, 12, n)
            val = rng.uniform(0.05, 4.0, n)
            cap = int(cost.sum() * 0.4)
            df = pd.DataFrame({"grid_id": [f"g{i}" for i in range(n)],
                               "target_total": cost, "biz_total": 0})
            capacity = OP.Capacity(inspectors=cap, per_day=1, days=1)
            got = OP.allocate(df, val, capacity, cost=pd.Series(cost, dtype=float))
            greedy = float(got["expected_fires"].sum())
            exact = _knapsack_exact(val, cost, cap)
            self.assertLessEqual(greedy, exact + 1e-9, "탐욕해가 최적해를 넘을 수 없다")
            ratios.append(greedy / exact if exact > 0 else 1.0)
        self.assertGreaterEqual(np.mean(ratios), 0.92,
                                f"탐욕해 평균 {np.mean(ratios):.3f} — 너무 나쁘다")
        self.assertGreaterEqual(min(ratios), 0.75,
                                f"최악 {min(ratios):.3f} — 최적해와 지나치게 벌어진다")

    def test_형평성_배분은_모든_관할에_최소치를_준다(self):
        rng = np.random.default_rng(21)
        n = 400
        df = pd.DataFrame({
            "grid_id": [f"g{i}" for i in range(n)],
            "sgg": rng.choice(["중구", "남구", "동구", "북구", "울주군"], n),
            "target_total": rng.integers(0, 30, n),
            "biz_total": rng.integers(0, 10, n),
            "fires": rng.poisson(0.6, n),
        })
        risk = rng.gamma(1.0, 1.0, n)
        cap = OP.Capacity(inspectors=3, per_day=4, days=20)
        alloc, info = OP.allocate_with_equity(df, risk, cap, min_share=1.0)
        self.assertLessEqual(alloc["cost"].sum(), cap.total_visits + 1e-6)
        self.assertEqual(alloc["grid_id"].nunique(), len(alloc))
        if info.get("equity_constrained"):
            got = set(alloc["sgg"].astype(str))
            for g, v in info["by_group"].items():
                if v["risk_share"] > 0:
                    self.assertIn(g, got, f"{g} 관할이 통째로 빠졌다")

    def test_점검비용_가중치가_실제로_반영된다(self):
        """스프링클러가 있는 격자는 같은 개수라도 비용이 더 커야 한다."""
        df = pd.DataFrame({
            "fac_n_스프링클러": [10, 0],
            "fac_n_일반대상물": [0, 10],
        })
        c = OP.inspection_cost(df)
        self.assertGreater(c.iloc[0], c.iloc[1])
        self.assertAlmostEqual(c.iloc[0] / c.iloc[1], OP.COST_WEIGHTS["fac_n_스프링클러"], places=6)


# ==================================================================== 지표

class TestMetrics(unittest.TestCase):

    def test_포착률_손계산(self):
        """격자 10개, 화재 총 10건. 상위 20%(2칸)에 7건이면 0.7 이다."""
        y = np.array([5, 2, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        score = np.arange(10)[::-1].astype(float)
        self.assertAlmostEqual(E.capture_at_k(y, score, 20.0), 0.7)
        self.assertAlmostEqual(E.lift_at_k(y, score, 20.0), 3.5)

    def test_PAI_정의(self):
        """PAI = (포착 화재비율) / (선택 면적비율). Chainey 2008."""
        rng = np.random.default_rng(5)
        y = rng.poisson(0.5, 500).astype(float)
        s = y + rng.normal(0, 0.4, 500)
        for k in (5.0, 10.0, 20.0):
            cap = E.capture_at_k(y, s, k)
            self.assertAlmostEqual(E.pai(y, s, k), cap / (k / 100.0), places=9)

    def test_PEI_는_0과_1_사이(self):
        """PEI = PAI / PAI_max. 예언자를 이길 수는 없다."""
        rng = np.random.default_rng(6)
        for s in range(15):
            y = rng.poisson(0.4, 400).astype(float)
            sc = y * rng.uniform(0, 1) + rng.normal(0, 1, 400)
            v = E.pei(y, sc, 20.0)
            self.assertGreaterEqual(v, -1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9, f"seed={s}: PEI {v:.4f} > 1")

    def test_예언자_순위의_PEI는_정확히_1(self):
        rng = np.random.default_rng(8)
        y = rng.poisson(0.5, 300).astype(float)
        self.assertAlmostEqual(E.pei(y, y, 20.0), 1.0, places=9)

    def test_등급표_단조성과_합(self):
        rng = np.random.default_rng(9)
        y = rng.poisson(0.5, 1000).astype(float)
        d = E.decile_table(y, y)
        self.assertTrue(E.is_monotonic(d))
        self.assertAlmostEqual(d["total_fires"].sum(), y.sum())
        self.assertAlmostEqual(d["share_of_fires"].sum(), 1.0, places=6)
        self.assertEqual(d["n_grids"].sum(), len(y))
        # 최고 등급이 최고위험이어야 한다
        self.assertGreater(d.loc[d["grade"].idxmax(), "mean_fires"],
                           d.loc[d["grade"].idxmin(), "mean_fires"])

    def test_지니계수_기준값(self):
        self.assertAlmostEqual(E.gini([1, 1, 1, 1]), 0.0, places=9)
        g = E.gini([0, 0, 0, 1])
        self.assertAlmostEqual(g, 0.75, places=6)

    def test_동점은_시드로만_갈린다(self):
        """베이스라인은 0점 동점이 수천 개다. 입력 순서가 성능을 바꾸면 안 된다."""
        rng = np.random.default_rng(10)
        y = rng.poisson(0.3, 800).astype(float)
        s = np.zeros(800)
        a = E.capture_at_k(y, s, 20.0, seed=42)
        perm = rng.permutation(800)
        b = E.capture_at_k(y[perm], s[perm], 20.0, seed=42)
        # 서로 다른 표본을 섞은 것이므로 값 자체는 달라질 수 있으나,
        # 무작위 배정의 기대값(=0.2) 근처여야 한다.
        for v in (a, b):
            self.assertLess(abs(v - 0.2), 0.12, f"동점 처리 편향: {v:.3f}")

    def test_부트스트랩_구간이_점추정을_포함한다(self):
        rng = np.random.default_rng(12)
        y = rng.poisson(0.5, 600).astype(float)
        s = y + rng.normal(0, 0.5, 600)
        ci = E.bootstrap_capture_ci(y, s, 20.0, n_boot=300)
        self.assertLessEqual(ci["lo"], ci["point"] + 1e-9)
        self.assertGreaterEqual(ci["hi"], ci["point"] - 1e-9)
        self.assertGreater(ci["n_boot"], 250)

    def test_짝지은_부트스트랩은_같은_재표본을_쓴다(self):
        """두 모델의 차이를 볼 때 재표본이 다르면 차이가 부풀거나 사라진다.
        같은 점수를 넣으면 차이는 정확히 0이어야 한다."""
        rng = np.random.default_rng(13)
        y = rng.poisson(0.5, 400).astype(float)
        s = y + rng.normal(0, 0.5, 400)
        d = E.bootstrap_delta_ci(y, s, s, 20.0, n_boot=200)
        self.assertAlmostEqual(d["point_pp"], 0.0, places=9)
        self.assertAlmostEqual(d["lo_pp"], 0.0, places=6)
        self.assertAlmostEqual(d["hi_pp"], 0.0, places=6)
        self.assertFalse(d["excludes_zero"])

    def test_형평성_비중_합계(self):
        rng = np.random.default_rng(14)
        n = 500
        y = rng.poisson(0.5, n).astype(float)
        s = y + rng.normal(0, 0.5, n)
        g = rng.choice(list("ABCD"), n)
        eq = E.equity(y, s, g, 20.0)
        self.assertAlmostEqual(eq["share_of_grids"].sum(), 1.0, places=6)
        self.assertAlmostEqual(eq["share_of_fires"].sum(), 1.0, places=6)
        self.assertAlmostEqual(eq["share_of_inspections"].sum(), 1.0, places=6)

    def test_캘리브레이션_완벽예측(self):
        rng = np.random.default_rng(15)
        y = rng.poisson(0.7, 300).astype(float)
        c = E.calibration(y, y)
        self.assertAlmostEqual(c["total_ratio"], 1.0, places=6)


# ==================================================================== 계절

class TestSeasonal(unittest.TestCase):

    def test_계절지수_평균은_1(self):
        rng = np.random.default_rng(16)
        rows = []
        for yr in range(2015, 2022):
            for m in range(1, 13):
                rows.append({"year": yr, "month": m,
                             "fires": rng.poisson(20 + 10 * np.cos((m - 1) / 12 * 2 * np.pi))})
        df = pd.DataFrame(rows)
        idx = MO.seasonal_index(df)
        self.assertEqual(len(idx), 12)
        self.assertAlmostEqual(float(idx["seasonal_index"].mean()), 1.0, delta=0.02)

    def test_계절지수는_연도별_규모에_흔들리지_않는다(self):
        """화재 총건수가 해마다 달라도 '몇 월이 위험한가'는 같아야 한다."""
        base = {m: 20 + 10 * np.cos((m - 1) / 12 * 2 * np.pi) for m in range(1, 13)}
        rows_a = [{"year": y, "month": m, "fires": base[m]}
                  for y in range(2015, 2022) for m in range(1, 13)]
        rows_b = [{"year": y, "month": m, "fires": base[m] * (1 + 0.5 * (y - 2015))}
                  for y in range(2015, 2022) for m in range(1, 13)]
        a = MO.seasonal_index(pd.DataFrame(rows_a)).set_index("month")["seasonal_index"]
        b = MO.seasonal_index(pd.DataFrame(rows_b)).set_index("month")["seasonal_index"]
        for m in range(1, 13):
            self.assertAlmostEqual(a[m], b[m], places=6,
                                   msg=f"{m}월 지수가 연도 규모에 끌려갔다")


# ==================================================================== 순찰 목적

class TestPatrolModes(unittest.TestCase):

    def test_목적이_다르면_대상도_다르다(self):
        rng = np.random.default_rng(17)
        n = 300
        df = pd.DataFrame({
            "grid_id": [f"g{i}" for i in range(n)],
            "lon": 129.3 + rng.normal(0, .05, n), "lat": 35.5 + rng.normal(0, .05, n),
            "pred": rng.gamma(1, 1, n),
            "biz_total": rng.integers(0, 30, n),
            "biz_n_유흥주점": rng.integers(0, 8, n),
            "target_total": rng.integers(0, 40, n),
            "n_hydrant": rng.integers(0, 5, n),
            "fires_cum": rng.integers(0, 12, n),
        })
        picks = {}
        for key, mode in PM.MODES.items():
            sel = PM.select_targets(df, mode, 20)
            self.assertEqual(len(sel), 20)
            self.assertEqual(sel["grid_id"].nunique(), 20)
            self.assertTrue((np.diff(sel["순찰점수"].to_numpy()) <= 1e-9).all(),
                            f"{key}: 순찰점수 내림차순이 아니다")
            picks[key] = set(sel["grid_id"])
        pairs = list(itertools.combinations(picks, 2))
        diff = [len(picks[a] ^ picks[b]) for a, b in pairs]
        self.assertGreater(max(diff), 0, "모든 목적이 같은 격자를 고르면 목적 구분이 무의미하다")

    def test_모든_목적에_법령근거가_붙어_있다(self):
        for key, mode in PM.MODES.items():
            self.assertTrue(mode.checks, f"{key}: 확인항목 없음")
            self.assertTrue(mode.legal_refs, f"{key}: 법령근거 없음")
            for ref in mode.legal_refs:
                self.assertRegex(str(ref), r"제\s*\d+\s*조",
                                 f"{key}: 조문 번호 없는 법령근거 '{ref}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
