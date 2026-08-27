"""합성 데이터로 파이프라인 전체를 돌린다.

원본 CSV 가 아직 없어도 '코드가 실제로 도는가'를 여기서 판정한다.
합성 데이터는 진짜 도시가 아니므로 성능 수치를 주장하지 않는다.
확인하는 것은 배선(형상, 누수, 프로토콜)이 맞물리는가 뿐이다.
"""
import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from firebird.config import load_config          # noqa: E402
from firebird import features as F, model as M, evaluate as E  # noqa: E402


def synthetic_panel(seed=0, n_grid=600, years=range(2014, 2022), sgg_n=5):
    """위험이 '대상물 수 + 과거화재'에 실제로 의존하는 가짜 도시."""
    rng = np.random.default_rng(seed)
    gx = rng.integers(0, 40, n_grid)
    gy = rng.integers(0, 40, n_grid)
    ids = pd.unique(pd.Series([f"{a}_{b}" for a, b in zip(gx, gy)]))
    n = len(ids)
    intensity = rng.gamma(0.6, 0.7, n)                 # 격자별 잠재 위험
    n_target = rng.poisson(3 + 12 * intensity, n)
    n_biz = rng.poisson(1 + 6 * intensity, n)
    sgg = rng.integers(0, sgg_n, n)

    recs = []
    for y in years:
        lam = 0.05 + 0.35 * intensity + 0.01 * n_target
        k = rng.poisson(lam)
        for gid, c in zip(ids, k):
            for _ in range(int(c)):
                recs.append({"grid_id": gid, "year": y})
    fires = pd.DataFrame(recs)

    static = pd.DataFrame({"grid_id": ids,
                           "target_total": n_target.astype(float),
                           "biz_total": n_biz.astype(float),
                           "sgg": [f"제{i+1}구" for i in sgg]})

    counts = F.fire_counts_by_grid_year(fires)
    skel = F.build_panel_skeleton(pd.Series(ids), list(years))
    panel = F.add_fire_history(skel, counts, ring=1)
    panel = panel.merge(static, on="grid_id", how="left")
    panel[["target_total", "biz_total"]] = panel[["target_total", "biz_total"]].fillna(0.0)
    return panel


class TestSyntheticEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.panel = synthetic_panel()
        cls.feats = F.feature_columns(cls.panel)

    def test_panel_is_well_formed(self):
        self.assertGreater(len(self.panel), 1000)
        self.assertIn("fires", self.panel.columns)
        F.assert_no_leakage(self.panel)

    def test_feature_columns_exclude_label_and_ids(self):
        for banned in ("fires", "grid_id", "year", "sgg"):
            self.assertNotIn(banned, self.feats)
        self.assertIn("fires_lag1", self.feats)
        self.assertIn("neigh_fires_lag1", self.feats)

    def test_temporal_validation_runs_and_beats_random(self):
        out = M.temporal_validation(self.panel, self.feats, self.cfg)
        r = out["result"]
        self.assertEqual(r["protocol"], "temporal_holdout")
        self.assertEqual(r["test_year"], self.cfg.holdout_year)
        self.assertNotIn(self.cfg.holdout_year, r["train_years"])   # 누수 없음
        # 신호가 있는 합성 데이터이므로 무작위(=k%)보다는 나와야 한다.
        self.assertGreater(r["headline"]["model_capture"], 0.20)
        self.assertGreater(r["headline"]["model_lift"], 1.0)

    def test_report_renders(self):
        out = M.temporal_validation(self.panel, self.feats, self.cfg)
        text = E.format_report(out["result"])
        self.assertIn("핵심(상위 20%)", text)

    def test_logo_validation_covers_every_group(self):
        out = M.logo_validation(self.panel, self.feats, self.cfg, group_col="sgg")
        self.assertEqual(out["protocol"], "leave_one_group_out")
        self.assertGreaterEqual(len(out["per_group"]), 3)
        self.assertTrue(0.0 <= out["capture_mean"] <= 1.0)

    def test_transfer_to_another_city_runs(self):
        target = synthetic_panel(seed=99, n_grid=300, sgg_n=3)
        out = M.transfer_validation(self.panel, target, self.feats, self.cfg,
                                    target_years=[self.cfg.holdout_year])
        self.assertEqual(out["protocol"], "cross_city_transfer")
        self.assertTrue(0.0 <= out["headline"]["model_capture"] <= 1.0)

    def test_transfer_tolerates_missing_feature_columns(self):
        """도시마다 업종/등급 컬럼 구성이 달라도 죽지 않아야 한다."""
        target = synthetic_panel(seed=7, n_grid=250).drop(columns=["biz_total"])
        out = M.transfer_validation(self.panel, target, self.feats, self.cfg,
                                    target_years=[self.cfg.holdout_year])
        self.assertIn("headline", out)

    def test_single_feature_probe_reports_each_candidate(self):
        probe = M.single_feature_probe(self.panel, self.cfg)
        self.assertGreater(len(probe), 1)
        self.assertIn("feature", probe.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
