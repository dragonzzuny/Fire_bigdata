"""가짜 원본 CSV 로 '적재 -> 격자 -> 패널' 전 구간을 실제로 돌린다.

합성 데이터는 `make_synthetic_raw.py` 가 만들며, 플랫폼 실제 배포본의 형태를
그대로 흉내낸다: 영문 컬럼 코드, 조각난 주소, 자주 비는 도로명,
`_0000`/`_2021` 두 벌, `_2021` 의 뭉개진 시각, 경계 밖 좌표.
지오코딩은 캐시를 미리 심어 API 없이 재현한다.
"""
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests_firebird"))
from firebird.config import Paths, load_config          # noqa: E402
from firebird import dataset as D, features as F        # noqa: E402
import make_synthetic_raw as SYN                        # noqa: E402


class TestRawToPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="firebird_test_"))
        SYN.main(cls.tmp)
        base = load_config()
        p = base.raw["paths"]
        paths = Paths(root=base.paths.root,
                      raw=cls.tmp / p["raw"], interim=cls.tmp / p["interim"],
                      processed=cls.tmp / p["processed"], cache=cls.tmp / p["cache"],
                      outputs=cls.tmp / p["outputs"], figures=cls.tmp / p["figures"]).ensure()
        cls.cfg = replace(base, paths=paths)
        cls.data = D.load_city(cls.cfg, "ulsan", use_api=False)
        cls.sejong = D.load_city(cls.cfg, "sejong", use_api=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------------------------------------------------- 적재

    def test_all_datasets_loaded(self):
        for name, df in (("fires", self.data.fires), ("targets", self.data.targets),
                         ("businesses", self.data.businesses), ("hydrants", self.data.hydrants)):
            self.assertGreater(len(df), 0, f"{name} 가 비었다")

    def test_english_column_codes_are_mapped_to_standard_names(self):
        """RCPT_DT -> occurred_at 같은 매핑이 실제로 걸려야 한다."""
        for col in ("occurred_at", "emd", "sido", "year"):
            self.assertIn(col, self.data.fires.columns)
        self.assertIn("biz_type", self.data.businesses.columns)
        self.assertIn("usage", self.data.targets.columns)

    # ---------------------------------------------------------- 지오코딩 계층

    def test_geocoding_uses_road_and_falls_back_to_emd(self):
        """도로명이 없는 화재도 읍면동으로 좌표를 얻어야 한다."""
        by_level = self.data.manifest["coverage"]["fire"]["by_level"]
        self.assertIn("road", by_level)
        self.assertIn("emd", by_level, "도로명 결측분이 읍면동으로 폴백되지 않았다")
        self.assertGreater(self.data.manifest["coverage"]["fire"]["rate"], 0.95)

    def test_level_breakdown_is_recorded_for_honesty(self):
        """'좌표 95%'만 말하고 그중 몇 %가 동 중심점인지 숨기면 안 된다."""
        for name in ("fire", "target", "business"):
            self.assertIn("by_level", self.data.manifest["coverage"][name])

    def test_source_coordinates_used_when_inside_bbox(self):
        """울산 소화전은 원본 좌표가 정확하므로 그대로 써야 한다."""
        self.assertEqual(self.data.manifest["coverage"]["hydrant"]["by_level"],
                         {"source_xy": self.data.manifest["coverage"]["hydrant"]["rows"]})

    def test_out_of_bbox_source_coordinates_are_rejected(self):
        """세종 소화전 좌표는 실제로 경계 밖이다 — 버리고 주소로 대체해야 한다."""
        sc = self.sejong.manifest["source_coords"]["hydrant"]
        self.assertGreater(sc["present"], 0)
        self.assertEqual(sc["within_bbox"], 0)
        self.assertNotIn("source_xy", self.sejong.manifest["coverage"]["hydrant"]["by_level"])

    # ---------------------------------------------------------- 라벨

    def test_year_extracted_for_every_fire(self):
        m = self.data.manifest["fires"]
        self.assertEqual(m["with_year"], m["rows"])
        self.assertGreater(m["usable"], 0)

    def test_masked_times_are_flagged_not_silently_used(self):
        """_2021 파일은 시각이 000000 이다. 시간대 분석에서 빠져야 한다."""
        m = self.data.manifest["fires"]
        self.assertLess(m["with_real_time"], m["rows"])
        self.assertGreater(m["with_real_time"], 0)
        self.assertTrue(self.data.fires["hour"].isna().any())
        self.assertTrue(self.data.fires["hour"].notna().any())

    def test_label_total_matches_usable_source_rows(self):
        self.assertEqual(self.data.panel["fires"].sum(),
                         self.data.manifest["fires"]["usable"])

    def test_every_year_in_range_is_present(self):
        years = self.data.manifest["panel"]["years"]
        self.assertEqual(years, list(range(self.cfg.year_min, self.cfg.year_max + 1)))

    # ---------------------------------------------------------- 패널

    def test_panel_is_complete_grid_by_year(self):
        p = self.data.panel
        self.assertEqual(len(p), p["grid_id"].nunique() * p["year"].nunique())

    def test_zero_fire_grids_present(self):
        self.assertGreater((self.data.panel["fires"] == 0).sum(), 0)

    def test_usage_and_facility_features_replace_missing_grade(self):
        """원본에 '특급/1급' 등급이 없으므로 용도·설치대상으로 대체돼야 한다."""
        cols = set(self.data.panel.columns)
        self.assertTrue(any(c.startswith("usage_n_") for c in cols))
        self.assertTrue(any(c.startswith("fac_n_") for c in cols))
        self.assertTrue(any(c.startswith("biz_n_") for c in cols))

    def test_hydrant_counts_only_hydrants(self):
        """저수조·급수탑을 소화전으로 세면 사각지대가 실제보다 적어 보인다."""
        self.assertIn("n_hydrant", self.data.panel.columns)
        n_rows = len(self.data.hydrants)
        self.assertLess(self.data.panel.groupby("grid_id")["n_hydrant"].first().sum(), n_rows)

    def test_hydrant_distance_finite_and_nonnegative(self):
        d = self.data.panel["dist_hydrant_m"]
        self.assertTrue(np.isfinite(d).any())
        self.assertTrue((d >= 0).all())

    def test_sgg_attached_for_logo(self):
        self.assertGreater(self.data.panel["sgg"].nunique(), 1)

    def test_single_tier_city_falls_back_to_emd_for_grouping(self):
        """세종은 시군구가 없다. 그래도 LOGO 그룹이 있어야 한다."""
        self.assertGreater(self.sejong.panel["sgg"].nunique(), 1)

    def test_no_leakage(self):
        F.assert_no_leakage(self.data.panel)

    def test_coordinates_excluded_from_features(self):
        feats = F.feature_columns(self.data.panel)
        for banned in ("gx", "gy", "lon", "lat", "fires"):
            self.assertNotIn(banned, feats)

    # ---------------------------------------------------------- 저장

    def test_save_and_reload_roundtrip(self):
        paths = D.save_city(self.data, self.cfg)
        self.assertTrue(Path(paths["panel"]).exists())
        again = D.load_panel(self.cfg, "ulsan")
        self.assertEqual(len(again), len(self.data.panel))
        man = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        self.assertIn("geocode", man)
        self.assertIn("coverage", man)


if __name__ == "__main__":
    unittest.main(verbosity=2)
