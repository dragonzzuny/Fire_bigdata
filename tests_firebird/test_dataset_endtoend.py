"""가짜 원본 CSV 로 '적재 -> 격자 -> 패널' 전 구간을 실제로 돌린다.

지오코딩은 캐시를 미리 심어 API 없이 재현한다.
여기서 확인하는 것: 인코딩 처리, 컬럼 별칭 해석, 연도 추출,
격자 배정, 화재 0 격자 보존, 소화전 거리, 누수 가드.
"""
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from firebird.config import Paths, load_config          # noqa: E402
from firebird import dataset as D, features as F        # noqa: E402


ROADS = ["삼산로", "번영로", "태화로", "왕생로", "문수로", "옥동로"]
SGGS = ["남구", "중구", "동구", "북구", "울주군"]
# 격자가 실제로 갈라지도록 (시군구, 도로) 조합마다 다른 좌표를 준다.
# 조합 수(30개)가 연도당 화재 건수보다 충분히 많아야 '화재 0인 격자'가 생긴다 —
# 그 격자가 없으면 상위 k% 포착률의 분모가 실제 도시와 달라진다.
COORDS = {(s, r): (129.30 + (i * len(ROADS) + j) * 0.009,
                   35.50 + (i * len(ROADS) + j) * 0.006)
          for i, s in enumerate(SGGS) for j, r in enumerate(ROADS)}


def make_raw(raw_dir: Path, seed: int = 0, n_fire: int = 260) -> None:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def addr(i):
        r = ROADS[i % len(ROADS)]
        s = SGGS[i % len(SGGS)]
        return f"울산광역시 {s} {r} {rng.integers(1, 300)}", s

    rows = []
    for i in range(n_fire):
        a, s = addr(int(rng.integers(0, 1000)))
        y = int(rng.integers(2014, 2022))
        rows.append({"화재발생일시": f"{y}-{rng.integers(1,13):02d}-{rng.integers(1,28):02d} "
                                  f"{rng.integers(0,24):02d}:{rng.integers(0,60):02d}:00",
                     "도로명주소": a, "시군구": s, "화재종별": rng.choice(["건축·구조물", "차량", "기타"])})
    # cp949 로 저장한다 — 플랫폼 CSV 가 흔히 그렇다.
    pd.DataFrame(rows).to_csv(raw_dir / "울산광역시소방본부_화재발생현황.csv",
                              index=False, encoding="cp949")

    rows = []
    for i in range(600):
        a, s = addr(i)
        rows.append({"대상물명": f"대상물{i}", "소재지도로명주소": a, "시군구명": s,
                     "특정소방대상물등급": rng.choice(["특급", "1급", "2급", "3급"], p=[.05, .15, .4, .4])})
    pd.DataFrame(rows).to_csv(raw_dir / "울산광역시소방본부_특정소방대상물현황.csv",
                              index=False, encoding="utf-8-sig")

    rows = []
    for i in range(400):
        a, s = addr(i)
        rows.append({"업소명": f"업소{i}", "영업장주소": a, "관할구역": s,
                     "영업의종류": rng.choice(["일반음식점", "노래연습장", "유흥주점",
                                          "인터넷컴퓨터게임시설제공업", "고시원"])})
    pd.DataFrame(rows).to_csv(raw_dir / "울산광역시소방본부_다중이용업소현황.csv",
                              index=False, encoding="cp949")

    rows = []
    for i in range(150):
        a, s = addr(i)
        rows.append({"설치장소": a, "소방용수시설구분": rng.choice(["소화전", "저수조", "급수탑"]),
                     "시군구": s})
    pd.DataFrame(rows).to_csv(raw_dir / "울산광역시소방본부_소방용수시설운영현황.csv",
                              index=False, encoding="cp949")


def seed_geocode_cache(cache_path: Path) -> None:
    """API 대신 (시군구, 도로)별 고정 좌표를 캐시에 심는다."""
    data = {}
    for (sgg, road), (lon, lat) in COORDS.items():
        data[f"울산광역시 {sgg} {road}"] = {"lon": lon, "lat": lat, "matched": True}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestRawToPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="firebird_test_"))
        base = load_config()
        paths = Paths(root=base.paths.root,
                      raw=cls.tmp / "raw", interim=cls.tmp / "interim",
                      processed=cls.tmp / "processed", cache=cls.tmp / "cache",
                      outputs=cls.tmp / "outputs", figures=cls.tmp / "figures").ensure()
        cls.cfg = replace(base, paths=paths)
        make_raw(paths.raw / "ulsan")
        seed_geocode_cache(paths.cache / cls.cfg["geocode"]["cache_file"])
        cls.data = D.load_city(cls.cfg, "ulsan")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_all_datasets_loaded(self):
        self.assertGreater(len(self.data.fires), 0)
        self.assertGreater(len(self.data.targets), 0)
        self.assertGreater(len(self.data.businesses), 0)
        self.assertGreater(len(self.data.hydrants), 0)

    def test_cp949_and_utf8_both_read(self):
        """인코딩이 섞여 있어도 한글 컬럼이 깨지지 않아야 한다."""
        self.assertTrue(self.data.targets["address"].str.contains("울산광역시").any())

    def test_geocoding_coverage_recorded(self):
        cov = self.data.manifest["coverage"]["fire"]
        self.assertGreater(cov["rate"], 0.9)
        self.assertEqual(cov["rows"], len(self.data.fires))

    def test_years_extracted_in_range(self):
        m = self.data.manifest["fires"]
        self.assertEqual(m["with_year"], m["rows"])
        self.assertGreater(m["usable"], 0)

    def test_panel_covers_every_grid_and_year(self):
        p = self.data.panel
        n_grid = p["grid_id"].nunique()
        n_year = p["year"].nunique()
        self.assertEqual(len(p), n_grid * n_year)
        self.assertEqual(n_year, self.cfg.year_max - self.cfg.year_min + 1)

    def test_zero_fire_grids_present(self):
        p = self.data.panel
        self.assertGreater((p["fires"] == 0).sum(), 0)

    def test_label_total_matches_source(self):
        """패널의 화재 합계 = 격자·연도 배정에 성공한 원본 화재 건수."""
        self.assertEqual(self.data.panel["fires"].sum(),
                         self.data.manifest["fires"]["usable"])

    def test_static_features_present(self):
        cols = set(self.data.panel.columns)
        self.assertTrue(any(c.startswith("target_n_") for c in cols))
        self.assertTrue(any(c.startswith("biz_n_") for c in cols))
        self.assertIn("n_hydrant", cols)
        self.assertIn("dist_hydrant_m", cols)

    def test_hydrant_distance_is_finite_where_hydrants_exist(self):
        d = self.data.panel["dist_hydrant_m"]
        self.assertTrue(np.isfinite(d).any())
        self.assertTrue((d >= 0).all())

    def test_sgg_attached_for_logo(self):
        self.assertIn("sgg", self.data.panel.columns)
        self.assertGreater(self.data.panel["sgg"].nunique(), 1)

    def test_no_leakage(self):
        F.assert_no_leakage(self.data.panel)

    def test_feature_columns_are_numeric_and_exclude_label(self):
        feats = F.feature_columns(self.data.panel)
        self.assertNotIn("fires", feats)
        self.assertGreater(len(feats), 5)

    def test_save_and_reload_roundtrip(self):
        paths = D.save_city(self.data, self.cfg)
        self.assertTrue(Path(paths["panel"]).exists())
        again = D.load_panel(self.cfg, "ulsan")
        self.assertEqual(len(again), len(self.data.panel))
        man = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        self.assertIn("geocode", man)


if __name__ == "__main__":
    unittest.main(verbosity=2)
