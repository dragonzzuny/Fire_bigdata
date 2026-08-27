"""스크립트 4개를 실제 프로세스로 처음부터 끝까지 돌린다.

모듈 단위 테스트가 다 통과해도 스크립트가 서로 안 맞물릴 수 있다.
여기서는 합성 원본 CSV 를 만들고 run_all 과 같은 순서로 실행해,
마지막에 실제 산출물 파일이 생겼는지 파일시스템에서 확인한다.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable


def run(script: str, *args: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=900)


@unittest.skipUnless(Path(PY).exists(), "venv 파이썬 없음")
class TestScriptsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="firebird_scripts_"))
        cls.env = dict(os.environ, FIREBIRD_DATA_ROOT=str(cls.tmp))
        gen = subprocess.run([PY, str(ROOT / "tests_firebird" / "make_synthetic_raw.py"),
                              str(cls.tmp)], capture_output=True, text=True, timeout=300)
        assert gen.returncode == 0, gen.stderr
        cls.out = cls.tmp / "outputs"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_inspect_finds_all_required_columns(self):
        r = run("01_inspect_raw.py", env=self.env)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn("모든 필수 컬럼이 해석된다", r.stdout)

    def test_02_geocode_dry_run_counts_keys(self):
        r = run("02_geocode.py", "--dry", env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("고유 지오코딩 키", r.stdout)

    def test_03_build_writes_panel_and_manifest(self):
        r = run("03_build_dataset.py", "--no-api", env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        proc = self.tmp / "data" / "processed"
        for city in ("ulsan", "sejong"):
            self.assertTrue((proc / f"panel_{city}.parquet").exists(), f"{city} 패널 없음")
            man = json.loads((proc / f"manifest_{city}.json").read_text(encoding="utf-8"))
            self.assertGreater(man["panel"]["grids"], 0)
            self.assertGreater(man["panel"]["total_fires"], 0)

    def test_04_train_writes_evaluation_with_all_protocols(self):
        run("03_build_dataset.py", "--no-api", env=self.env)
        r = run("04_train_eval.py", env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        ev = json.loads((self.out / "evaluation.json").read_text(encoding="utf-8"))
        self.assertIn("temporal", ev)
        self.assertIn("logo", ev)
        self.assertIn("transfer", ev)
        self.assertIn("shap_importance", ev)
        # 홀드아웃 연도가 학습에 들어가면 안 된다 — 보고서 자체로 확인한다.
        self.assertNotIn(ev["temporal"]["test_year"], ev["temporal"]["train_years"])
        # 좌표가 피처에 섞이면 새 도시에서 무너진다.
        for banned in ("gx", "gy", "gx_c", "gy_c", "lon", "lat"):
            self.assertNotIn(banned, ev["features"])
        self.assertTrue((self.out / "model_ulsan.joblib").exists())

    def test_05_artifacts_writes_every_screen(self):
        run("03_build_dataset.py", "--no-api", env=self.env)
        run("04_train_eval.py", env=self.env)
        r = run("05_artifacts.py", "--no-llm", "--plans", "3", env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        year = 2021
        expect = [f"priority_ulsan_{year}.csv",
                  f"inspection_plans_ulsan_{year}.json",
                  f"patrol_route_ulsan_{year}.csv",
                  f"patrol_hours_ulsan.csv",
                  f"artifacts_summary_ulsan_{year}.json"]
        for name in expect:
            self.assertTrue((self.out / name).exists(), f"{name} 없음")
        plans = list((self.out / "inspection_plans").glob("*.txt"))
        self.assertEqual(len(plans), 3)
        text = plans[0].read_text(encoding="utf-8")
        self.assertIn("점검계획서", text)
        self.assertIn("유의사항", text)
        summary = json.loads((self.out / f"artifacts_summary_ulsan_{year}.json")
                             .read_text(encoding="utf-8"))
        self.assertTrue(summary["hydrant_coverage"]["available"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
