"""법령 검색·질의응답·계획서 생성. 네트워크 없이 검증한다."""
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from firebird import assistant as A, lawdata as LW, patrol_modes as PM, plans as PL  # noqa: E402


ARTICLES = pd.DataFrame([
    {"law": "화재의 예방 및 안전관리에 관한 법률", "article": "제18조",
     "title": "화재예방강화지구의 지정 등",
     "text": "시·도지사는 시장지역, 공장·창고가 밀집한 지역, 목조건물이 밀집한 지역을 "
             "화재예방강화지구로 지정할 수 있다. 소방관서장은 화재예방강화지구 안의 "
             "소방대상물에 대하여 화재안전조사를 하여야 한다."},
    {"law": "화재의 예방 및 안전관리에 관한 법률 시행령", "article": "제20조",
     "title": "화재예방강화지구의 관리",
     "text": "소방관서장은 화재안전조사를 연 1회 이상 실시하여야 한다. "
             "훈련 또는 교육 10일 전까지 관계인에게 통보하여야 한다."},
    {"law": "다중이용업소의 안전관리에 관한 특별법", "article": "제9조",
     "title": "안전시설등의 설치·유지",
     "text": "다중이용업주는 영업장에 안전시설등을 설치·유지하여야 한다. "
             "노래연습장업의 경우 비상구와 피난통로를 확보하여야 한다."},
])


class TestTokenize(unittest.TestCase):
    def test_strips_korean_particles(self):
        toks = A.tokenize("점검주기가 어떻게 되나요")
        self.assertIn("점검주기", toks)

    def test_indexes_compound_prefix(self):
        toks = A.tokenize("소방안전관리자")
        self.assertTrue(any(t.startswith("소방") for t in toks))

    def test_handles_empty_and_symbols(self):
        self.assertEqual(A.tokenize(""), [])
        self.assertEqual(A.tokenize("!!! ???"), [])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.index = A.BM25(A.law_docs(ARTICLES))

    def test_finds_the_right_article(self):
        hits = self.index.search("화재예방강화지구 지정 지역", top_k=3)
        self.assertTrue(hits)
        self.assertIn("제18조", hits[0][0].ref)

    def test_finds_cycle_article(self):
        hits = self.index.search("화재안전조사 연 1회", top_k=3)
        refs = [d.ref for d, _ in hits]
        self.assertTrue(any("제20조" in r for r in refs))

    def test_unrelated_query_scores_low_or_empty(self):
        hits = self.index.search("김치찌개 조리법", top_k=3)
        self.assertTrue(not hits or hits[0][1] < 5.0)

    def test_scores_are_sorted(self):
        hits = self.index.search("다중이용업소 안전시설", top_k=3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestDocSources(unittest.TestCase):
    def test_rule_docs_cover_business_and_patrol(self):
        docs = A.rule_docs()
        titles = " ".join(d.title for d in docs)
        self.assertIn("노래연습장", titles)
        self.assertIn("야간순찰", titles)

    def test_data_docs_summarise_stations(self):
        panel = pd.DataFrame({
            "grid_id": ["1_1", "1_2"], "pred": [2.0, 1.0],
            "center": ["삼산119안전센터"] * 2, "emd": ["삼산동"] * 2,
            "station": ["남부소방서"] * 2, "target_total": [10.0, 5.0],
            "n_hydrant": [0.0, 2.0],
        })
        docs = A.data_docs(panel, city_label="울산광역시", year=2021)
        text = " ".join(d.text for d in docs)
        self.assertIn("삼산119안전센터", text)
        self.assertIn("소화전 없는 격자", text)


class TestAnswer(unittest.TestCase):
    def setUp(self):
        self.index = A.BM25(A.law_docs(ARTICLES))
        from firebird.config import load_config
        self.cfg = load_config()

    def test_falls_back_to_search_when_llm_absent(self):
        with mock.patch.object(A.L, "is_available", return_value=False):
            r = A.answer(self.cfg, "화재예방강화지구 지정", self.index)
        self.assertEqual(r["source_backend"], "search_only")
        self.assertTrue(r["sources"])

    def test_no_match_is_reported_not_invented(self):
        r = A.answer(self.cfg, "김치찌개 조리법 알려줘", A.BM25([]), use_llm=False)
        self.assertEqual(r["source_backend"], "no_match")
        self.assertEqual(r["sources"], [])

    def test_prompt_forbids_invention(self):
        hits = self.index.search("화재예방강화지구", top_k=2)
        prompt = A.build_prompt("질문", hits)
        self.assertIn("지어내지", prompt)
        self.assertIn("조문 번호", prompt)

    def test_sources_carry_reference(self):
        with mock.patch.object(A.L, "is_available", return_value=False):
            r = A.answer(self.cfg, "화재안전조사 주기", self.index)
        self.assertTrue(all(s["ref"] for s in r["sources"]))


class TestLawParsing(unittest.TestCase):
    def test_norm_ignores_spaces(self):
        self.assertEqual(LW._norm("화재의 예방 및 안전관리에 관한 법률"),
                         LW._norm("화재의예방및안전관리에관한법률"))

    def test_clean_strips_cdata_and_tags(self):
        self.assertEqual(LW._clean("<a><![CDATA[제18조]]></a>"), "제18조")

    def test_search_rejects_non_exact_match(self):
        xml = ("<법령일련번호>111</법령일련번호>"
               "<법령명한글><![CDATA[소방시설 설치 및 관리에 관한 법률]]></법령명한글>")
        with mock.patch.object(LW.requests, "get") as g:
            g.return_value.text = xml
            g.return_value.raise_for_status = lambda: None
            got = LW.search_law("화재의 예방 및 안전관리에 관한 법률")
        self.assertIsNone(got, "다른 법을 이름만 맞는 척 가져오면 안 된다")

    def test_search_accepts_exact_match_ignoring_spaces(self):
        xml = ("<법령일련번호>222</법령일련번호>"
               "<법령명한글><![CDATA[소방기본법]]></법령명한글>")
        with mock.patch.object(LW.requests, "get") as g:
            g.return_value.text = xml
            g.return_value.raise_for_status = lambda: None
            got = LW.search_law("소방기본법")
        self.assertEqual(got["mst"], "222")


def make_route(n=3, depot="삼산119안전센터"):
    r = pd.DataFrame({
        "순번": range(1, n + 1),
        "grid_id": [f"g{i}" for i in range(n)],
        "emd": ["삼산동"] * n, "sgg": ["남구"] * n,
        "lon": np.linspace(129.3, 129.35, n), "lat": np.linspace(35.53, 35.56, n),
        "이동거리_m": [500.0] * n, "누적거리_m": np.cumsum([500.0] * n),
        "이동시간_분": [3.0] * n, "누적시간_분": np.cumsum([3.0] * n),
        "출동관서": [depot] * n,
    })
    r.attrs["depot"] = {"name": depot, "lon": 129.31, "lat": 35.54}
    return r


class TestPlans(unittest.TestCase):
    def setUp(self):
        routes = [make_route()]
        summary = pd.DataFrame([{"출동관서": "삼산119안전센터", "팀": 1, "회차": 1,
                                 "격자수": 3, "총_km": 3.0, "총_분": 15.0}])
        mp = pd.DataFrame([{"month": m, "위험계수": 1.0 + (m == 12) * 0.2,
                            "등급": "높음" if m == 12 else "보통"} for m in range(1, 13)])
        self.ctx = PL.PlanContext(
            city_label="울산광역시", year=2021, mode=PM.MODES["general"],
            targets=routes[0], routes=routes, summary=summary,
            distance_source="osrm", month_plan=mp,
            law_index=A.BM25(A.law_docs(ARTICLES)))

    def test_daily_plan_includes_route_and_checks(self):
        doc = PL.daily_plan(self.ctx, date(2026, 9, 15))
        md = PL.render(doc, self.ctx)
        self.assertIn("삼산119안전센터", md)
        self.assertIn("복귀", md)
        for chk in PM.MODES["general"].checks:
            self.assertIn(chk, md)

    def test_daily_plan_has_legal_basis(self):
        doc = PL.daily_plan(self.ctx, date(2026, 9, 15))
        self.assertTrue(doc["legal"], "법령 근거가 비어 있다")
        self.assertIn("법령 근거", PL.render(doc, self.ctx))

    def test_monthly_plan_uses_risk_multiplier(self):
        dec = PL.monthly_plan(self.ctx, 12)
        sep = PL.monthly_plan(self.ctx, 9)
        self.assertGreater(dec["risk_multiplier"], sep["risk_multiplier"])
        self.assertGreater(dec["total_rounds"], 0)

    def test_month_weeks_cover_the_month(self):
        weeks = PL.month_weeks(2026, 2)
        self.assertEqual(weeks[0][0], date(2026, 2, 1))
        self.assertEqual(weeks[-1][1], date(2026, 2, 28))

    def test_annual_plan_has_twelve_months_and_statutes(self):
        doc = PL.annual_plan(self.ctx)
        self.assertEqual(len(doc["calendar"]), 12)
        self.assertEqual(len(doc["quarters"]), 4)
        md = PL.render(doc, self.ctx)
        self.assertIn("법정 이행 사항", md)
        self.assertIn("연 1회 이상", md)

    def test_polish_keeps_text_when_llm_absent(self):
        from firebird.config import load_config
        md = "# 제목\n숫자 123"
        with mock.patch.object(PL.L, "is_available", return_value=False):
            out = PL.polish(load_config(), md)
        self.assertEqual(out["text"], md)
        self.assertFalse(out["polished"])

    def test_polish_rule_forbids_changing_numbers(self):
        self.assertIn("절대 바꾸지", PL.POLISH_RULE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
