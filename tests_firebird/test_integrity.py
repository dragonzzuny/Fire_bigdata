"""사실성 장치가 실제로 작동하는지 확인한다.

숫자를 지어내거나 조문을 바꾸는 것은 이 서비스에서 가장 큰 사고다.
막는 장치가 있다는 것만으로는 부족하고, 그 장치가 실제로 잡아내는지를
위조 입력으로 확인해야 한다.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firebird import assistant as A, plans as PLN     # noqa: E402


DOC = """# 2026년 3월 10일 예방순찰 계획(안)

울산소방서

1. 관련
   가. 「소방기본법」 제10조
   나. 「화재의 예방 및 안전관리에 관한 법률」 제7조

2. 위 호와 관련하여 아래와 같이 예방순찰을 실시하고자 합니다.

| 출동관서 | 구역 | 거리 |
|---|---|---|
| 삼산119안전센터 | 2331_3456 | 4.4 km |
| 성남119안전센터 | 2329_3460 | 6.0 km |

전년 대비 포착률 68.5% (95% 신뢰구간 63.5~72.9%)

붙임 1. 순찰 대상 목록 1부.  끝.
"""


class TestPlanFidelity(unittest.TestCase):
    """다듬은 계획서가 원문의 수·조문을 그대로 담는지."""

    def test_문체만_바꾼_문서는_통과한다(self):
        cand = DOC.replace("실시하고자 합니다", "시행하고자 합니다")
        self.assertTrue(PLN.check_fidelity(DOC, cand)["ok"])

    def test_숫자를_반올림하면_막는다(self):
        r = PLN.check_fidelity(DOC, DOC.replace("68.5%", "약 70%"))
        self.assertFalse(r["ok"])
        self.assertIn("70", r["added_numbers"])

    def test_조문_번호를_바꾸면_막는다(self):
        r = PLN.check_fidelity(DOC, DOC.replace("제7조", "제17조"))
        self.assertFalse(r["ok"])
        self.assertIn("제17조", r["added_legal"])
        self.assertIn("제7조", r["dropped_legal"])

    def test_법령명을_바꾸면_막는다(self):
        r = PLN.check_fidelity(DOC, DOC.replace("소방기본법", "소방시설법"))
        self.assertFalse(r["ok"])

    def test_없던_별표를_붙이면_막는다(self):
        r = PLN.check_fidelity(DOC, DOC + "\n※ 별표 4 참조\n")
        self.assertFalse(r["ok"])

    def test_격자_번호를_바꾸면_막는다(self):
        r = PLN.check_fidelity(DOC, DOC.replace("2331_3456", "2331_3999"))
        self.assertFalse(r["ok"])

    def test_표_행이_통째로_빠지면_막는다(self):
        cand = DOC.replace("| 성남119안전센터 | 2329_3460 | 6.0 km |\n", "")
        r = PLN.check_fidelity(DOC, cand)
        self.assertFalse(r["ok"])
        self.assertTrue(r["dropped_too_many"])

    def test_자릿점_표기_차이는_넘긴다(self):
        a = "총 1,234 m 이동"
        b = "총 1234m 이동"
        self.assertTrue(PLN.check_fidelity(a, b)["ok"])


class TestAnswerGrounding(unittest.TestCase):
    """업무 도우미 답변이 검색된 자료 밖의 조문을 인용하지 않는지."""

    def _hits(self):
        return [(A.Doc(
            doc_id="d1", source="법령",
            title="화재의 예방 및 안전관리에 관한 법률 제7조(화재안전조사)",
            ref="화재의 예방 및 안전관리에 관한 법률 제7조",
            text="소방관서장은 화재안전조사를 실시할 수 있다. 별표 2 를 따른다. "
                 "위반한 자는 같은 법 제50조제1항에 따라 처벌한다."), 9.0)]

    def test_자료_안의_조문은_통과한다(self):
        ans = ("「화재의 예방 및 안전관리에 관한 법률」 제7조에 따라 조사할 수 "
               "있으며, 별표 2 를 따릅니다.")
        self.assertTrue(A.check_grounding(ans, self._hits())["ok"])

    def test_괘선으로_끊긴_원문도_찾아낸다(self):
        """별표 본문은 표 괘선이 낱말 중간에 박혀 온다. 그래도 대조되어야 한다."""
        g = A.check_grounding("같은 법 제50조제1항에 따릅니다.", self._hits())
        self.assertTrue(g["ok"], g["unsupported"])

    def test_없는_조문을_인용하면_잡아낸다(self):
        g = A.check_grounding("제31조의2 에 따릅니다.", self._hits())
        self.assertFalse(g["ok"])
        self.assertIn("조:31의2", g["unsupported"])

    def test_없는_법령을_인용하면_잡아낸다(self):
        g = A.check_grounding("「소방시설 설치 및 관리에 관한 법률」 제7조입니다.",
                              self._hits())
        self.assertFalse(g["ok"])

    def test_없는_별표를_인용하면_잡아낸다(self):
        g = A.check_grounding("별표 9 를 보십시오.", self._hits())
        self.assertFalse(g["ok"])

    def test_모른다고_답하면_통과한다(self):
        g = A.check_grounding("제공된 자료에서 확인되지 않습니다.", self._hits())
        self.assertTrue(g["ok"])


class TestLawTextCleaning(unittest.TestCase):
    """법제처 별표 본문의 표 괘선 제거."""

    def test_낱말_중간_괘선은_지운다(self):
        from firebird.lawdata import _strip_rules
        self.assertIn("제50조제1항", _strip_rules("같은 법 제50│조제1항에 따라"))

    def test_칸_경계_괘선은_공백으로(self):
        from firebird.lawdata import _strip_rules
        out = _strip_rules("일자 │ 조치사항 │ 보완일자")
        self.assertNotIn("│", out)
        self.assertIn("일자", out)
        self.assertIn("조치사항", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
