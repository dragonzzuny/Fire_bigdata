"""LLM 백엔드 선택과 폴백. 네트워크·CLI 없이 검증한다."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from firebird.config import load_config          # noqa: E402
from firebird import llm as L                     # noqa: E402


def cfg_with(backends, **llm_over):
    base = load_config()
    raw = {**base.raw}
    raw["llm"] = {**raw["llm"], "backends": backends, **llm_over}
    return replace(base, raw=raw)


SAMPLE = dict(
    grid_id="1_1",
    risk={"score_0_100": 90.0, "rank": 3, "percentile": 1.5},
    drivers=[{"label": "다중이용업소 수", "value": 12, "contribution": 0.5}],
    checklist={"sections": [{"구분": "업종:일반음식점", "근거": "주방 화기",
                             "항목": ["주방 자동소화장치 작동 상태"]}]},
)


class TestBackendSelection(unittest.TestCase):
    def test_rule_based_is_always_available(self):
        cfg = cfg_with(["rule_based"])
        self.assertEqual(L.available_backends(cfg), ["rule_based"])
        self.assertFalse(L.is_available(cfg))     # 진짜 LLM 은 없다

    def test_missing_cli_binary_is_skipped(self):
        cfg = cfg_with(["cli", "rule_based"],
                       cli={"command": ["존재하지않는명령어xyz"], "preamble": ""})
        self.assertNotIn("cli", L.available_backends(cfg))

    def test_api_needs_key_env(self):
        cfg = cfg_with(["api", "rule_based"],
                       api={"endpoint": "http://x", "model": "m", "key_env": "NO_SUCH_KEY_ENV"})
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertFalse(L.api_available(cfg))


class TestFallbackChain(unittest.TestCase):
    def test_falls_back_to_rules_when_every_backend_fails(self):
        cfg = cfg_with(["cli", "rule_based"],
                       cli={"command": ["존재하지않는명령어xyz"], "preamble": ""})
        plan = L.inspection_plan(cfg, **SAMPLE)
        self.assertEqual(plan["source"], "rule_based")
        self.assertIn("점검계획서", plan["text"])
        self.assertIn("유의사항", plan["text"])

    def test_uses_cli_when_it_answers(self):
        cfg = cfg_with(["cli", "rule_based"], cli={"command": ["fakecli"], "preamble": ""})
        with mock.patch.object(L, "cli_available", return_value=True), \
             mock.patch.object(L, "_call_cli", return_value=("CLI 초안 본문", "cli:fakecli")):
            plan = L.inspection_plan(cfg, **SAMPLE)
        self.assertEqual(plan["source"], "cli:fakecli")
        self.assertEqual(plan["text"], "CLI 초안 본문")

    def test_second_backend_used_when_first_returns_empty(self):
        cfg = cfg_with(["cli", "ollama", "rule_based"],
                       cli={"command": ["fakecli"], "preamble": ""})
        with mock.patch.object(L, "cli_available", return_value=True), \
             mock.patch.object(L, "ollama_available", return_value=True), \
             mock.patch.object(L, "_call_cli", return_value=("", "cli_empty")), \
             mock.patch.object(L, "_call_ollama", return_value=("로컬 초안", "ollama:x")):
            plan = L.inspection_plan(cfg, **SAMPLE)
        self.assertEqual(plan["source"], "ollama:x")
        self.assertIn("cli_empty", plan["tried"])   # 무엇이 실패했는지 남는다

    def test_use_llm_false_skips_everything(self):
        cfg = cfg_with(["cli", "rule_based"], cli={"command": ["fakecli"], "preamble": ""})
        with mock.patch.object(L, "_call_cli") as called:
            plan = L.inspection_plan(cfg, **SAMPLE, use_llm=False)
        called.assert_not_called()
        self.assertEqual(plan["source"], "rule_based")


class TestPromptGrounding(unittest.TestCase):
    def test_prompt_carries_every_checklist_item(self):
        p = L.build_prompt(SAMPLE["grid_id"], SAMPLE["risk"],
                           SAMPLE["drivers"], SAMPLE["checklist"])
        self.assertIn("주방 자동소화장치 작동 상태", p)
        self.assertIn("다중이용업소 수", p)
        self.assertIn("새로 만들지 마라", p)      # 환각 억제 지시가 실제로 들어간다

    def test_cli_argv_appends_prompt_when_no_placeholder(self):
        cfg = cfg_with(["cli"], cli={"command": ["echo", "-n"], "preamble": "PRE:"})
        text, source = L._call_cli(cfg, "PROMPT")
        self.assertTrue(text.endswith("PROMPT"))
        self.assertIn("PRE:", text)
        self.assertEqual(source, "cli:echo")

    def test_cli_placeholder_substitution(self):
        cfg = cfg_with(["cli"], cli={"command": ["echo", "-n", "[{prompt}]"], "preamble": ""})
        text, _ = L._call_cli(cfg, "XYZ")
        self.assertEqual(text, "[XYZ]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
