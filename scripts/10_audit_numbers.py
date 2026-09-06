#!/usr/bin/env python
"""발표자료의 모든 숫자가 실제 산출물에서 나온 것인지 증명한다.

숫자를 '어디선가 본 적 있는 값 목록'과 맞춰 보는 방식은 검증이 아니다.
두 자리 숫자는 그런 목록을 우연히 90% 통과한다.

그래서 **바꿔치기 시험**을 한다.
  1) 진짜 산출물로 장표를 만들어 숫자를 모은다.               (집합 A)
  2) 산출물의 모든 수를 흔든 뒤 같은 코드로 다시 만든다.       (집합 B)
  3) A 에 있는데 B 에도 그대로 있는 숫자는 산출물에서 나온 것이 아니다.

산출물에서 나온 숫자라면 산출물이 바뀔 때 반드시 따라 바뀐다. 안 바뀌었다면
코드에 손으로 적힌 값이며, 그런 값은 출처를 밝힌 목록(LITERALS)에 있어야 한다.
목록에도 없으면 FAIL 이다.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from firebird.config import load_config          # noqa: E402

import importlib.util                            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "deck", Path(__file__).resolve().parent / "07_deck.py")
deck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deck)

#: 흔들 배수. 자릿수가 달라지도록 크게, 그러나 표가 깨지지 않을 만큼만.
SHAKE = 1.4173

#: 어느 문서에나 나오는 값이라 대조가 무의미한 것. 최소한으로 둔다.
TRIVIAL = {"1", "2", "3", "4", "5", "6", "7", "9", "119"}

#: 코드에 손으로 적은 값과 그 출처. 여기 없는 손 상수가 장표에 있으면 FAIL.
LITERALS = {
    # 2024년 +5명 은 통계연보 원본을 이 환경에서 대조하지 못해 뺐다.
    # 대신 확인 가능한 2차 출처가 있는 값을 쓴다.
    "913": "한국일보 2026-01-18 — 2026년 소방공무원 정원 65,860 → 66,773명. "
           "기사는 '2022년 이후 4년 만의 증원' 이라고 적고 있다.",
    "8": "2025 소방청 통계연보(2024년 기준) — “특정소방대상물 가운데 "
         "고층건축물(30층 이상)은 전년 대비 484개소(8.04%) 증가했고”",
    "484": "위와 같음 — 30층 이상 고층건축물 증가 개소",
    "2022": "한국일보 2026-01-18 — 소방공무원 정원의 직전 확대 연도",
    "2024": "소방청 통계연보 기준 연도",
    # 아래 다섯 값은 논문 원문(arXiv:1602.09067 / KDD 2016)에서 직접 확인했다.
    "2016": "Madaio et al., Firebird, KDD 2016 (arXiv:1602.09067)",
    "5000": "KDD 2016 초록 — “fire risk scores for over 5,000 buildings”",
    "2573": "KDD 2016 §4 — 애틀랜타의 기존 점검 대상 2,573개소",
    "19397": "KDD 2016 §4 — 기존 2,573개소에 더해 새로 찾아낸 점검 후보 "
             "19,397개소(‘점검 대상 총수’가 아니다)",
    "6096": "KDD 2016 §4 — 위 19,397개를 추려 권고한 6,096개소",
    "237": "KDD 2016 §6.3 — “Increasing the number of annual inspections by "
           "6,096 (237%) overnight was not feasible without significant changes "
           "in organizational processes, local ordinances, or increased staffing.”",
    "70": "KDD 2016 §5 — TPR 71.36% @ FPR 20%. 장표에는 ‘70% 이상’으로 낮춰 적었다",
    "71": "KDD 2016 §5 — true positive rate 71.36%",
    "20": "KDD 2016 §5 — false positive rate 20% · 상위 20% 구간",
    "10": "상위 10% 구간 표기",
    "2026": "발표 연도",
    "95": "신뢰구간 수준 95%",
    "1000": "부트스트랩 반복 1,000회 (configs/config.yaml n_bootstrap)",
    "1.35": "도로거리 미확보 시 우회계수 (src/firebird/routing.py DETOUR_FACTOR)",
    "0": "표기용 0",
    "5.0": "위와 같음",
    "0.5": "표기용",
    "30": "2025 소방청 통계연보 — 고층건축물 기준 층수(30층 이상)",
    "100": "‘도달 가능한 최선을 100으로 볼 때’ 라는 설명 표현",
    "11": "별지 제11호서식 (src/firebird/forms.py FORM_NO)",
    "09": "스크립트 파일명 scripts/09_forms.py",
    "27": "별지 제11호서식의 기입 칸 수 (src/firebird/forms.py zone_ledger)",
}


def shake(obj, factor: float = None, bump: int = 7):
    """자료 구조 안의 모든 수를 흔든다. 문자열 속 숫자까지 바꾼다.

    한 번만 흔들면 흔든 값이 진짜 값과 우연히 같아질 수 있다. 실제로
    순찰 구역 15개가 흔들려 119안전센터 28개와 겹쳤다. 그래서 서로 다른
    계수로 두 번 흔들고, 두 번 다 살아남은 수만 손 상수로 본다.
    """
    factor = SHAKE if factor is None else factor
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: shake(v, factor, bump) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(shake(v, factor, bump) for v in obj)
    if isinstance(obj, int):
        return int(round(obj * factor)) + bump
    if isinstance(obj, float):
        return obj * factor + 0.137
    if isinstance(obj, str):
        return re.sub(r"\d[\d,]*(?:\.\d+)?",
                      lambda m: _shake_token(m.group(), factor, bump), obj)
    return obj


def _shake_token(tok: str, factor: float = None, bump: int = 7) -> str:
    factor = SHAKE if factor is None else factor
    raw = tok.replace(",", "")
    try:
        v = float(raw)
    except ValueError:
        return tok
    out = v * factor + bump
    if "." in raw:
        return f"{out:.{len(raw.split('.')[1])}f}"
    return f"{int(round(out)):,}" if "," in tok else str(int(round(out)))


def numbers_in(prs) -> dict[str, list[tuple[int, str]]]:
    """장표에 나온 숫자 -> [(장 번호, 그 줄)]"""
    out: dict[str, list[tuple[int, str]]] = {}
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    texts += [c.text for c in row.cells]
        for t in texts:
            for line in str(t).splitlines():
                for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", line):
                    out.setdefault(m.group().replace(",", ""), []).append(
                        (i, line.strip()))
    return out


def main() -> int:
    cfg = load_config()
    ev_path = cfg.paths.outputs / "evaluation.json"
    if not ev_path.exists():
        print("evaluation.json 이 없습니다. scripts/04_train_eval.py 를 먼저 실행하십시오.")
        return 1
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    city, year = ev.get("city", "ulsan"), ev["temporal"]["test_year"]

    def load(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    summary = load(cfg.paths.outputs / f"artifacts_summary_{city}_{year}.json")
    manifest = load(cfg.paths.processed / f"manifest_{city}.json")
    figs = cfg.paths.figures
    ds_rows = deck.dataset_rows(cfg)
    extra = deck.collect_extra(cfg, city)

    real = numbers_in(deck.build(cfg, ev, summary, manifest, figs,
                                 ds_rows, extra))
    def shaken(factor, bump):
        return numbers_in(deck.build(
            cfg, shake(copy.deepcopy(ev), factor, bump),
            shake(copy.deepcopy(summary), factor, bump),
            shake(copy.deepcopy(manifest), factor, bump), figs,
            shake(copy.deepcopy(ds_rows), factor, bump),
            shake(copy.deepcopy(extra), factor, bump)))

    fakes = [shaken(SHAKE, 7), shaken(SHAKE * 2.3 + 0.41, 13)]

    derived, hardcoded, unsourced = [], [], []
    for n, spots in sorted(real.items(), key=lambda kv: -len(kv[1])):
        if any(n not in f for f in fakes):
            derived.append(n)                      # 산출물이 바뀌자 따라 바뀜
        elif n in TRIVIAL or n in LITERALS:
            hardcoded.append(n)
        else:
            unsourced.append((n, spots[0]))

    print(f"발표자료 숫자 {len(real)}종")
    print(f"  산출물에서 계산된 값        {len(derived):3d}종")
    print(f"  출처를 밝힌 손 상수         {len(hardcoded):3d}종")
    print(f"  출처 미확인                 {len(unsourced):3d}종\n")
    if unsourced:
        for n, (slide, line) in unsourced:
            print(f"  [{slide:2d}장] {n:>10s}   {line[:86]}")
        print()
    print("FAIL — 산출물에서 나오지도 않고 출처도 없는 숫자가 있습니다." if unsourced
          else "PASS — 모든 숫자가 산출물에서 계산되었거나 출처가 밝혀져 있습니다.")
    return 1 if unsourced else 0


if __name__ == "__main__":
    raise SystemExit(main())
