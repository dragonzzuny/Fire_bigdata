#!/usr/bin/env python
"""문서에 적힌 수치가 산출물과 같은지 대조한다.

README 와 발표 대본의 숫자는 손으로 적는다. 산출물이 바뀌어도 문서는 안 바뀌므로,
가만히 두면 반드시 어긋난다. 실제로 이 저장소의 README 에는 데이터를 다시 받아
전부 재현한 뒤에도 "재현 전까지 어떤 수치도 주장하지 않는다"가 한동안 남아 있었다.

여기서는 **문서에 그 표기가 있는지**가 아니라 **문서에 적힌 표기가 산출물과 같은지**를
본다. 산출물이 바뀌면 이 검사가 먼저 실패하고, 그때 문서를 고친다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firebird.config import load_config          # noqa: E402

DOCS = ("README.md", "docs/DEMO_SCRIPT.md", "docs/DATA_PROPOSAL.md")


def expected(cfg) -> dict[str, str]:
    """(설명, 산출물에서 계산한 표기). 문서에 이 표기 그대로 있어야 한다."""
    out = cfg.paths.outputs
    ev = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    city, year = ev.get("city", "ulsan"), ev["temporal"]["test_year"]
    s = json.loads((out / f"artifacts_summary_{city}_{year}.json").read_text(encoding="utf-8"))

    bt_path = out / f"backtest_patrol_{city}.json"
    bt = json.loads(bt_path.read_text(encoding="utf-8")) if bt_path.exists() else {}

    _lp = out / f"form_ledger_{city}_{cfg.holdout_year}.json"
    _ledger = json.loads(_lp.read_text(encoding="utf-8")) if _lp.exists() else {}

    t, a = ev["temporal"], s["allocation"]
    h, ci = t["headline"], t["ci"][f"top{int(t['headline_k'])}"]
    tk, op = a["top_k_percent"], a["optimized"]
    return {
        "상위 20% 포착률": f"{h['model_capture']:.1%}",
        "단순 기준 포착률": f"{h['baseline_capture']:.1%}",
        "신뢰구간 상한": f"{ci['model']['hi']:.1%}",
        "무작위 대비 배수": f"{h['model_lift']:.2f}배",
        "세종 이식": f"{ev['transfer']['headline']['model_capture']:.0%}",
        "누적화재 단일피처": f"{ev['single_feature_probe'][0]['capture_top20']:.1%}",
        "상위 20% 구역 수": f"{tk['n_grids_selected']}개",
        "상위 20% 점검 소요": f"{tk['cost_if_all']:,.0f}건",
        # 화면·CSV 에 나가는 배분(관할별 최소 배분 적용)이 문서에 적히는 수다.
        "배분 구역 수": f"{(a.get('equity') or {}).get('n_grids', op['n_grids'])}개",
        "배분 개선폭": f"{(a.get('equity') or a)['gain_pp']:+.1f}%p",
        "순찰 회고 기준선": f"{(bt.get('headline') or {}).get('baseline_fires', 0):,.0f}건",
        "순찰 회고 포착": f"{(bt.get('headline') or {}).get('capture_share', 0):.1%}",
        "순찰 회고 화재": f"{(bt.get('headline') or {}).get('model_fires', 0):,.0f}건",
        "순찰 회고 구역비중": f"{(bt.get('headline') or {}).get('share_of_city', 0):.1%}",
        "법정 서식 자동 입력": f"{_ledger.get('n_filled', 0)}개",
        "형평성 대가": f"{(a.get('equity') or {}).get('equity_cost_pp', 0):.1f}%p",
        "1위 구역 점검비용": f"{(a.get('top1_grid') or {}).get('inspection_cost', 0):,.0f}건",
        "소화전 사각 구역": f"{s['n_blind_spots']}개",
        "전체 구역 수": f"{s['n_grids']:,}개",
    }


def extra_expected(cfg) -> dict:
    """산출물 파일이 따로 있는 수치. 없으면 건너뛴다."""
    out = {}
    p = cfg.paths.outputs / "building_feature_eval_ulsan.json"
    if p.exists():
        b = json.loads(p.read_text(encoding="utf-8"))
        out["노후도 실측(전체)"] = f"{b['with_buildings_capture']:.1%}"
        out["노후도 차이"] = f"{b['delta_pp']:.1f}%p".lstrip("+")
    return out


#: (설명, 문장 꼴, 산출물 값을 꺼내는 함수).
#: '값이 어딘가 한 번 있으면 통과' 하는 검사로는 모순을 못 잡는다. 짧은 대본이
#: 13개, 상세 대본이 17개로 갈라져 있어도 '17개' 가 어딘가 있으니 통과했다.
#: 여기서는 문장 꼴에 걸리는 숫자를 **전부** 꺼내 산출물 값과 대조한다.
CONTRADICTIONS = [
    ("법정 서식 채운 칸",
     r"(\d+)개\s*칸\s*(?:중|가운데)\s*\**(\d+)개",
     lambda cfg, led: (str(led.get("n_fields", "")), str(led.get("n_filled", "")))),
]


def contradictions(cfg, texts: dict, ledger: dict) -> list:
    """문장 꼴에 걸리는 숫자가 산출물과 다른 곳을 모두 찾는다."""
    import re as _re
    bad = []
    for label, pat, pick in CONTRADICTIONS:
        want = pick(cfg, ledger)
        if not all(want):
            continue
        for doc, text in texts.items():
            for m in _re.finditer(pat, text):
                got = m.groups()
                if got != want:
                    line = text[:m.start()].count("\n") + 1
                    bad.append((label, f"{doc}:{line}",
                                "/".join(got), "/".join(want)))
    return bad


def main() -> int:
    cfg = load_config()
    root = Path(__file__).resolve().parents[1]
    try:
        want = expected(cfg)
        want.update(extra_expected(cfg))
    except (FileNotFoundError, KeyError) as exc:
        print(f"산출물을 읽지 못했습니다: {exc}")
        return 1

    texts = {}
    for d in DOCS:
        p = root / d
        if p.exists():
            texts[d] = p.read_text(encoding="utf-8")

    def norm(t: str) -> str:
        """빼기 기호를 하나로 맞춘다. 문서는 −(U+2212)를 쓰고 코드는 -를 쓴다.
        같은 값을 표기 차이로 불일치라고 말하면 검사가 무뎌진다."""
        return t.replace("\u2212", "-").replace("\u2013", "-")

    _lp = cfg.paths.outputs / f"form_ledger_ulsan_{cfg.holdout_year}.json"
    _led = json.loads(_lp.read_text(encoding="utf-8")) if _lp.exists() else {}
    clash = contradictions(cfg, texts, _led)

    missing = []
    for label, value in want.items():
        v = norm(value)
        where = [d for d, t in texts.items() if v in norm(t)]
        if not where:
            missing.append((label, value))

    print(f"산출물 기준 수치 {len(want)}종을 {len(texts)}개 문서와 대조")
    for d in texts:
        print(f"  · {d}")
    print()
    if missing:
        print(f"문서에서 찾지 못한 값 {len(missing)}종")
        print("  (산출물이 바뀌었는데 문서를 안 고쳤거나, 표기 방식이 다릅니다)")
        for label, value in missing:
            print(f"  - {label:22s} 산출물 값 '{value}'")
        print()
    if clash:
        print(f"산출물과 다른 값이 적힌 곳 {len(clash)}군데")
        for label, where, got, want_v in clash:
            print(f"  - {label:20s} {where}  문서 '{got}' ≠ 산출물 '{want_v}'")
        print()
    print("FAIL — 문서와 산출물이 어긋납니다." if (missing or clash)
          else "PASS — 문서의 수치가 모두 산출물과 일치합니다.")
    return 1 if (missing or clash) else 0


if __name__ == "__main__":
    raise SystemExit(main())
