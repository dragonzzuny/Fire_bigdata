#!/usr/bin/env python
"""발표자료·대본·영상이 서로 어긋나지 않는지 본다.

셋은 따로 고쳐진다. 장표를 한 장 늘리고 대본을 안 고치거나, 영상을 다시
뽑고 대본의 시각을 그대로 두면 발표장에서야 드러난다. 실제로 그런 일이
있었다. 장표 10(법정 서식)이 영상에도 대본에도 없어 통째로 빠져 있었다.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firebird.config import load_config  # noqa: E402

#: 영상이 대신하는 장표. 대본이 따로 부르지 않아도 된다.
COVERED_BY_VIDEO = {6, 7, 8, 9}


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def main() -> int:
    try:
        from pptx import Presentation
    except ImportError:
        print("python-pptx 가 없습니다.")
        return 1

    cfg = load_config()
    decks = sorted(cfg.paths.outputs.glob("불씨예보_발표자료_*.pptx"))
    script_p = ROOT / "docs" / "DEMO_SCRIPT.md"
    card_p = ROOT / "docs" / "REHEARSAL_CARD.md"
    video = cfg.paths.outputs / "demo_video" / "시연_핵심.mp4"
    if not decks or not script_p.exists():
        print("발표자료나 대본이 없습니다. scripts/07_deck.py 를 먼저 돌리십시오.")
        return 1

    n = len(Presentation(decks[0]).slides._sldIdLst)
    script = script_p.read_text(encoding="utf-8")
    card = card_p.read_text(encoding="utf-8") if card_p.exists() else ""
    head = script[:script.index("## 3. 상세 대본")]
    bad: list[str] = []

    if f"장표 {n}장" not in script:
        bad.append(f"대본의 장표 수 표기가 실제({n}장)와 다릅니다")
    for m in re.finditer(r"장표 (\d+)", script):
        if int(m.group(1)) > n:
            bad.append(f"대본이 없는 장표 {m.group(1)} 을 가리킵니다")

    covered = {int(x) for m in re.finditer(r"장표 ([\d·\s]+)", head)
               for x in re.findall(r"\d+", m.group(1))} | COVERED_BY_VIDEO
    missing = sorted(set(range(1, n + 1)) - covered)
    if missing:
        bad.append(f"발표에서 언급되지 않는 장표: {missing}")

    if video.exists():
        sec = duration(video)
        mm, ss = int(sec // 60), int(sec % 60)
        for name, text in (("대본", script), ("리허설 카드", card)):
            if text and f"{mm}분 {ss}초" not in text:
                bad.append(f"{name}의 영상 길이 표기가 실제({mm}분 {ss}초)와 다릅니다")
        try:
            ov = script[script.index("〈재생 ·"):script.index("나머지 구간은")]
            for m in re.finditer(r"\*\*(\d):(\d\d)\*\*", ov):
                t = int(m.group(1)) * 60 + int(m.group(2))
                if t > sec:
                    bad.append(f"얹어 말할 시각 {m.group(0)} 이 영상 길이를 넘습니다")
        except ValueError:
            bad.append("대본에서 영상 구간 표를 찾지 못했습니다")

        scenes = cfg.paths.outputs / "demo_video" / "장면.json"
        if scenes.exists():
            subs = [s["text"] for s in json.loads(scenes.read_text(encoding="utf-8"))]
            for need in ("119안전센터 출발", "같은 범위", "숫자와 법령"):
                if not any(need in t for t in subs):
                    bad.append(f"영상 자막에 '{need}' 가 없는데 대본이 그 자리를 가리킵니다")
        print(f"장표 {n}장 · 영상 {mm}분 {ss}초 · 대본 {len(script.splitlines())}줄\n")
    else:
        print(f"장표 {n}장 · 영상 없음 · 대본 {len(script.splitlines())}줄\n")

    # 문서가 가리키는 산출물 파일이 실제로 있는가.
    # 리허설 카드가 이름이 바뀐 옛 파일(시연_불씨예보.mp4)을 가리키고 있었다.
    import re as _re
    for name, text in (("대본", script), ("리허설 카드", card)):
        for m in _re.finditer(r"`?(outputs/[\w가-힣/_\.]+\.(?:mp4|json|pptx|pdf))`?", text):
            if not (ROOT / m.group(1)).exists():
                bad.append(f"{name}가 없는 파일을 가리킵니다: {m.group(1)}")
        for m in _re.finditer(r"`(scripts/[\w_]+\.py)`", text):
            if not (ROOT / m.group(1)).exists():
                bad.append(f"{name}가 없는 스크립트를 가리킵니다: {m.group(1)}")

    # 두 문서의 총 시간이 서로 다르면 하나는 낡은 것이다.
    tt = [_re.search(r"총 (\d+)분 (\d+)초", t) for t in (script, card)]
    got = [f"{m.group(1)}분 {m.group(2)}초" for m in tt if m]
    if len(got) == 2 and got[0] != got[1]:
        bad.append(f"대본과 리허설 카드의 총 시간이 다릅니다: {got[0]} vs {got[1]}")

    if bad:
        print(f"FAIL — 발표자료·대본·영상이 어긋납니다 ({len(bad)}건)")
        for b in bad:
            print("  ·", b)
        return 1
    print("PASS — 발표자료·대본·영상이 서로 맞습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
