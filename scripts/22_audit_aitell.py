#!/usr/bin/env python
"""AI 가 쓴 티가 나는 표현을 찾는다.

심사위원이 'AI 가 써 준 것' 이라고 느끼는 순간 내용과 무관하게 점수가 깎인다.
그 판단은 논리가 아니라 문장의 결에서 온다 — 번역투, 기계적 열거, 균일한
문장 길이, 남발된 볼드.

분류는 epoko77-ai/im-not-ai 의 패턴표를 따랐다.
  S1 결정적 — 한 번만 나와도 티가 난다. 무조건 고친다.
  S2 강함   — 한두 번은 괜찮고, 세 번부터 문제다.

발표에서 실제로 말하는 글과 심사위원이 읽는 글만 본다. 코드 주석과
내부 문서는 대상이 아니다 — 거기서는 이 표현들이 문제가 아니다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (등급, 패턴, 무엇이 문제인가, 어떻게 고치나)
RULES: list[tuple[str, str, str, str]] = [
    # ---- S1: 한 번만 나와도 ----
    ("S1", r"결론적으로", "AI 관용구", "그냥 결론을 말한다"),
    ("S1", r"시사하는 바가?\s*(크|큽)", "AI 관용구", "무엇을 뜻하는지 직접 쓴다"),
    ("S1", r"바탕으로 하여", "번역투", "'~로'"),
    ("S1", r"(되어지|하여지|불려지|쓰여지)", "이중 피동", "'된다·한다·불린다·쓰인다'"),
    # 연결어미 뒤 쉼표(C-11)는 넣었다가 뺐다. '대조하고, 없으면' 같은
    # 멀쩡한 한국어를 18곳이나 잡았다. 정확한 조건을 모르는 채로 두면
    # 검사가 틀린 것을 가리키고, 그러면 맞는 지적까지 안 믿게 된다.
    ("S1", r"함에 있어", "번역투", "'~할 때'"),
    ("S1", r"~?에 다름 아니", "번역투", "'~이다'"),
    # ---- S2: 세 번부터 ----
    ("S2", r"를? 통해(서)?\s", "번역투 '~를 통해'", "'~로'"),
    ("S2", r"에 있어서?\s", "번역투 '~에 있어'", "'~에서·~할 때'"),
    ("S2", r"에 대한\s", "번역투 '~에 대한'", "'~의' 또는 생략"),
    ("S2", r"(혁신적|획기적|비약적|괄목할)", "과장 어휘", "무엇이 달라지는지 쓴다"),
    ("S2", r"주목할 만", "AI 관용구", "왜 봐야 하는지 쓴다"),
    ("S2", r"할 필요가 있다", "권고형 형식명사", "'~한다'"),
    ("S2", r"(매우|정말|굉장히|상당히)\s", "강조 부사", "숫자로 바꾼다"),
    ("S2", r"할 수 있을 것으로 (보인다|기대된다|예상된다)", "다중 완곡",
     "'~한다' 또는 '~인지는 재지 않았다'"),
    ("S2", r"^\s*(또한|따라서|즉|나아가|그리고)\s", "문두 접속사", "빼거나 문장을 잇는다"),
    ("S2", r"(극대화|최적화|고도화|활성화|내재화)(하|한|를|가|에|,|\.|$)",
     "'~화' 명사화", "동사로 푼다"),
]

#: 검사할 곳. 발표에서 말하는 글과 심사위원이 읽는 글.
TARGETS = [
    ROOT / "docs" / "DEMO_SCRIPT.md",
    ROOT / "docs" / "REHEARSAL_CARD.md",
    ROOT / "docs" / "DATA_PROPOSAL.md",
    ROOT / "README.md",
]

#: 검사에서 뺄 줄. 규칙표 자신과 코드 블록은 예시를 싣고 있다.
SKIP = re.compile(r"^\s*(```|\||>?\s*`|#|-{3,})")


def deck_text() -> list[tuple[str, int, str]]:
    """장표 안의 글. 심사위원이 읽는 글이라 같은 잣대를 댄다."""
    try:
        from pptx import Presentation
    except ImportError:
        return []
    out = []
    for f in sorted((ROOT / "outputs").glob("불씨예보_발표자료_*.pptx")):
        prs = Presentation(str(f))
        for i, sl in enumerate(prs.slides, 1):
            for sh in sl.shapes:
                if sh.has_text_frame:
                    for para in sh.text_frame.paragraphs:
                        t = "".join(r.text for r in para.runs).strip()
                        if t:
                            out.append((f"{f.name} 장표 {i}", 0, t))
    return out


def lines() -> list[tuple[str, int, str]]:
    out = []
    for p in TARGETS:
        if not p.exists():
            continue
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if ln.strip() and not SKIP.match(ln):
                out.append((str(p.relative_to(ROOT)), i, ln))
    return out + deck_text()


def bold_density(path: Path) -> tuple[int, int]:
    """볼드가 장식으로 쓰이는가.

    대본의 인용문(> 로 시작하는 줄)에서 굵게는 장식이 아니라 '힘주어 읽는
    자리' 표시다. 문서가 스스로 그렇게 정의하고 있으므로 세지 않는다.
    설명하는 산문에서만 센다 — 거기서 굵게가 잦으면 장식이다.
    """
    if not path.exists():
        return 0, 0
    text = path.read_text(encoding="utf-8")
    paras = []
    for para in text.split("\n\n"):
        body = [ln for ln in para.splitlines()
                if ln.strip()
                and not ln.lstrip().startswith((">", "|", "#", "-", "*", "·"))
                and not re.match(r"\s*\d+[.)]\s", ln)]
        if body:
            paras.append("\n".join(body))
    heavy = sum(1 for p in paras if len(re.findall(r"\*\*[^*]+\*\*", p)) >= 3)
    return heavy, len(paras)


def main() -> int:
    hits: dict[str, list] = {"S1": [], "S2": []}
    for where, no, ln in lines():
        for grade, pat, what, how in RULES:
            for m in re.finditer(pat, ln):
                hits[grade].append((where, no, what, how, m.group(0).strip(), ln.strip()))

    print(f"검사한 곳 {len(TARGETS)}개 문서 + 장표\n")
    bad = False
    for grade, title in (("S1", "S1 — 한 번만 나와도 티가 난다"),
                         ("S2", "S2 — 세 번부터 문제다")):
        got = hits[grade]
        if not got:
            print(f"  {title}: 없음")
            continue
        by_what: dict[str, list] = {}
        for h in got:
            by_what.setdefault(h[2], []).append(h)
        print(f"  {title}")
        for what, items in sorted(by_what.items(), key=lambda kv: -len(kv[1])):
            over = grade == "S1" or len(items) >= 3
            mark = "✗" if over else "·"
            bad = bad or over
            print(f"    {mark} {what} {len(items)}회 — {items[0][3]}")
            for w, n, _, _, frag, ln in items[:3]:
                loc = f"{w}:{n}" if n else w
                print(f"        {loc}  …{frag}…")
            if len(items) > 3:
                print(f"        (그 밖에 {len(items) - 3}곳)")
        print()

    for p in TARGETS:
        heavy, total = bold_density(p)
        if total and heavy / total > 0.25:
            bad = True
            print(f"  ✗ 볼드 과다 — {p.name}: {total}문단 중 {heavy}문단이 "
                  f"한 문단에 굵게 3회 이상")

    print("\n" + ("FAIL — 고칠 곳이 있습니다." if bad
                  else "PASS — S1 없음, S2 도 반복되지 않습니다."))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
