#!/usr/bin/env python
"""대본을 소리 내어 읽는 데 걸리는 시간을 잰다.

시간표를 손으로 적어 두었더니 대본을 고칠 때마다 어긋났다. 실제로 §1 표는
합계 9분 42초라고 적혀 있는데 행을 더하면 8분 52초였다 — 50초 차이다.
10분 제한에서 여유가 18초인지 68초인지는 무대에서 판단을 바꾼다.

그래서 표를 대본에서 잰다. 세는 방법은 하나뿐이다.
  · 말하는 글자    : 330자/분 (리허설로 맞춘 속도)
  · 읽기 표시       : / 0.5초 · // 1.0초 · /// 2.0초
  · 세지 않는 것    : 〈동작·시선〉, 굵게 표시, 장표 지시, 표, 각주

--write 를 주면 §1 시간표와 리허설 카드의 누적 시각을 잰 값으로 고쳐 쓴다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "DEMO_SCRIPT.md"
CARD = ROOT / "docs" / "REHEARSAL_CARD.md"

CPM = 330.0                      # 분당 글자. 리허설에서 맞춘 속도.
PAUSE = {"///": 2.0, "//": 1.0, "/": 0.5}

#: §2 의 블록 머리. 예: **⑦ 검증 · 1분 40초 · 장표 12·13·14**
HEAD = re.compile(r"^\*\*([①-⑪])\s*([^·*]+?)\s*·\s*([^·*]+?)\s*·\s*([^*]+?)\*\*\s*$")
#: 영상 블록은 길이가 고정이라 따로 잡는다.
VIDEO = re.compile(r"^\*\*⑥\s*시연\s*·\s*(\d+)분\s*(\d+)초")


def _groups(lines: list[str]) -> list[list[str]]:
    """이어지는 인용 줄을 한 덩어리로 묶는다.

    대본의 한 문단은 여러 줄에 걸쳐 있고, 따옴표는 첫 줄과 끝 줄에만 있다.
    줄 단위로 보면 가운데 줄이 통째로 빠진다 — 처음 이 도구를 쓸 때
    ② 배경이 47자로 나온 이유가 그것이다.
    """
    out, cur = [], []
    for ln in lines:
        if ln.lstrip().startswith(">"):
            cur.append(ln.lstrip()[1:].strip())
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def spoken_seconds(lines: list[str]) -> tuple[float, int, dict]:
    """인용문 안의 말만 골라 초를 낸다."""
    chars, pauses = 0, {"/": 0, "//": 0, "///": 0}
    for g in _groups(lines):
        body = " ".join(g)
        if '"' not in body:
            continue                      # 지시문·주석 덩어리는 말이 아니다
        for s in g:
            s = re.sub(r"〈[^〉]*〉", "", s)   # 동작·시선은 소리를 내지 않는다
            for mark in ("///", "//", "/"):
                pauses[mark] += len(re.findall(r"(?<!/)" + re.escape(mark) + r"(?!/)", s))
            s = s.replace("/", " ")
            s = re.sub(r"\*+", "", s).replace('"', "")
            chars += len(re.sub(r"\s", "", s))
    secs = chars / CPM * 60.0 + sum(PAUSE[m] * n for m, n in pauses.items())
    return secs, chars, pauses


#: 한 블록 안에서 장표가 넘어가는 자리. 리허설 카드는 장표 단위라 여기서 쪼갠다.
TURN = re.compile(r"〈장표\s*(\d+)|영상이 끝나면")


def split_by_slide(chunk: list[str]) -> list[list[str]]:
    """블록을 장표가 넘어가는 자리에서 쪼갠다."""
    parts, cur = [], []
    for ln in chunk:
        if TURN.search(ln) and cur:
            parts.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        parts.append(cur)
    return parts


def blocks() -> list[dict]:
    text = SCRIPT.read_text(encoding="utf-8")
    body = text.split("## 2. 실제로 말할 대본", 1)[1].split("## 3. 상세 대본", 1)[0]
    lines = body.splitlines()
    heads = [i for i, ln in enumerate(lines) if HEAD.match(ln) or VIDEO.match(ln)]
    out = []
    for j, i in enumerate(heads):
        end = heads[j + 1] if j + 1 < len(heads) else len(lines)
        chunk = lines[i:end]
        m = VIDEO.match(lines[i])
        if m:
            secs = int(m.group(1)) * 60 + int(m.group(2))
            extra, _, _ = spoken_seconds(chunk)   # 앞뒤로 얹는 말
            parts = [spoken_seconds(c)[0] for c in split_by_slide(chunk)]
            parts[0] += secs           # 첫 조각에 영상 길이를 얹는다
            out.append({"head": lines[i], "kind": "video", "parts": parts,
                        "secs": secs + extra, "video": secs, "talk": extra})
            continue
        h = HEAD.match(lines[i])
        secs, chars, pauses = spoken_seconds(chunk)
        parts = [spoken_seconds(c)[0] for c in split_by_slide(chunk)]
        out.append({"head": lines[i], "kind": "talk", "no": h.group(1), "parts": parts,
                    "name": h.group(2), "stated": h.group(3), "slide": h.group(4),
                    "secs": secs, "chars": chars, "pauses": pauses})
    return out


def mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpm", type=float, default=CPM)
    ap.add_argument("--write", action="store_true",
                    help="잰 값으로 대본 §1 표와 리허설 카드 누적 시각을 고쳐 쓴다")
    args = ap.parse_args()
    globals()["CPM"] = args.cpm

    bs = blocks()
    total = 0.0
    print(f"분당 {CPM:.0f}자 기준\n")
    print(f"  {'구간':<22}{'잰 값':>7}{'대본 표기':>10}{'글자':>7}  쉼")
    for b in bs:
        total += b["secs"]
        if b["kind"] == "video":
            print(f"  {'⑥ 시연 (영상)':<21}{mmss(b['secs']):>8}"
                  f"{'':>10}{'':>7}  영상 {mmss(b['video'])} + 말 {b['talk']:.0f}초")
            continue
        p = b["pauses"]
        gap = ""
        stated = re.match(r"(?:(\d+)분\s*)?(\d+)초", b["stated"])
        if stated:
            sv = int(stated.group(1) or 0) * 60 + int(stated.group(2))
            d = b["secs"] - sv
            gap = f"{d:+.0f}초" if abs(d) >= 3 else ""
        print(f"  {b['no']} {b['name']:<19}{mmss(b['secs']):>8}"
              f"{b['stated']:>10}{b['chars']:>7}  "
              f"/{p['/']} //{p['//']} ///{p['///']}  {gap}")
    print(f"\n  {'합계':<22}{mmss(total):>8}     여유 {600 - total:+.0f}초")
    if total > 600:
        print("\n  ※ 10분을 넘는다. 「밀릴 때 버리는 순서」에서 위부터 뺄 것.")
    if args.write:
        write_back(bs, total)
        print("\n  대본 §1 표와 리허설 카드 누적 시각을 잰 값으로 고쳤다.")
    return 0 if total <= 600 else 1


#: §1 표의 구간 이름 → 그 구간을 이루는 블록 번호.
#: 표는 심사위원이 아니라 발표자가 보는 것이라 장표 묶음 단위로 적는다.
TABLE_ROWS = [
    ("인사", "1", ["①"]),
    ("배경 및 문제점", "2", ["②"]),
    ("**결과물 먼저 보이기**", "**3**", ["③"]),
    ("제안 내용", "4", ["④"]),
    ("활용 데이터", "5", ["⑤"]),
    ("**영상 재생 + 법정 서식 · 업무 도우미**", "**6~9 대체 · 10 · 11**", ["⑥"]),
    ("검증 결과", "12 · 13 · 14", ["⑦"]),
    ("차별성", "15 · 16", ["⑧"]),
    ("기대효과", "17", ["⑨"]),
    ("한계", "18", ["⑩"]),
    ("마무리", "19", ["⑪"]),
]


def write_back(bs: list[dict], total: float) -> None:
    """잰 값을 대본 §1 표와 리허설 카드에 도로 써넣는다."""
    by_no = {b.get("no", "⑥"): b for b in bs}

    rows = ["| 구간 | 시간 | 장표 |", "|---|---|---|"]
    for name, slide, nos in TABLE_ROWS:
        secs = sum(by_no[k]["secs"] for k in nos if k in by_no)
        t = mmss(secs)
        rows.append(f"| {name} | {'**' + t + '**' if name.startswith('**') else t} | {slide} |")
    rows.append(f"| **합계** | **{mmss(total)}** | 여유 {600 - total:+.0f}초 |")
    table = "\n".join(rows)

    txt = SCRIPT.read_text(encoding="utf-8")
    head = f"## 1. 시간 배분 (장표 19장 · 총 {mmss(total)})"
    a = txt.index("## 1. 시간 배분")
    b = txt.index("| 구간 | 시간 | 장표 |")
    c = txt.index("\n\n", txt.index("| **합계**", b))
    txt = (txt[:a] + head + txt[txt.index("\n", a):b] + table + txt[c:])
    txt = re.sub(r"## 2\. 실제로 말할 대본 \([^)]*\)",
                 f"## 2. 실제로 말할 대본 ({mmss(total)})", txt, count=1)
    # 읽는 속도별 표도 손으로 적혀 있었다. 말과 영상을 나눠 다시 낸다.
    vid = next((b["video"] for b in bs if b["kind"] == "video"), 0.0)
    speech = total - vid
    rate_rows = ["| 빠르기 | 말 + 쉼 | 영상 | 합계 |", "|---|---|---|---|"]
    for cpm, note in ((300, "또박또박"), (int(CPM), "보통"), (360, "조금 빠르게")):
        sp = speech * CPM / cpm
        tot = sp + vid
        mark = " ✔" if tot <= 600 else ""
        cells = (f"분당 {cpm}자 ({note})", f"{sp/60:.1f}분",
                 f"{int(vid)//60}:{int(vid)%60:02d}", f"{tot/60:.1f}분{mark}")
        if cpm == int(CPM):
            cells = tuple(f"**{c}**" for c in cells)
        rate_rows.append("| " + " | ".join(cells) + " |")
    a2 = txt.index("| 빠르기 | 말 + 쉼 | 영상 | 합계 |")
    b2 = txt.index("\n\n", txt.index("\n", txt.index("| 분당 360자", a2)))
    txt = txt[:a2] + "\n".join(rate_rows) + txt[b2:]
    txt = re.sub(r"> \*\*분당 \d+자로 \d+분 \d+초\.\*\* \d+초가 남습니다\.",
                 f"> **분당 {int(CPM)}자로 {int(round(total))//60}분 "
                 f"{int(round(total))%60}초.** {600 - int(round(total))}초가 남습니다.",
                 txt, count=1)

    # 블록 머리의 '· 20초 ·' 도 손으로 적힌 값이라 같이 어긋난다. 함께 고친다.
    for b in bs:
        if b["kind"] != "talk":
            continue
        secs = int(round(b["secs"]))
        t = (f"{secs // 60}분 {secs % 60}초" if secs >= 60 else f"{secs}초")
        old_head = b["head"]
        new_head = old_head.replace(f"· {b['stated']} ·", f"· {t} ·", 1)
        if new_head != old_head:
            txt = txt.replace(old_head, new_head, 1)
            # 상세 대본의 같은 제목(### ① 인사 (20초) — 장표 1)도 맞춘다
            txt = re.sub(rf"(### {re.escape(b['no'])} [^(\n]*)\([^)]*초\)",
                         rf"\1({t})", txt, count=1)
    SCRIPT.write_text(txt, encoding="utf-8")

    # 리허설 카드의 누적 시각. 카드는 장표 단위라 블록 안을 쪼개 쓴다.
    #   (카드 행 이름, 블록 번호, 그 블록의 몇 번째 조각인가)
    #: ALL = 그 블록의 조각을 전부 (한 행이 블록 하나를 통째로 맡는 자리)
    ALL = "all"
    CARD_ROWS = [
        ("1", "①", ALL), ("2", "②", ALL), ("**3**", "③", ALL),
        ("4", "④", ALL), ("5", "⑤", ALL),
        ("**영상**", "⑥", [0]), ("10·11", "⑥", "rest"),
        ("12", "⑦", [0]), ("13", "⑦", [1]), ("**14**", "⑦", "rest"),
        ("15·16", "⑧", ALL), ("17", "⑨", ALL),
        ("18", "⑩", ALL), ("19", "⑪", ALL),
    ]
    cum, stamps, used = 0.0, {}, {}
    for label, no, idxs in CARD_ROWS:
        parts = by_no[no]["parts"]
        if idxs == ALL:
            take = range(len(parts))
        elif idxs == "rest":
            take = range(used.get(no, 0), len(parts))
        else:
            take = idxs
        take = [i for i in take if i < len(parts)]
        used[no] = max(used.get(no, 0), max(take) + 1) if take else used.get(no, 0)
        cum += sum(parts[i] for i in take)
        stamps[label] = mmss(cum)
    assert abs(cum - total) < 1.5, f"카드 합계 {cum:.0f} 와 표 합계 {total:.0f} 가 다르다"

    card = CARD.read_text(encoding="utf-8")
    lines = []
    for ln in card.splitlines():
        m = re.match(r"^(\|\s*(?:\*\*)?[^|]+?(?:\*\*)?\s*)\|\s*(?:\*\*)?\d+:\d\d(?:\*\*)?\s*\|(.*)$", ln)
        if m:
            key = m.group(1).strip("| ").strip()
            if key in stamps:
                t = f"**{stamps[key]}**" if key.startswith("**") else stamps[key]
                ln = f"{m.group(1)}| {t} |{m.group(2)}"
        lines.append(ln)
    card = "\n".join(lines) + "\n"
    card = re.sub(r"총 \d+분 \d+초", f"총 {int(round(total))//60}분 {int(round(total))%60:02d}초",
                  card, count=1)
    CARD.write_text(card, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
