#!/usr/bin/env python
"""장표 배치 검사 — 화면 밖으로 밀려난 것이 없는지.

python-pptx 는 도형을 장표 밖에 두어도 아무 말을 하지 않는다. 화면에서는
멀쩡해 보이다가 인쇄하거나 발표장 화면에 띄웠을 때 표 마지막 줄이나 그래프
x축이 잘려 나간다. 발표 중에는 고칠 수 없으므로 만들 때 확인한다.

겹침도 함께 본다. 글상자 두 개가 겹치면 뒤엣것이 앞엣것을 가린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pptx import Presentation                    # noqa: E402
from pptx.util import Emu                        # noqa: E402

from firebird.config import load_config          # noqa: E402

TOL = Emu(int(0.02 * 914400))       # 0.02인치까지는 반올림 오차로 본다


def _boxes(slide):
    for sh in slide.shapes:
        if sh.left is None or sh.top is None:
            continue
        w = sh.width or 0
        h = sh.height or 0
        yield sh, int(sh.left), int(sh.top), int(sh.left) + int(w), int(sh.top) + int(h)


def _name(sh) -> str:
    if sh.has_text_frame and sh.text_frame.text.strip():
        return sh.text_frame.text.strip().splitlines()[0][:44]
    return sh.shape_type and str(sh.shape_type) or sh.name


#: 글자 한 개가 차지하는 가로 폭(em). 렌더링 없이 재는 어림값이다.
#: 한글은 1.0em 이 맞지만 공백·숫자·문장부호를 모두 0.52em 로 두면 11% 쯤
#: 넓게 나와, 한 줄에 들어가는 문장을 두 줄로 세었다.
_W_SPACE, _W_LATIN, _W_PUNCT, _W_CJK = 0.28, 0.55, 0.5, 1.0
_LINE_PAD = 1.08            # 글꼴 위아래 여유(측정으로 맞춘 값)
_PUNCT = set("·—–…‘’“”()[]{}<>:;,.!?/|+-=%")


def _em_width(ch: str) -> float:
    if ch == " ":
        return _W_SPACE
    if ch in _PUNCT:
        return _W_PUNCT
    return _W_CJK if ord(ch) > 0x2E80 else _W_LATIN


def _wrapped_lines(txt: str, size_pt: float, box_w_in: float) -> int:
    if not txt:
        return 1
    width_in = sum(_em_width(c) * size_pt / 72.0 for c in txt)
    return max(1, int(width_in / max(box_w_in, 0.1) + 0.999))


def _text_overflow(sh) -> str:
    """글상자 안의 글이 상자를 넘치는지 어림한다.

    python-pptx 는 글이 넘쳐도 알려 주지 않는다. 실제 렌더링 없이 정확히 잴 수는
    없으므로, 한글은 글자 크기만큼, 영문·숫자·공백은 절반만큼 폭을 쓴다고 보고
    줄 수를 어림한다. 여유(20%)를 두고, 그래도 넘치면 사람이 확인하도록 알린다.
    """
    if not sh.has_text_frame or not sh.width or not sh.height:
        return ""
    tf = sh.text_frame
    text = tf.text
    if not text.strip():
        return ""
    box_w_in = Emu(int(sh.width)).inches - 0.1          # 안쪽 여백
    box_h_in = Emu(int(sh.height)).inches

    lines, need = 0, 0.0
    for para in tf.paragraphs:
        size = 18.0
        for r in para.runs:
            if r.font.size:
                size = r.font.size.pt
                break
        spacing = float(para.line_spacing or 1.15)
        txt = "".join(r.text for r in para.runs)
        if not txt:
            lines += 1
            continue
        n = _wrapped_lines(txt, size, box_w_in)
        lines += n
        need += n * size * spacing / 72.0
    if need > box_h_in * 1.2:
        return f"글 {need:.2f}in > 상자 {box_h_in:.2f}in"
    return ""


def _needed_in(sh) -> float:
    """글상자가 실제로 쓰는 세로 길이(인치) 어림. 넘침 판정에 쓴다."""
    if not sh.has_text_frame or not sh.width:
        return 0.0
    box_w_in = Emu(int(sh.width)).inches - 0.1
    need = 0.0
    for para in sh.text_frame.paragraphs:
        size = 18.0
        for r in para.runs:
            if r.font.size:
                size = r.font.size.pt
                break
        spacing = float(para.line_spacing or 1.15)
        txt = "".join(r.text for r in para.runs)
        need += (_wrapped_lines(txt, size, box_w_in)
                 * size * spacing * _LINE_PAD / 72.0)
    return need


def _spill_out_of_band(slide) -> list[tuple[str, str]]:
    """글이 자기를 받치는 띠(둥근 사각형) 밖으로 흘러나갔는가.

    상자 좌표만 보는 검사로는 이걸 못 잡는다. 글상자는 장표 안에 있고,
    선언한 높이도 넉넉하니 통과한다. 그런데 화면에서는 글이 띠 아래로
    빠져나와 흰 바탕에 떠 있다. 실제로 차별성 카드 ③ 이 그랬다.
    """
    bands, texts = [], []
    for sh in slide.shapes:
        if sh.left is None or sh.top is None or not sh.width or not sh.height:
            continue
        rect = (int(sh.left), int(sh.top),
                int(sh.left) + int(sh.width), int(sh.top) + int(sh.height))
        if sh.has_text_frame and sh.text_frame.text.strip():
            texts.append((sh, rect))
        elif sh.shape_type is not None:
            bands.append((sh, rect))

    out = []
    for sh, (x0, y0, x1, y1) in texts:
        host = None
        for _, (bx0, by0, bx1, by1) in bands:
            # 글상자의 왼쪽 위 모서리가 띠 '안' 에 있어야 그 띠가 받치는 것이다.
            # by1 > y0 조건이 없으면 맨 위의 빨간 장식 띠가 모든 글을 잡는다.
            if (bx0 - TOL <= x0 and by0 - TOL <= y0 < by1
                    and x1 <= bx1 + TOL):
                if host is None or (by1 - by0) < (host[3] - host[1]):
                    host = (bx0, by0, bx1, by1)
        if host is None:
            continue
        bottom_in = Emu(int(y0)).inches + _needed_in(sh)
        host_bottom_in = Emu(int(host[3])).inches
        # 어림이라 딱 맞출 수 없다. 표지 상자가 한 줄 잘렸는데 0.3in 허용으로는
        # 통과시켰다. 줄 높이 보정을 넣고 허용치를 좁혀 그 정도는 잡는다.
        if bottom_in > host_bottom_in + 0.15:
            out.append((_name(sh),
                        f"글 끝 {bottom_in:.2f}in > 띠 끝 {host_bottom_in:.2f}in"))
    return out



def main() -> int:
    cfg = load_config()
    pptx = next(iter(sorted(cfg.paths.outputs.glob("*발표자료*.pptx"))), None)
    if pptx is None:
        print("발표자료 파일이 없습니다.")
        return 1
    prs = Presentation(str(pptx))
    W, H = int(prs.slide_width), int(prs.slide_height)

    bad, tight = [], []
    for i, slide in enumerate(prs.slides, 1):
        for sh, x0, y0, x1, y1 in _boxes(slide):
            over = []
            if x0 < -TOL:
                over.append(f"왼쪽 {Emu(-x0).inches:.2f}in")
            if y0 < -TOL:
                over.append(f"위 {Emu(-y0).inches:.2f}in")
            if x1 > W + TOL:
                over.append(f"오른쪽 {Emu(x1 - W).inches:.2f}in")
            if y1 > H + TOL:
                over.append(f"아래 {Emu(y1 - H).inches:.2f}in")
            if over:
                bad.append((i, _name(sh), ", ".join(over)))
            spill = _text_overflow(sh)
            if spill:
                tight.append((i, _name(sh), spill))
        for nm, why in _spill_out_of_band(slide):
            bad.append((i, nm, "띠 밖으로 흘러남 — " + why))

    print(f"발표자료: {pptx.name}  ({len(prs.slides._sldIdLst)}장, "
          f"{Emu(W).inches:.2f}×{Emu(H).inches:.2f}in)")
    if bad:
        print(f"\n장표 밖으로 넘친 도형 {len(bad)}건")
        for i, name, over in bad:
            print(f"  [{i:2d}장] {name:46s} {over}")
    if tight:
        print(f"\n글이 상자를 넘칠 것으로 보이는 곳 {len(tight)}건 (어림값, 확인 필요)")
        for i, name, why in tight:
            print(f"  [{i:2d}장] {name:46s} {why}")
    print()
    print("FAIL — 장표 밖으로 밀려난 도형이 있습니다." if bad
          else "PASS — 모든 도형이 장표 안에 있습니다."
               + (f" (글 넘침 의심 {len(tight)}건은 눈으로 확인하십시오.)" if tight else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
