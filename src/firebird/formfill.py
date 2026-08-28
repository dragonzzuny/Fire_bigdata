"""법정 서식을 **원본 파일 위에** 채운다.

서식을 HTML 로 다시 그리면 아무리 맞춰도 '비슷하게 만든 표'다. 괘선 두께,
글자 간격, 칸 너비가 조금씩 다르고 결재선에서 그게 먼저 보인다.

그래서 법제처가 배포하는 서식 PDF 를 그대로 배경에 깔고 값만 얹는다.
서식은 정부 원본 그 자체이고, 우리가 더하는 것은 글자뿐이다.

HWP 원본은 5.x 바이너리(OLE)라 파이썬으로 열어 고칠 수 없다. HWPX(zip+XML)를
배포하면 그때는 파일 자체를 채울 수 있다. 지금은 PDF 위에 얹는 것이
'원본과 완전히 같은 서식'을 얻는 유일한 길이다.

칸 위치는 손으로 재지 않는다. `pdftotext -bbox` 로 서식에 인쇄된 글자
(명칭·건물동수·개·㎡ …)의 실제 좌표를 읽어, 그 옆이나 단위 앞에 값을 놓는다.
서식이 개정돼 칸이 밀려도 좌표를 다시 읽어 따라간다.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: A4 서식 한 장. pdftotext 가 주는 좌표계(포인트)와 같다.
PAGE_W, PAGE_H = 595.0, 841.0


@dataclass(frozen=True)
class Slot:
    """값이 들어갈 자리.

    anchor  : 서식에 인쇄된 글자. 이 글자를 찾아 기준으로 삼는다.
    align   : "left"  = anchor 오른쪽에 이어 쓴다 (명칭, 위치 …)
              "right" = anchor 왼쪽에 붙여 쓴다 (단위 앞 숫자)
    nth     : 같은 글자가 여러 번 나올 때 몇 번째인가 (개·㎡·명 은 여러 개다)
    dx      : anchor 로부터의 여백(pt)
    dy      : 세로 보정(pt). 기본은 anchor 와 같은 줄.
    """
    anchor: str
    align: str = "left"
    nth: int = 0
    dx: float = 8.0
    dy: float = 0.0


#: 관리대장 [별지 제11호서식] 의 칸 배치.
#: 숫자 칸은 단위(개·㎡·명·㎞) 바로 왼쪽에 오른쪽 맞춤으로 놓는다 — 서식에
#: 인쇄된 단위를 기준으로 삼으면 칸 너비를 몰라도 정확히 들어간다.
LEDGER_SLOTS: dict[str, Slot] = {
    "명칭": Slot("명칭"),
    "위치": Slot("위치"),
    "대표자": Slot("대표자"),
    "전화번호": Slot("전화번호"),
    "지정일자": Slot("지정일자"),
    "건축연도": Slot("건축연도"),
    "연면적": Slot("㎡", "right", nth=0),
    "건축면적": Slot("㎡", "right", nth=1),
    "건물동수": Slot("개", "right", nth=0),
    "점포수": Slot("개", "right", nth=1),
    "유동인구": Slot("명", "right", nth=0),
    "상주인구": Slot("명", "right", nth=1),
    "소방조직": Slot("명", "right", nth=2),
    "건물구조": Slot("건물구조"),
    "지구면적": Slot("㎡", "right", nth=2),
    "소방시설:소화설비": Slot("개", "right", nth=2),
    "소방시설:경보설비": Slot("개", "right", nth=3),
    "소방시설:피난구조설비": Slot("개", "right", nth=4),
    "소방시설:소방용수": Slot("개", "right", nth=5),
    "소방시설:소화활동설비": Slot("개", "right", nth=6),
    "소방시설:기타설비": Slot("개", "right", nth=7),
    "소방관서거리:본서": Slot("㎞", "right", nth=0),
    "소방관서거리:관할119안전센터": Slot("㎞", "right", nth=1),
}

#: 가운뎃점(ㆍ)이 미리 찍혀 있는 서술 칸. 점 오른쪽에 한 줄씩 쓴다.
BULLET_FIELDS = ("지구특징", "취약요소", "도로여건", "현대화사업")

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def words_of(pdf: Path, page: int = 1) -> list[tuple[float, float, float, float, str]]:
    """서식에 인쇄된 글자와 그 좌표(pt). (xMin, yMin, xMax, yMax, 글자)"""
    xml = subprocess.run(
        ["pdftotext", "-bbox-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True, text=True, check=True).stdout
    return [(float(a), float(b), float(c), float(d), t) for a, b, c, d, t in
            re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" '
                       r'xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', xml)]


def _find(words, text: str, nth: int = 0):
    hits = sorted((w for w in words if w[4] == text), key=lambda w: (w[1], w[0]))
    return hits[nth] if nth < len(hits) else None


def rules_at(img, y_pt: float, page_w: float = PAGE_W,
             page_h: float = PAGE_H) -> list[float]:
    """그 줄을 가로지르는 세로 괘선 위치(pt).

    표 전체에서 괘선을 찾으면 맨 왼쪽 선 하나만 잡힌다 — 줄마다 칸이 달라
    끝까지 이어지는 선이 그것뿐이기 때문이다. 그래서 **해당 줄에서만** 찾는다.

    값을 라벨 바로 옆에 놓으면 라벨 칸을 침범한다. 값이 들어갈 칸이 어디서
    시작하는지는 그 줄의 괘선이 알려 준다. 서식이 개정돼 칸 너비가 바뀌어도
    다시 찾아내므로 좌표를 손으로 고칠 일이 없다.
    """
    import numpy as np

    a = np.asarray(img.convert("L"))
    sy = a.shape[0] / page_h
    sx = a.shape[1] / page_w
    y = int(round(y_pt * sy))
    half = max(2, int(round(2.0 * sy)))          # 줄 두께 ±2pt
    band = a[max(0, y - half):y + half + 1, :] < 150
    if band.size == 0:
        return []
    col = band.mean(axis=0)
    hit = np.where(col > 0.9)[0]                 # 그 구간 내내 어두운 열 = 괘선
    if hit.size == 0:
        return []
    groups, cur = [], [int(hit[0])]
    for i in hit[1:]:
        if i - cur[-1] <= 3:
            cur.append(int(i))
        else:
            groups.append(cur)
            cur = [int(i)]
    groups.append(cur)
    return [float(sum(g) / len(g)) / sx for g in groups]


def render_page(pdf: Path, out: Path, dpi: int = 200, page: int = 1) -> Path:
    """서식 한 장을 그림으로. 이 그림이 배경이 된다."""
    stem = out.with_suffix("")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", str(page),
                    "-l", str(page), str(pdf), str(stem)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for p in stem.parent.glob(stem.name + "-*.png"):
        p.rename(out)
        break
    return out


def fill(pdf: Path, values: dict[str, str], bullets: dict[str, list[str]],
         out_png: Path, *, dpi: int = 200, page: int = 1) -> Path:
    """원본 서식 위에 값을 얹어 그림으로 만든다.

    values : 칸 이름 -> 적을 값 (LEDGER_SLOTS 에 있는 이름만 쓰인다)
    bullets: 서술 칸 이름 -> 줄 목록 (가운뎃점 옆에 한 줄씩)
    """
    from PIL import Image, ImageDraw, ImageFont

    bg = render_page(pdf, out_png.with_name("_bg.png"), dpi=dpi, page=page)
    im = Image.open(bg).convert("RGB")
    s = im.width / PAGE_W                    # pt -> px
    d = ImageDraw.Draw(im)

    path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    font = ImageFont.truetype(path, int(round(8.6 * s))) if path else ImageFont.load_default()
    small = ImageFont.truetype(path, int(round(7.4 * s))) if path else font
    ink = (17, 17, 17)

    words = words_of(pdf, page)

    def cell_start(x_after: float, y_pt: float) -> float:
        """그 줄에서 x_after 오른쪽의 첫 괘선 = 값 칸의 왼쪽 끝."""
        nxt = [r for r in rules_at(im, y_pt) if r > x_after + 1.0]
        return (nxt[0] if nxt else x_after) + 5.0

    def put(text, x, y_top, y_bot, fnt, align="left"):
        if not text:
            return
        box = d.textbbox((0, 0), text, font=fnt)
        h = box[3] - box[1]
        cy = (y_top + y_bot) / 2 * s
        py = cy - h / 2 - box[1]
        px = x * s if align == "left" else x * s - (box[2] - box[0])
        d.text((px, py), text, font=fnt, fill=ink)

    for key, slot in LEDGER_SLOTS.items():
        val = str(values.get(key, "") or "").strip()
        if not val:
            continue
        a = _find(words, slot.anchor, slot.nth)
        if a is None:
            continue
        x0, y0, x1, y1, _ = a
        if slot.align == "left":
            put(val, cell_start(x1, (y0 + y1) / 2), y0 + slot.dy,
                y1 + slot.dy, font, "left")
        else:
            put(val, x0 - slot.dx, y0 + slot.dy, y1 + slot.dy, font, "right")

    dots = sorted((w for w in words if w[4] == "ㆍ"), key=lambda w: w[1])
    for name in BULLET_FIELDS:
        lab = _find(words, name)
        if lab is None:
            continue
        # 그 칸에 속한 가운뎃점 = 라벨 y 를 사이에 두고 가장 가까운 두 개
        near = sorted(dots, key=lambda w: abs((w[1] + w[3]) / 2 - (lab[1] + lab[3]) / 2))[:2]
        near = sorted(near, key=lambda w: w[1])
        for line, dot in zip(bullets.get(name, []), near):
            put(line, dot[2] + 4.0, dot[1], dot[3], small, "left")

    im.save(out_png)
    Path(bg).unlink(missing_ok=True)
    return out_png
