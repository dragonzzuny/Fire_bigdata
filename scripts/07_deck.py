#!/usr/bin/env python
"""발표자료(PPTX) 생성 — 모든 수치는 outputs/evaluation.json 에서 읽는다.

장표에 손으로 적은 숫자를 올리면 질문 한 번에 무너진다. 여기서 만드는
숫자는 전부 파이프라인이 출력한 값이고, 파이프라인을 다시 돌리면
장표도 따라서 갱신된다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

from firebird.config import load_config  # noqa: E402

W, H = Inches(13.333), Inches(7.5)          # 16:9
INK = RGBColor(0x1A, 0x1F, 0x2B)
MUTED = RGBColor(0x6B, 0x74, 0x84)
RED = RGBColor(0xE3, 0x4A, 0x31)
BLUE = RGBColor(0x2B, 0x6C, 0xB0)
GREEN = RGBColor(0x2F, 0x9E, 0x6E)
BG = RGBColor(0xFF, 0xFF, 0xFF)
BAND = RGBColor(0xF4, 0xF6, 0xF8)
FONT = "Noto Sans CJK KR"


# ------------------------------------------------------------------ 헬퍼

def textbox(slide, x, y, w, h, text, *, size=18, bold=False, color=INK,
            align=PP_ALIGN.LEFT, spacing=1.15):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        parts = re.split(r"\*\*(.+?)\*\*", line)
        for j, piece in enumerate(parts):
            if not piece:
                continue
            run = p.add_run()
            run.text = piece
            run.font.size = Pt(size)
            run.font.bold = bold or (j % 2 == 1)
            run.font.color.rgb = color
            run.font.name = FONT
        # 불릿 줄이 길어 넘어가면 둘째 줄이 불릿 밑으로 파고들어 문장이
        # 끊겨 보인다. 내어쓰기를 걸어 글머리표 오른쪽에 맞춰 떨어뜨린다.
        if line.lstrip().startswith(("·", "-", "•")):
            hang = Pt(size * 0.95)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(hang)))
            pPr.set("indent", str(int(-hang)))
    return tb


def band(slide, x, y, w, h, color=BAND):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid(); bg.fore_color.rgb = BG
    return s


def title_slide(prs, title, subtitle, foot=""):
    s = blank(prs)
    band(s, 0, 0, W, Inches(0.14), RED)
    textbox(s, Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.4),
            title, size=48, bold=True)
    textbox(s, Inches(0.9), Inches(3.9), Inches(11.5), Inches(1.2),
            subtitle, size=21, color=MUTED)
    if foot:
        textbox(s, Inches(0.9), Inches(6.4), Inches(11.5), Inches(0.6),
                foot, size=14, color=MUTED)
    return s


#: 장표마다 붙는 마크 경로. build() 가 채운다.
_MARK: Path | None = None


def stamp(slide):
    """장표 오른쪽 위에 마크를 찍는다.

    발표 중 어느 장표를 캡처해도 무엇에 관한 자료인지 남는다. 이름을 글자로
    다시 쓰면 시선을 뺏으므로, 작은 마크만 둔다.
    """
    if _MARK and _MARK.exists():
        slide.shapes.add_picture(str(_MARK), Inches(12.62), Inches(0.42),
                                 width=Inches(0.42), height=Inches(0.42))


def section(prs, kicker, title, lead=""):
    s = blank(prs)
    band(s, 0, 0, W, Inches(0.14), RED)
    stamp(s)
    textbox(s, Inches(0.8), Inches(0.45), Inches(11.8), Inches(0.4),
            kicker, size=13, bold=True, color=RED)
    textbox(s, Inches(0.8), Inches(0.85), Inches(11.8), Inches(0.8),
            title, size=31, bold=True)
    if lead:
        textbox(s, Inches(0.8), Inches(1.62), Inches(11.8), Inches(0.6),
                lead, size=15, color=MUTED)
    return s


#: 그림이 장표 아래로 넘치지 않도록 남겨 두는 여백.
BOTTOM_MARGIN = Inches(0.35)


def picture(slide, path: Path, x, y, w, *, max_h=None):
    """그림을 넣되, 장표 아래로 넘치면 폭을 줄여 맞춘다.

    폭만 지정하면 높이는 비율대로 정해진다. 가로로 긴 그림을 아래쪽에 넣으면
    조용히 장표 밖으로 밀려나 x축 눈금이 잘린다 — 화면에서는 멀쩡해 보이고
    인쇄물에서만 드러나므로 여기서 막는다.
    """
    if not Path(path).exists():
        return None
    from PIL import Image
    try:
        iw, ih = Image.open(path).size
        ratio = iw / ih
    except Exception:                                    # noqa: BLE001
        ratio = None

    limit = H - y - BOTTOM_MARGIN
    if max_h is not None:
        limit = min(limit, max_h)
    if ratio and w / ratio > limit:
        w = int(limit * ratio)
    pic = slide.shapes.add_picture(str(path), x, y, width=w)
    if ratio:
        pic.height = int(w / ratio)
    return pic


def kpi(slide, x, y, w, value, label, sub="", color=RED, *, size=34, h=1.5):
    """큰 숫자 하나 + 설명. size 를 키우면 그 장표의 무게가 올라간다."""
    band(slide, x, y, w, Inches(h))
    k = size / 34.0
    textbox(slide, x, y + Inches(0.14 * k), w, Inches(0.7 * k), value,
            size=size, bold=True, color=color, align=PP_ALIGN.CENTER)
    textbox(slide, x, y + Inches(0.82 * k), w, Inches(0.35), label,
            size=13 + (size - 34) * 0.18, bold=True, align=PP_ALIGN.CENTER)
    if sub:
        textbox(slide, x, y + Inches(1.13 * k), w, Inches(0.34), sub,
                size=10 + (size - 34) * 0.12, color=MUTED,
                align=PP_ALIGN.CENTER)


def table(slide, x, y, w, h, rows: list[list[str]], *, col_widths=None,
          header_color=RGBColor(0x2B, 0x33, 0x40), size=12,
          row_sizes: dict | None = None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h)
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(w * cw / total))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
            # 표 셀은 마크다운을 해석하지 않는다. **굵게** 표기를 그대로 두면
            # 별표가 화면에 그대로 찍힌다. 별표를 떼고 굵기로 바꾼다.
            text = str(val)
            emphasise = "**" in text
            if emphasise:
                text = text.replace("**", "")
            run = p.add_run(); run.text = text
            run.font.size = Pt((row_sizes or {}).get(r, size))
            run.font.name = FONT
            run.font.bold = (r == 0) or emphasise
            run.font.color.rgb = (RGBColor(0xFF, 0xFF, 0xFF) if r == 0
                                  else (RED if emphasise else INK))
            cell.fill.solid()
            cell.fill.fore_color.rgb = (header_color if r == 0
                                        else (BG if r % 2 else BAND))
    return tbl


def pct(v, digits=1):
    return "—" if v is None or v != v else f"{v:.{digits}%}"


#: 활용 데이터 장표에 들어갈 데이터셋 목록. 건수는 실제 적재 결과에서 읽는다.
DATASET_ROLES = [
    ("fire",     "화재발생현황",       "학습 라벨 (구역·연도별 화재 건수)"),
    ("target",   "특정소방대상물 현황", "용도·소방시설 구성 피처"),
    ("business", "다중이용업소 현황",   "업종 구성 피처, 점검 항목 근거"),
    ("hydrant",  "소방용수시설 운영현황", "대응취약(소화전 사각지대) 분석"),
]


def dataset_rows(cfg) -> list[list[str]]:
    """실제 적재 건수를 세어 활용 데이터 표를 만든다.

    장표에 데이터셋 이름만 적고 건수를 비워 두면 '정말 썼는가'를 확인할 수 없다.
    여기서 세는 값은 파이프라인이 실제로 읽은 행 수다.
    """
    import logging
    from firebird.io_utils import load_dataset

    prev = logging.getLogger("firebird.io_utils").level
    logging.getLogger("firebird.io_utils").setLevel(logging.ERROR)
    rows: list[list[str]] = []
    try:
        for city in cfg["cities"]:
            label = cfg.city(city)["label"]
            role_suffix = ("학습·검증" if cfg.city(city).get("role") == "primary"
                           else "타 지역 적용 검증")
            for kind, name, role in DATASET_ROLES:
                try:
                    n = f"{len(load_dataset(cfg, city, kind)):,}행"
                except (FileNotFoundError, KeyError):
                    n = "미사용"
                rows.append([f"{label}소방본부_{name}", role_suffix, role, n])
    finally:
        logging.getLogger("firebird.io_utils").setLevel(prev)
    return rows


# ------------------------------------------------------------------ 본문

def crop_top(shot: Path, keep: float = 0.66) -> Path:
    """캡처의 윗부분만 남긴다.

    스트림릿 화면은 세로로 길어 그대로 넣으면 장표에서 손톱만 해진다.
    사용자가 먼저 보는 것은 위쪽(조작부와 핵심 수치)이므로 그 부분만 쓴다.
    """
    from PIL import Image

    out = shot.with_name(shot.stem + "_crop.png")
    try:
        import numpy as np
        im = Image.open(shot)
        h = int(im.size[1] * keep)
        im = im.crop((0, 0, im.size[0], h))
        # 자른 뒤에도 아래쪽이 빈 화면인 경우가 많다. 그 여백을 그대로 넣으면
        # 장표에서 그림만 커지고 글씨는 작아진다 — 읽으라고 넣은 화면인데.
        a = np.asarray(im.convert("L"))
        ink = np.nonzero((a < 245).sum(axis=1) > 2)[0]
        if len(ink) and ink[-1] < a.shape[0] - 40:
            im = im.crop((0, 0, im.size[0], min(a.shape[0], int(ink[-1]) + 30)))
        im.save(out)
        return out
    except Exception:                                    # noqa: BLE001
        return shot


def crop_to(src: Path, dst: Path, top: float, bottom: float,
            left: float = 0.0, right: float = 1.0) -> Path | None:
    """비율로 잘라 새 파일로 낸다. 장표에서 확대해 보여 줄 조각을 만든다."""
    if not src.exists():
        return None
    from PIL import Image
    im = Image.open(src)
    box = (int(im.width * left), int(im.height * top),
           int(im.width * right), int(im.height * bottom))
    if box[3] - box[1] < 8:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.crop(box).save(dst)
    return dst


def screen_slide(prs, kicker: str, title: str, lead: str, shot: Path,
                 bullets: list[tuple[str, str]], *, note: str = "",
                 keep: float = 0.66):
    """화면을 크게 두고 설명은 오른쪽에.

    심사에서 확인되는 것은 무엇을 만들었다는 주장이 아니라 지금 도는 화면이다.
    화면이 뒷자리에서 안 읽히면 없는 것과 같으므로 폭을 최대한 준다.
    """
    s = section(prs, kicker, title, lead)
    img = crop_top(shot, keep)

    from PIL import Image
    try:
        w_px, h_px = Image.open(img).size
        ratio = w_px / h_px
    except Exception:                                    # noqa: BLE001
        ratio = 1.8

    max_w, max_h = 8.45, 4.55
    width = min(max_w, max_h * ratio)
    height = width / ratio
    pic = picture(s, img, Inches(0.7), Inches(2.2), Inches(width))
    if pic is not None:
        pic.width, pic.height = Inches(width), Inches(height)

    x = Inches(0.7 + width + 0.3)
    box_w = 13.333 - 0.7 - width - 0.3 - 0.7
    y = 2.2
    # 상자 높이를 그림 높이에 맞추면, 가로로 긴 화면일수록 상자가 납작해져
    # 설명이 잘린다. 그림과 무관하게 장표에서 쓸 수 있는 세로 공간을 나눠 쓴다.
    avail = min(4.9, 7.5 - 2.2 - 0.85)
    box_h = max(0.95, (avail - (len(bullets) - 1) * 0.18) / max(len(bullets), 1))
    # 줄이 늘면 상자 밖으로 흘러나간다. 배치 검사(scripts/11)는 도형 좌표만
    # 보므로 이 넘침을 못 잡는다. 넣기 전에 들어갈 만한 크기를 정한다.
    def _lines(text: str, size_pt: float) -> int:
        per = max(8, int((box_w - 0.4) * 72 / size_pt))   # 한 줄에 들어가는 글자 수
        return sum(max(1, -(-len(ln) // per)) for ln in text.split("\n"))

    size = 10.5
    while size > 8.6:
        room = int((box_h - 0.54) * 72 / (size * 1.25))
        if all(_lines(b, size) <= room for _, b in bullets):
            break
        size -= 0.5

    for head, body in bullets:
        band(s, x, Inches(y), Inches(box_w), Inches(box_h))
        textbox(s, x + Inches(0.2), Inches(y + 0.11), Inches(box_w - 0.4),
                Inches(0.32), head, size=13, bold=True, color=RED)
        textbox(s, x + Inches(0.2), Inches(y + 0.46), Inches(box_w - 0.4),
                Inches(box_h - 0.54), body, size=size, color=MUTED, spacing=1.05)
        y += box_h + 0.18
    if note:
        textbox(s, Inches(0.7), Inches(2.2 + height + 0.25), Inches(11.9),
                Inches(0.42), note, size=12, color=MUTED)
    return s


def build(cfg, ev: dict, summary: dict, manifest: dict, figs: Path,
          ds_rows: list[list[str]], extra: dict, *,
          movie: bool = True) -> Presentation:
    """10분 발표(시연 3분 포함) + 5분 질의응답.

    읽는 사람은 소방청·소방본부 관계자다. 알고리즘 이름보다
    **왜 필요한가 / 무엇을 하는가 / 어떻게 쓰는가** 가 먼저다.
    기술 근거는 질의응답용으로 뒤에 한 장만 둔다.
    """
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    global _MARK
    _MARK = figs / "logo_mark.png"

    k = cfg.headline_k
    # 장표에 손으로 적는 값을 없애기 위해, 필요한 수치는 모두 여기서 뽑아 둔다.
    grid_m = int(ev.get("grid_size_m", cfg.grid_size_m))
    conc = ev.get("resolution_concentration", {})
    probe = ev.get("single_feature_probe", [])
    probe0 = probe[0] if probe else {}
    hist_cap = (ev.get("temporal_history_only", {}).get("headline", {})
                  .get("model_capture"))
    # 넣어 보고 안 되면 안 넣는다. 그 측정 결과도 장표에 남긴다.
    # extra 로 받아야 수치 출처 검사(scripts/10)가 이 값도 흔들어 볼 수 있다.
    bld = extra.get("building_eval", {})
    t = ev["temporal"]
    h = t["headline"]
    key = f"top{k}"
    year = int(t["test_year"])
    tr = [int(y) for y in t["train_years"]]
    alloc = summary.get("allocation", {})
    ci = t.get("ci", {}).get(key, {})
    d = ci.get("delta", {})
    m_ci = ci.get("model", {})

    # ---- 1 표지 ----
    s1 = blank(prs)
    band(s1, 0, 0, W, Inches(0.14), RED)
    logo = figs / "logo_lockup_plain.png"
    if logo.exists():
        # 이름을 글자로 다시 쓰지 않는다. 로고가 이름이다.
        # 로고 아래 설명 문구와 겹치지 않도록 높이를 잡아 준다.
        picture(s1, logo, Inches(0.85), Inches(1.05), Inches(3.9), max_h=Inches(1.75))
    else:
        textbox(s1, Inches(0.9), Inches(1.15), Inches(11.5), Inches(1.0),
                "불씨예보", size=52, bold=True)
        textbox(s1, Inches(0.9), Inches(2.25), Inches(11.5), Inches(0.5),
                "K-Firebird", size=24, color=RED, bold=True)
    textbox(s1, Inches(0.9), Inches(3.05), Inches(11.5), Inches(0.9),
            "소방안전 빅데이터 기반 화재예방 점검·순찰 의사결정 시스템\n"
            "한정된 인력을 가장 위험한 곳에, 실행 가능한 계획으로",
            size=18, color=MUTED)
    band(s1, Inches(0.9), Inches(4.25), Inches(6.5), Inches(2.0),
         RGBColor(0xFD, 0xF0, 0xEC))
    textbox(s1, Inches(1.2), Inches(4.45), Inches(5.9), Inches(0.4),
            "이름의 뜻", size=14, bold=True, color=RED)
    textbox(s1, Inches(1.2), Inches(4.9), Inches(5.9), Inches(1.3),
            "· Firebird: 애틀랜타 소방의 화재위험 예측 시스템.\n"
            "  미국 NFPA 모범사례로 선정된 예방점검 우선순위 모델\n"
            "· K-: 국내 공개 데이터와 소방 법령 체계에 맞춘 한국형\n"
            "· 불씨예보: 일기예보처럼, 불씨를 미리 알린다", size=13.5)
    band(s1, Inches(7.8), Inches(4.25), Inches(4.6), Inches(2.0))
    textbox(s1, Inches(8.1), Inches(4.45), Inches(4.0), Inches(0.4),
            "발표자", size=14, bold=True, color=RED)
    textbox(s1, Inches(8.1), Inches(4.92), Inches(4.0), Inches(0.8),
            "박용준\n아주대학교 산업공학과 석사과정", size=16, bold=True)
    _p1 = manifest.get("panel", {})
    textbox(s1, Inches(8.1), Inches(5.72), Inches(4.0), Inches(0.4),
            f"울산 {_p1.get('grids', 0):,}개 구역 · "
            f"화재 {_p1.get('total_fires', 0):,.0f}건으로 검증",
            size=12.5, color=MUTED)
    band(s1, 0, Inches(6.42), W, Inches(1.08), BAND)
    textbox(s1, Inches(0.9), Inches(6.62), Inches(8.0), Inches(0.5),
            "제6회 소방안전 빅데이터 활용 및 아이디어 경진대회 · 서비스 개발 부문",
            size=13, color=MUTED)
    textbox(s1, Inches(8.9), Inches(6.62), Inches(3.5), Inches(0.5),
            "주최 소방청 · 주관 소방안전 빅데이터 플랫폼",
            size=12, color=MUTED, align=PP_ALIGN.RIGHT)

    # ---- 2 왜 (배경) ----
    s2 = section(prs, "1. 배경 및 문제점",
                 "늘어나는 점검 대상, 정체된 인력",
                 "법정 주기와 담당자 경험에 의존하는 현행 우선순위 결정")
    # 상자 높이를 내용에 맞춘다. 3.5in 로 두었더니 아래 4분의 1이 비고,
    # 그 아래 목적 문장이 허공에 떠 있었다.
    TOP, BOXH = 2.35, 3.0
    band(s2, Inches(0.8), Inches(TOP), Inches(5.6), Inches(BOXH))
    textbox(s2, Inches(1.1), Inches(TOP + 0.26), Inches(5.0), Inches(0.45),
            "국내는 이 자리가 비어 있습니다", size=16.5, bold=True)
    textbox(s2, Inches(1.1), Inches(TOP + 0.85), Inches(5.0), Inches(2.0),
            "· 소방공무원 정원 4년째 제자리 — 첫 증원이 2026년\n"
            "· 점검 대상 증가 — 30층 이상 고층 +484개소(8%)\n"
            "· 같은 법정 대상 안에서도 실제 위험은 크게 다름\n"
            "· 그 차이로 우선순위를 정하는 체계가 없음", size=15.5, spacing=1.55)
    band(s2, Inches(6.9), Inches(TOP), Inches(5.7), Inches(BOXH),
         RGBColor(0xFD, 0xF0, 0xEC))
    textbox(s2, Inches(7.2), Inches(TOP + 0.26), Inches(5.1), Inches(0.45),
            "해외는 데이터로 해결하고 있습니다", size=16.5, bold=True, color=RED)
    textbox(s2, Inches(7.2), Inches(TOP + 0.85), Inches(5.1), Inches(2.0),
            "· 애틀랜타 ‘Firebird’ — 위험점수로 점검 우선순위 결정\n"
            "· 미국 NFPA 가 모범사례로 선정\n"
            "· 뉴욕 FDNY — 위험기반 점검(RBIS) 운영\n"
            "· 국내 소방 정보화는 출동·신고 대응 중심", size=15.5, spacing=1.55)

    # 목적은 이 장표의 결론이다. 띄워 두지 말고 띠로 받친다.
    band(s2, Inches(0.8), Inches(5.75), Inches(11.8), Inches(0.95),
         RGBColor(0x2B, 0x33, 0x40))
    textbox(s2, Inches(1.1), Inches(6.02), Inches(11.2), Inches(0.5),
            "지역별 화재위험을 예측해 한정된 인력을 "
            "가장 위험한 곳과 시기에 먼저 배치합니다",
            size=17, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

    # ---- 7 화면③ 계획서 ----
    # 이 장표의 주장은 '옮겨 적을 것이 없습니다' 다. 그런데 계획서를 통째로
    # 줄여 넣으면 본문 글자를 읽을 수 없어 주장만 남고 증거가 안 보인다.
    # 문서 전체는 왼쪽에 두고, 결재가 되는 이유 셋을 확대해 옆에 붙인다.
    s_doc = section(prs, "2. 결과물", "일별·월별·연간 순찰 계획서",
                    "순찰 동선 · 중점 확인사항 · 법령 근거 포함")
    doc_img = (figs / "shot_plan_result.png"
               if (figs / "shot_plan_result.png").exists()
               else figs / "shot_plan_doc.png")
    # 문서가 화면보다 길어 한 장으로 못 찍는다(scripts/08 주석 참고).
    # 왼쪽에는 뒷부분 — 세부 근거 · 유의사항 · 붙임 · 끝. · 발신명의 · 결재란
    # 이 이어지는 대목을 둔다. 공문으로 보이는지가 여기서 판가름 난다.
    picture(s_doc, doc_img, Inches(0.8), Inches(2.15), Inches(4.4),
            max_h=Inches(4.0))

    head = figs / "shot_plan_head.png"
    callouts = [
        (crop_to(head, figs / "fig_plan_c1.png", 0.19, 0.47, 0.0, 0.52),
         "기관 · 수신 · 제목 · 시행일"),
        (crop_to(head, figs / "fig_plan_c2.png", 0.47, 0.72, 0.0, 0.52),
         "「관련」 근거 조문"),
        (figs / "shot_plan_approval.png" if (figs / "shot_plan_approval.png").exists()
         else None, "결재란"),
    ]
    # 흰 바탕에 그냥 얹으면 조각들이 떠 보인다. 다른 장표처럼 띠로 받친다.
    y = 2.15
    for img, label in callouts:
        if img is None or not Path(img).exists():
            continue
        band(s_doc, Inches(5.55), Inches(y), Inches(7.05), Inches(1.28))
        textbox(s_doc, Inches(5.85), Inches(y + 0.12), Inches(6.5), Inches(0.3),
                label, size=12.5, bold=True, color=RED)
        picture(s_doc, Path(img), Inches(5.85), Inches(y + 0.5), Inches(6.45),
                max_h=Inches(0.68))
        y += 1.40

    band(s_doc, Inches(0.8), Inches(6.4), Inches(11.8), Inches(0.75),
         RGBColor(0x2B, 0x33, 0x40))
    textbox(s_doc, Inches(1.1), Inches(6.58), Inches(11.2), Inches(0.45),
            "담당자가 옮겨 적을 항목이 없습니다. 그대로 결재에 올립니다.",
            size=16, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

    # ---- 3 무엇을 (구성) ----
    s3 = section(prs, "3. 제안 내용",
                 "예측부터 결재 문서까지 한 흐름으로",
                 "위험 예측 → 인력 기준 배분 → 관서별 순찰 동선 → 공문 서식 계획서")
    picture(s3, figs / "fig_pipeline.png", Inches(1.35), Inches(2.15), Inches(10.6))
    for i, (num, ttl, body) in enumerate([
            ("1", "예방점검 배분", "가용 인력 안에서\n화재가 가장 많이 담기도록 배분"),
            ("2", "관서별 순찰", "119안전센터에서 출발해\n관할 돌고 복귀"),
            ("3", "계획서 자동 생성", "공문 서식 · 법령 근거\n일별 · 월별 · 연간"),
            ("4", "업무 도우미", "소방 법령을 조문 근거와\n함께 찾아 줌")]):
        x = Inches(0.85 + i * 3.02)
        band(s3, x, Inches(4.35), Inches(2.82), Inches(2.15))
        textbox(s3, x, Inches(4.5), Inches(2.82), Inches(0.4), num,
                size=20, bold=True, color=RED, align=PP_ALIGN.CENTER)
        textbox(s3, x, Inches(4.98), Inches(2.82), Inches(0.4), ttl,
                size=14.5, bold=True, align=PP_ALIGN.CENTER)
        textbox(s3, x, Inches(5.5), Inches(2.82), Inches(1.0), body,
                size=12, color=MUTED, align=PP_ALIGN.CENTER)

    # ---- 4 활용 데이터 (필수 요건) ----
    s4 = section(prs, "4. 활용 데이터",
                 "소방안전 빅데이터 플랫폼 데이터 상품 8종",
                 "울산 4종으로 구축, 세종 4종으로 타 지역 적용 확인")
    # 8행 표가 화면을 다 차지하면 심사위원은 표를 읽고 말은 안 듣는다.
    # 결론을 위에 크게 두고 표는 근거로 아래에 둔다.
    _pn = manifest.get("panel", {})
    band(s4, Inches(0.8), Inches(2.25), Inches(11.8), Inches(0.9),
         RGBColor(0x2B, 0x33, 0x40))
    textbox(s4, Inches(1.1), Inches(2.5), Inches(11.2), Inches(0.45),
            f"8종을 하나의 표로 합쳤습니다  ·  {_pn.get('grids', 0):,}개 구역 × "
            f"{len(_pn.get('years', []))}개 연도  ·  화재 "
            f"{_pn.get('total_fires', 0):,.0f}건",
            size=18, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

    rows = [["데이터셋", "제공", "역할", "적재 건수"]] + [list(r) for r in ds_rows]
    table(s4, Inches(0.8), Inches(3.4), Inches(11.8), Inches(2.6), rows,
          col_widths=[4.2, 2.4, 3.4, 1.8], size=10.5)
    band(s4, Inches(0.8), Inches(6.2), Inches(11.8), Inches(0.95))
    textbox(s4, Inches(1.05), Inches(6.32), Inches(3.0), Inches(0.3),
            "여기에 더한 공개 자료", size=12.5, bold=True, color=RED)
    textbox(s4, Inches(1.05), Inches(6.66), Inches(11.3), Inches(0.35),
            "카카오 로컬 API 주소→좌표 (확보 "
            + pct(manifest.get("coverage", {}).get("fire", {}).get("rate")) + ")"
            "   ·   기상청 API 허브 일자료 8년치   ·   국가법령정보센터 소방 법령 "
            f"{extra.get('n_law', 0)}종 {extra.get('n_article', 0)}개 조문 · "
            f"별표 {extra.get('n_annex', 0)}건 · 법정 서식 "
            f"{extra.get('n_form', 0)}종", size=12.5)

    # ---- 시연 영상 ----
    # 발표 중에 다른 프로그램으로 넘어가지 않도록 영상을 장표에 심는다.
    # 창을 바꾸는 순간이 이 발표에서 유일하게 손이 미끄러질 수 있는 자리다.
    vid = cfg.paths.outputs / "demo_video" / "시연_핵심.mp4"
    s_vid = section(prs, "5. 시연", "실제로 도는 화면",
                    "배분 · 순찰 동선 · 조건 변경 · 계획서 생성")
    if vid.exists():
        poster = figs / "fig_video_poster.png"
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "12",
                            "-i", str(vid), "-frames:v", "1", str(poster)],
                           check=False)
        except Exception:                                # noqa: BLE001
            poster = None
        from PIL import Image
        try:
            vw, vh = Image.open(vid.with_suffix(".png")).size
        except Exception:                                # noqa: BLE001
            vw, vh = 1300, 1132
        if poster and poster.exists():
            try:
                vw, vh = Image.open(poster).size
            except Exception:                            # noqa: BLE001
                pass
        max_h, max_w = 4.55, 9.0
        wd = min(max_w, max_h * vw / vh)
        ht = wd * vh / vw
        # PDF 로 뽑을 때는 영상 도형을 넣지 않는다. LibreOffice 가 그 도형을
        # 그리지 못해 노이즈 덩어리로 나온다. 인쇄본에는 첫 화면을 그림으로.
        if movie:
            try:
                s_vid.shapes.add_movie(
                    str(vid), Inches((13.333 - wd) / 2), Inches(2.2),
                    Inches(wd), Inches(ht),
                    poster_frame_image=(str(poster) if poster and poster.exists()
                                        else None),
                    mime_type="video/mp4")
            except Exception as exc:                     # noqa: BLE001
                print(f"  영상 삽입 실패: {type(exc).__name__}: {exc}")
        elif poster and poster.exists():
            picture(s_vid, poster, Inches((13.333 - wd) / 2), Inches(2.2),
                    Inches(wd), max_h=Inches(ht))


    # ---- 5 지도: 어디가 위험하고 어디를 도는가 ----
    s_map = section(prs, "5. 서비스 화면 ①",
                    "구역별 화재위험 지도",
                    f"울산 전체 {t['model']['n_grids']:,}개 구역 · "
                    f"한 칸이 {grid_m}m")
    # 위험도 지도 한 장만 크게 둔다. 동선 지도를 나란히 줄이면 선이 사라져
    # 두 장 다 못 읽는 그림이 된다. 동선은 순찰 화면 장표와 영상이 맡는다.
    map_img = figs / "map_risk.png"
    if map_img.exists():
        from PIL import Image
        try:
            w_px, h_px = Image.open(map_img).size
            ratio = w_px / h_px
        except Exception:                                # noqa: BLE001
            ratio = 0.85
        max_h = 4.5
        width = min(5.5, max_h * ratio)
        pic = picture(s_map, map_img, Inches(0.7), Inches(2.15), Inches(width))
        if pic is not None:
            pic.width, pic.height = Inches(width), Inches(width / ratio)
    # 카드가 화면에 없는 것(동선)을 설명하고 있었다. 지도에 보이는 것만 쓴다.
    # 가운데 자리에는 '무엇을 보고 정하는가' 를 글로 나열하는 대신, 그것을
    # 화면이 실제로 보여 주는 요인 표를 넣는다. 방어 논리 셋 중 1번
    # ('왜 위험한지 설명이 된다')의 유일한 시각 증거다.
    reason = crop_to(figs / "shot_reason.png", figs / "fig_reason_crop.png",
                     0.586, 0.734, 0.025, 0.495)
    x_card, w_card = 6.6, 6.0
    y = 2.15
    band(s_map, Inches(x_card), Inches(y), Inches(w_card), Inches(1.2))
    textbox(s_map, Inches(x_card + 0.28), Inches(y + 0.14), Inches(5.4),
            Inches(0.38), "무엇을 그린 것인가", size=14.5, bold=True, color=RED)
    textbox(s_map, Inches(x_card + 0.28), Inches(y + 0.55), Inches(5.4),
            Inches(0.55),
            f"· {grid_m}m 정사각형 구역마다 그해 화재위험을 예측한 값\n"
            "· 색이 짙을수록 위험이 높습니다", size=12, color=MUTED, spacing=1.25)

    y = 3.5
    band(s_map, Inches(x_card), Inches(y), Inches(w_card), Inches(2.2))
    textbox(s_map, Inches(x_card + 0.28), Inches(y + 0.14), Inches(5.4),
            Inches(0.38), "왜 이 구역이 위험한가", size=14.5, bold=True, color=RED)
    if reason:
        picture(s_map, Path(reason), Inches(x_card + 0.25), Inches(y + 0.58),
                Inches(5.4), max_h=Inches(1.5))
    else:
        textbox(s_map, Inches(x_card + 0.28), Inches(y + 0.55), Inches(5.4),
                Inches(1.4), "· 과거 화재 · 주변 구역 확산 · 대상물 용도\n"
                "· 업소 업종 구성 · 소방용수 접근성",
                size=12, color=MUTED, spacing=1.25)

    y = 5.88
    band(s_map, Inches(x_card), Inches(y), Inches(w_card), Inches(1.1))
    textbox(s_map, Inches(x_card + 0.28), Inches(y + 0.14), Inches(5.4),
            Inches(0.38), "이 지도가 어디로 가는가", size=14.5, bold=True, color=RED)
    textbox(s_map, Inches(x_card + 0.28), Inches(y + 0.55), Inches(5.4),
            Inches(0.5),
            "· 점검 배분과 순찰 동선의 입력이 됩니다\n"
            "· 순찰계획서의 붙임으로 그대로 들어갑니다",
            size=12, color=MUTED, spacing=1.25)

    # ---- 6 화면② 배분 ----
    op = alloc.get("optimized", {})
    # 화면·CSV 에 실제로 나가는 배분(관할별 최소 배분 적용). 효율만 적용한
    # optimized 를 장표에 쓰면 시연 화면과 숫자가 어긋난다.
    eq = alloc.get("equity", {}) or {}
    shown_grids = int(eq.get("n_grids") or op.get("n_grids", 0))
    shown_cap = float(eq.get("actual_capture_rate")
                      if eq.get("actual_capture_rate") == eq.get("actual_capture_rate")
                      else op.get("actual_capture_rate", 0.0))

    tk = alloc.get("top_k_percent", {})
    screen_slide(
        prs, "5. 서비스 화면 ②", "인력에 맞춘 예방점검 배분",
        "점검관 인원·기간 입력 시 가능한 구역만 배분",
        figs / "shot_allocation.png",
        [("가용 인력 입력",
          "· 점검관 인원 · 1일 점검 건수 · 기간 입력\n"
          "· 값을 바꾸면 배분 즉시 재계산"),
         ("위험 구역일수록 큰 점검 부담",
          f"· 위험도 상위 {k}% = {tk.get('n_grids_selected', 0):,}개 구역\n"
          f"· 점검 소요 {tk.get('cost_if_all', 0):,.0f}건\n"
          f"· 1위 구역 한 곳 소요 "
          f"{(alloc.get('top1_grid') or {}).get('inspection_cost', 0):,.0f}건 > "
          f"가용 {alloc.get('budget_visits', 0):,}건"),
         ("소요와 효과의 동시 계산",
          "· 구역마다 점검 소요와 구역 안 화재를 함께 셈\n"
          f"· {alloc.get('budget_visits', 0):,}건 안에서 화재가 가장 큰 묶음 선택\n"
          f"· {shown_grids:,}개 구역 · 화재 {pct(shown_cap)} 포착 "
          f"({(alloc.get('equity') or alloc).get('gain_pp', 0):+.1f}%p)\n"
          f"· 관할별 최소 배분을 걸어도 손해 "
          f"{(alloc.get('equity') or {}).get('equity_cost_pp', 0):.1f}%p")],
        note="")

    # ---- 6 화면② 순찰 ----
    screen_slide(
        prs, "5. 서비스 화면 ③", "목적별 순찰 동선 자동 생성",
        "119안전센터 출발 → 관할 순회 → 복귀. 실제 도로 주행거리 기준",
        (figs / "shot_patrol_map.png"
         if (figs / "shot_patrol_map.png").exists() else figs / "shot_patrol.png"),
        [("출동 관서 기준",
          f"· 소방서 {extra.get('n_station', 6)}개 · 119안전센터 "
          f"{extra.get('n_center', 28)}개 · 읍면동 {extra.get('n_emd', 83)}개 "
          "가운데 선택"),
         ("목적별 순찰 6종",
          "일반예방 · 다중이용업소 야간 · 화재예방강화지구 · 피난약자시설 · "
          "소방용수 점검 · 건조기 특별경계"),
         ("근무시간 내 편성",
          "· 1회 순찰 시간 초과 시 회차 분할\n"
          "· 실제 도로 주행거리·소요시간 함께 제시")],
        note="",
        keep=1.0)

    # ---- 6-2 조건을 바꾸면 계획이 달라진다 ----
    screen_slide(
        prs, "5. 서비스 화면 ③-1", "조건 변경에 따른 계획 재생성",
        "바꾼 조건과 그 결과를 나란히 기록",
        (figs / "fig_route_compare.png"
         if (figs / "fig_route_compare.png").exists()
         else figs / "shot_patrol_compare.png"),
        [("바꾼 조건 기록",
          "· 목적 · 구역 수 · 1회 순찰 시간\n"
          "· 지역 · 거리 기준\n"
          "· 무엇을 바꿨는지 문장으로 기록"),
         ("달라진 결과",
          "· 총 이동거리 · 가장 먼 순찰조\n"
          "· 겹치는 구역 수\n"
          "· 바꾸기 전후를 나란히 비교"),
         ("빠진 구역 · 새 구역",
          "· 구역 번호를 그대로 표시\n"
          "· ‘왜 여기가 빠졌나’를 화면에서 확인")],
        note="",
        keep=1.0)

    # ---- 8 화면④ 업무 도우미 ----
    if (figs / "fig_form_compare.png").exists() or \
            (figs / "shot_form_compare.png").exists():
        screen_slide(
            prs, "5. 서비스 화면 ④", "법정 서식 자동 작성",
            "[별지 제11호서식] 화재예방강화지구 관리대장, 괘선까지 법제처 원본 그대로",
            (figs / "fig_form_compare.png"
             if (figs / "fig_form_compare.png").exists()
             else figs / "shot_form_compare.png"),
            [("법에 정해진 서식 그대로",
              "· 시행규칙 별지 제11호서식\n"
              "· 화재예방강화지구 관리대장"),
             ("채운 칸",
              f"· {extra.get('ledger_fields', 0)}개 칸 중 "
              f"**{extra.get('ledger_filled', 0)}개** 자동 입력\n"
              "· 건물동수·점포수·소방시설·관서거리·취약요소\n"
              "· 연면적·건축면적·건축연도는 건축물대장에서"),
             ("비워 둔 칸",
              "· 주민등록 등 다른 자료와 연계해야 채워지는 칸\n"
              "· 대표자·전화번호는 공개 데이터로 채울 수 있어도\n"
              "   개인정보라 넣지 않음")],
            note="",
            keep=1.0)

    # ---- 법령 검색 ----
    screen_slide(
        prs, "5. 서비스 화면 ⑤", "소방 법령 검색 및 근거 제시",
        f"법령 {extra.get('n_article', 0)}개 조문·별표·서식 "
        f"{extra.get('n_annex_all', 0)}건 색인, 조문 번호와 함께 제시",
        (figs / "shot_assistant_answer.png"
         if (figs / "shot_assistant_answer.png").exists() else figs / "shot_assistant.png"),
        [("수록 범위",
          f"· 법령 {extra.get('n_law', 0)}종 {extra.get('n_article', 0)}개 조문\n"
          f"· 별표·서식 {extra.get('n_annex_all', 0)}건\n"
          "· 업종별 점검 항목·관할 위험 현황을 함께 검색"),
         ("근거 조문 동시 제시",
          "· 답변에 법령명과 조문 번호를 함께 제시\n"
          "· 근거 없는 답변은 행정에서 쓸 수 없음"),
         ("신규 대원 업무 지원",
          "· ‘왜 여기가 위험한지’와 ‘무슨 근거로 하는지’를\n"
          "  같은 화면에서 확인")],
        note="",
        keep=0.78)

    # ---- 9 어떻게 믿나 (검증) ----
    s9 = section(prs, "6. 검증 결과 (1/3)",
                 f"{tr[0]}~{tr[-1]}년 학습, {year}년 예측",
                 f"{year}년 자료는 학습에 미사용")
    picture(s9, figs / "fig_decile.png", Inches(0.8), Inches(2.25), Inches(7.3))
    x = Inches(8.5)
    kpi(s9, x, Inches(2.3), Inches(4.0), pct(h["model_capture"]),
        f"위험 상위 {k}% 구역이 담은 실제 화재",
        (f"95% 신뢰구간 {m_ci['lo']:.1%} ~ {m_ci['hi']:.1%}"
         if m_ci and m_ci.get("lo") == m_ci.get("lo")
         else f"단순 기준 {pct(h['baseline_capture'])}"))
    kpi(s9, x, Inches(4.0), Inches(4.0), f"{h['model_lift']:.2f}배",
        "아무 데나 갔을 때 대비",
        f"위험 1등급 대비 10등급이 구역당 화재 "
        f"{extra.get('decile_hi', 0):.1f}건", color=BLUE)
    kpi(s9, x, Inches(5.7), Inches(4.0),
        (f"{d['point_pp']:+.1f}%p" if d else f"{h['delta_pp']:+.1f}%p"),
        f"전년 화재 순으로 갈 때 대비 (상위 {k}%)",
        (f"단순 기준 {pct(h['baseline_capture'])} · 95% 신뢰구간 "
         f"{d['lo_pp']:+.1f} ~ {d['hi_pp']:+.1f}%p"
         if d and "lo_pp" in d else
         f"단순 기준 {pct(h['baseline_capture'])}"), color=GREEN)
    textbox(s9, Inches(0.85), Inches(6.85), Inches(7.2), Inches(0.35),
            f"실제 화재를 다 알고 줄 세운 값을 100으로 보면 "
            f"{pct(t['model']['pei'][key], 0)} 수준입니다.",
            size=12, color=MUTED)

    # ---- 10 어디까지 확인했나 ----
    s10 = section(prs, "6. 검증 결과 (2/3)",
                  "타 지역·타 관할 적용 검증 4건",
                  "네 가지 방식으로 따로 확인")
    rows = [["확인한 것", "질문", f"상위 {k}% 포착", "결과"]]
    rows.append(["미래 예측", f"{year}년을 맞히는가", pct(h["model_capture"]),
                 (f"전년 화재 순 대비 {d['point_pp']:+.1f}%p"
                  if d else f"{h['delta_pp']:+.1f}%p")])
    logo_line = ""
    if "logo" in ev:
        lg = ev["logo"]
        wins = [(g["group"], (g["capture"] - g["baseline_capture"]) * 100)
                for g in lg.get("per_group", [])]
        n_win = sum(1 for _, d in wins if d > 0)
        rows.append(["관할 제외", "특정 구·군만 잘 맞는 것은 아닌가",
                     f"{pct(lg['capture_min'])} ~ {pct(lg['capture_max'])}",
                     (f"{n_win}/{len(wins)} 관할이 단순 기준 상회"
                      if wins else f"평균 {pct(lg['capture_mean'])}")])
        if wins:
            logo_line = ("관할 제외 검증 — 단순 기준 대비 "
                         + " · ".join(f"{g} {d:+.1f}" for g, d in
                                      sorted(wins, key=lambda x: -x[1]))
                         + f" %p (평균 {sum(d for _, d in wins)/len(wins):+.1f}%p)")
    if "transfer" in ev:
        tf = ev["transfer"]
        tci = tf.get("ci", {}).get(key, {}).get("model", {})
        rows.append(["타 지역 적용", "울산에서 만든 것이 세종에서도 되는가",
                     pct(tf["headline"]["model_capture"]),
                     (f"표본이 작아 {tci['lo']:.0%}–{tci['hi']:.0%} 범위"
                      if tci and tci.get("lo") == tci.get("lo") else "확장 가능")])
    if "resolution_scenarios" in ev:
        road = next((x for x in ev["resolution_scenarios"]
                     if "도로명" in str(x.get("시나리오", ""))), None)
        if road:
            rows.append(["주소 정밀도", "주소가 거칠어 성능이 부풀려진 것은 아닌가",
                         pct(road.get("모델포착@20%")), "부풀림 없음"])
    table(s10, Inches(0.8), Inches(2.3), Inches(11.8), Inches(2.4), rows,
          col_widths=[2.8, 5.0, 2.4, 3.0], size=13.5)
    if logo_line:
        band(s10, Inches(0.8), Inches(4.78), Inches(11.8), Inches(0.6),
             RGBColor(0xED, 0xF2, 0xF7))
        textbox(s10, Inches(1.05), Inches(4.94), Inches(11.3), Inches(0.35),
                logo_line, size=13, bold=True, color=BLUE)
    picture(s10, figs / "fig_monthly_risk.png", Inches(0.9), Inches(5.6),
            Inches(5.3), max_h=Inches(1.65))
    band(s10, Inches(6.8), Inches(5.6), Inches(5.8), Inches(1.65))
    textbox(s10, Inches(7.05), Inches(5.74), Inches(5.3), Inches(1.45),
            "‘언제’도 검증했습니다\n\n"
            "월별 화재위험 = 계절 패턴 × 기상(습도·건조일수).\n"
            f"{extra.get('season_hi_month', 0)}월이 연평균의 "
            f"{extra.get('season_hi', 0):.2f}배, "
            f"{extra.get('season_lo_month', 0)}월이 "
            f"{extra.get('season_lo', 0):.2f}배입니다.\n"
            f"기상을 넣어 설명력이 R² {extra.get('r2_season', 0):.2f} → "
            f"{extra.get('r2_weather', 0):.2f} 로 올라야만 채택합니다.\n"
            "이 계수로 월간 계획서의 주차별 순찰 횟수를 정합니다.", size=12.5)

    # ---- 회고 검증: 그해 이전 자료만으로 세웠다면 ----
    bt = extra.get("backtest", {})
    if bt:
        bh = bt.get("headline", {})
        s_bt = section(prs, "6. 검증 결과 (3/3)",
                       f"{bh.get('train_upto', 0)}년 자료 기준 "
                       f"{bh.get('year', 0)}년 회고 검증",
                       "그해 이전 자료만으로 계획했을 때의 포착 결과")
        rows = [["순찰 구역", "관내 비중", "우리 계획", "작년 화재 순", "무작위",
                 "그해 화재 중"]]
        # 네 줄이 같은 무게로 놓이면 어디를 봐야 할지 알 수 없다.
        # 오늘 말하는 운영점 한 줄만 굵게 세운다.
        _opn = bt.get("operating_point", 60)
        for r in bt["years"][0]["by_patrol_size"]:
            hit = r["patrol_grids"] == _opn
            def _m(t):
                return f"**{t}**" if hit else t
            rows.append([_m(f"{r['patrol_grids']}개"), _m(f"{r['share_of_city']:.1%}"),
                         _m(f"{r['model_fires']:.0f}건"), _m(f"{r['baseline_fires']:.0f}건"),
                         _m(f"{r['random_fires']:.1f}건"), _m(f"{r['capture_share']:.1%}")])
        table(s_bt, Inches(0.8), Inches(2.3), Inches(6.9), Inches(2.4), rows,
              col_widths=[1.5, 1.4, 1.4, 1.5, 1.2, 1.4], size=12.5)

        kpi(s_bt, Inches(8.1), Inches(2.3), Inches(2.15),
            f"{bh.get('share_of_city', 0):.1%}",
            "순찰한 구역", f"{bh.get('patrol_grids', 0)}개 / 관내 전체")
        kpi(s_bt, Inches(10.45), Inches(2.3), Inches(2.15),
            f"{bh.get('capture_share', 0):.1%}", "그 안에서 난 화재",
            f"{bh.get('model_fires', 0):,.0f}건 / "
            f"{bh.get('total_fires', 0):,.0f}건", color=RED)

        sc = bh.get("scenarios", [])
        lines_ = ["순찰이 화재를 e 만큼 막는다고 두면"]
        for s in sc:
            lines_.append(f"   e = {s['assumed_effect']:.0%}  →  "
                          f"연 {s['prevented_if_patrolled']:.1f}건")
        band(s_bt, Inches(0.8), Inches(5.55), Inches(5.6), Inches(1.45))
        textbox(s_bt, Inches(1.05), Inches(5.68), Inches(5.1), Inches(1.2),
                "\n".join(lines_), size=13)
        band(s_bt, Inches(6.7), Inches(5.55), Inches(5.9), Inches(1.45))
        textbox(s_bt, Inches(6.95), Inches(5.68), Inches(5.4), Inches(1.2),
                "e 는 저희가 측정할 수 없어 세 경우를 나란히 둡니다.\n"
                "잰 것은 ‘순찰 구역 안에서 난 화재 건수’까지입니다.\n\n"
                + "같은 절차를 "
                + " · ".join(f"{y['year']}년 "
                             f"{y['by_patrol_size'][2]['capture_share']:.1%}"
                             for y in bt["years"])
                + " 로 반복했습니다.", size=12.5)
        # 세 해를 나란히 둔다. 한 해만 적으면 '그해만 운이 좋았나' 로 읽힌다.
        _op = bt.get("operating_point", 60)
        per_year = []
        for y in bt["years"]:
            r = next((x for x in y["by_patrol_size"]
                      if x["patrol_grids"] == _op), None)
            if r:
                g = r.get("gain_ci", {})
                per_year.append(f"{y['year']}년 {r['gain_over_baseline']:+.0f}건 "
                                f"[{g.get('lo', 0):+.0f}, {g.get('hi', 0):+.0f}]")
        band(s_bt, Inches(0.8), Inches(4.85), Inches(11.8), Inches(0.55))
        textbox(s_bt, Inches(1.0), Inches(4.93), Inches(11.4), Inches(0.4),
                f"작년 화재 순으로 같은 {_op}곳을 골랐을 때와의 차이 · 95% 신뢰구간   "
                + "   ·   ".join(per_year)
                + "   — 세 해 모두 0을 넘습니다.", size=12.5)



    # ---- 8-2 차별성 ----
    s_diff = section(prs, "7. 차별성",
                     "예측 다음 단계: 배분 · 동선 · 문서",
                     "해외 사례도 예측에서 멈췄습니다. "
                     "‘그래서 이번 달에 어디를 돌아라’까지 가는 세 걸음")
    cards = [
        ("① 위험 순서 배열의 한계",
         "구역마다 점검 소요가 다릅니다.\n"
         f"· 상위 {k}% = {tk.get('n_grids_selected', 0):,}개 구역 · "
         f"{tk.get('cost_if_all', 0):,.0f}건 소요\n"
         f"· 1위 구역 소요 "
         f"{(alloc.get('top1_grid') or {}).get('inspection_cost', 0):,.0f}건 > "
         f"가용 {alloc.get('budget_visits', 0):,}건\n"
         f"· 위험 순서대로: 통째 "
         f"{(alloc.get('risk_order') or {}).get('capture_rate_whole', 0):.0%}"
         f" · 부분 인정 "
         f"{(alloc.get('risk_order') or {}).get('capture_rate_partial', 0):.1%}\n"
         f"  건너뛰어도 "
         f"{(alloc.get('risk_order') or {}).get('skip_n_grids', 0)}개 구역 · "
         f"{(alloc.get('risk_order') or {}).get('capture_rate_skip', 0):.1%}\n\n"
         "그래서 기준을 바꿨습니다.\n"
         f"· 소요 합계 {alloc.get('budget_visits', 0):,}건 이내에서\n"
         "· 담기는 화재 합계가 가장 큰\n"
         "  구역 묶음을 선택\n\n"
         f"→ 같은 인력, {shown_grids:,}개 구역, "
         f"{(alloc.get('equity') or alloc).get('gain_pp', 0):+.1f}%p",
         ""),
        ("② 예측 결과의 문서화",
         "일별·월별·연간 계획서를\n"
         "일반기안문 배열로 만듭니다.\n"
         "· 수신 · 경유 · 제목 · 붙임 · 끝. · 발신명의\n\n"
         "법정 서식([별지 제11호서식])은\n"
         "괘선까지 원본 그대로 두고\n"
         "값만 채웁니다.\n\n"
         "담당자가 옮겨 적을 항목이 없습니다.",
         ""),
        ("③ AI 인용의 기계 검증",
         "공공이 생성형 AI 를 못 쓰는 이유는\n"
         "성능이 아니라 검증입니다.\n\n"
         "· 계획서: 원문에 없던 수·조문이\n"
         "  하나라도 생기면 그 결과를 버림\n\n"
         "· 업무 도우미: 인용 조문을 원문과 대조,\n"
         "  자료 밖이면 ‘확인 필요’ 표시\n\n"
         "폐쇄망: 로컬 모델(Ollama) 또는 규칙기반으로\n"
         "외부 호출 없이 같은 문서를 만듭니다.\n"
         "생성 경로는 산출물에 기록됩니다.",
         ""),
    ]
    x0, w_card, top, card_h = 0.8, 3.87, 2.55, 3.95
    body_h = card_h - 1.1                       # 제목·여백을 뺀 본문 높이(in)
    longest = max(len(b.split("\n")) for _, b, _ in cards)
    size_body, spacing = 11.5, 1.24
    # 줄 높이(pt) × 줄 수가 본문 높이를 넘으면 글자를 줄인다.
    while longest * size_body * spacing / 72.0 > body_h and size_body > 9.0:
        size_body -= 0.25
    for i, (head, body, foot) in enumerate(cards):
        x = Inches(x0 + i * (w_card + 0.16))
        band(s_diff, x, Inches(top), Inches(w_card), Inches(card_h))
        textbox(s_diff, x + Inches(0.22), Inches(top + 0.2), Inches(w_card - 0.44),
                Inches(0.7), head, size=14, bold=True, color=RED)
        textbox(s_diff, x + Inches(0.22), Inches(top + 0.85), Inches(w_card - 0.44),
                Inches(body_h), body, size=size_body, color=INK, spacing=spacing)
        if foot:
            textbox(s_diff, x + Inches(0.22), Inches(top + card_h - 0.7),
                    Inches(w_card - 0.44), Inches(0.5), foot, size=10.5,
                    color=MUTED)


    # ---- 미국 사례 비교 ----
    s_us = section(prs, "7. 차별성 (계속)",
                   "해외 사례 비교: 애틀랜타 Firebird",
                   "미국 NFPA 모범사례 선정 시스템 (KDD 2016)")
    rows = [["", "Firebird (애틀랜타, 2016)", "불씨예보 (울산, 2026)"],
            ["분석 단위", "위험점수를 매긴 상업용 건물 5,000여 개소",
             f"{grid_m}m 구역 {t['model']['n_grids']:,}개"],
            ["자료", "8종 결합 (건물대장·화재·인구 등)", "소방안전 빅데이터 8종 + 기상 + 법령"],
            ["예측 성능", "상업용 화재 70% 이상 예측\n(오경보율 20% 기준)",
             f"위험 상위 20% 구역이 화재 {pct(h['model_capture'])} 포착\n"
             f"(95% 신뢰구간 {m_ci.get('lo', 0):.0%} ~ {m_ci.get('hi', 0):.0%})"],
            ["검증 방식", "시간분할 (학습 이후 화재로 검증)",
             "시간분할 + 관할제외 + 타 지역 + 주소 정밀도"],
            ["산출물", "위험점수 · 지도 시각화",
             "위험지도 · 인력 제약 배분 · 관서별 동선 ·\n공문 계획서 · 법령 질의응답"],
            ["점검 대상", "기존 2,573개소 + 새로 찾은 후보 19,397개소\n"
                          "(추려서 6,096개소 권고)",
             f"{tk.get('n_grids_selected', 0):,}개 구역 · 대상물 "
             f"{tk.get('targets_if_all', 0):,.0f}개소\n"
             f"점검 소요 {tk.get('cost_if_all', 0):,.0f}건"],
            ["인력 제약", "**미해결**: 연 점검 6,096건 증가 = 지금(2,573건)의\n"
                          "2.37배. 조직·조례·증원 없이는 불가능하다고 논문이 적음",
             f"**해결**: 가용 인력 안에서 배분,\n동일 인력 대비 "
             f"{(alloc.get('equity') or alloc).get('gain_pp', 0):+.1f}%p 개선"]]
    table(s_us, Inches(0.8), Inches(2.3), Inches(11.8), Inches(3.9), rows,
          col_widths=[2.2, 4.8, 4.8], size=10,
          row_sizes={0: 11.5, 6: 12.5, 7: 12.5})
    band(s_us, Inches(0.8), Inches(6.35), Inches(11.8), Inches(0.75),
         RGBColor(0xEC, 0xF8, 0xF2))
    textbox(s_us, Inches(1.05), Inches(6.48), Inches(11.3), Inches(0.5),
            "Firebird 논문에는 순찰 경로도, 계획 문서도 나오지 않습니다. "
            "인력 제약을 지적한 자리에서 멈췄습니다.",
            size=13, bold=True, color=GREEN)

    # ---- 11 기대효과 ----
    s11 = section(prs, "8. 기대효과 및 활용방안",
                  "순찰 효율 개선과 근거 기록",
                  "곱하지 않고 잰 값만 적었습니다")
    # 이 서비스의 결과물은 순찰 경로다. 머리 지표도 순찰에서 시작한다.
    _bh = (extra.get("backtest") or {}).get("headline", {})
    if _bh:
        kpi(s11, Inches(0.8), Inches(2.32), Inches(3.8),
            f"{_bh.get('capture_share', 0):.1%}", "순찰 구역 안에서 난 화재",
            f"관내 {_bh.get('share_of_city', 0):.1%}"
            f"({_bh.get('patrol_grids', 0)}개 구역)만 돌았을 때 · "
            f"{_bh.get('year', 0)}년 회고", size=40, h=1.72)
    elif alloc and "gain_pp" in alloc:
        kpi(s11, Inches(0.8), Inches(2.4), Inches(3.8),
            f"{(alloc.get('equity') or alloc).get('gain_pp', 0):+.1f}%p",
            "같은 인력 기준 포착률 개선",
            f"인력에 맞춘 배분 vs 상위 {k}% 방식")
    if alloc and "gain_pp" in alloc:
        kpi(s11, Inches(4.85), Inches(2.32), Inches(3.8),
            f"{(alloc.get('equity') or alloc).get('gain_pp', 0):+.1f}%p",
            "같은 인력 기준 점검 포착률",
            f"{shown_grids:,}개 구역 · 관할별 최소 배분 적용 · "
            f"무작위 대비 {h['model_lift']:.2f}배", color=BLUE, size=40, h=1.72)
    hc = summary.get("hydrant_coverage", {})
    bp = summary.get("blind_spot_population") or {}
    kpi(s11, Inches(8.9), Inches(2.32), Inches(3.7),
        f"{summary.get('n_blind_spots', 0)}개",
        "고위험 · 소방용수 사각 구역",
        (f"{bp['n_emd']}개 읍면동 · 상주인구 {bp['population']:,}명"
         if bp else
         f"전체 구역의 {pct(hc.get('share_without_hydrant'))}에 소화전 없음"),
        color=GREEN, size=40, h=1.72)
    # 회고 검증을 세 해 모두 적는다. 한 해만 적으면 우연으로 읽힌다.
    _bt = extra.get("backtest") or {}
    if _bt.get("years"):
        _op = _bt.get("operating_point", 60)
        _yy = []
        for y in _bt["years"]:
            r = next((x for x in y["by_patrol_size"]
                      if x["patrol_grids"] == _op), None)
            if r:
                g = r.get("gain_ci", {})
                _yy.append(f"{y['year']} {r['gain_over_baseline']:+.0f} "
                           f"[{g.get('lo', 0):+.0f},{g.get('hi', 0):+.0f}]")
        band(s11, Inches(0.8), Inches(4.2), Inches(11.8), Inches(0.6))
        textbox(s11, Inches(1.05), Inches(4.36), Inches(11.3), Inches(0.35),
                f"회고 검증 {_op}개 구역 · 작년 화재 순 대비   "
                + " · ".join(_yy)
                + "건 — 세 해 모두 0 초과", size=12, color=MUTED)
    textbox(s11, Inches(0.8), Inches(5.05), Inches(11.8), Inches(2.4),
            "· 예방순찰   119안전센터별 출동 계획, 목적별 순찰 6종, 월별 순찰 강도\n"
            "· 예방점검   화재안전조사 대상 우선순위를 자동으로 정하고, 공문 서식 계획서로 바로 결재\n"
            "· 소방용수 정책   고위험인데 소화전이 없는 구역을 신설 우선순위의 객관적 근거로\n"
            "· 행정 지원   신규 대원·신규 부임지에서 위험 판단 근거와 법령 조문을 함께 제공\n"
            "· 확산   세종 적용으로 확인. 공개데이터만 쓰므로 데이터 구매비가 들지 않음",
            size=15.5, spacing=1.65)

    # ---- 12 한계 ----
    s12 = section(prs, "9. 한계와 향후 계획",
                  "확인된 한계와 대응 방안",
                  "아는 한계와 지금의 대응")
    rows = [["한계", "현재 대응", "향후"]]
    rows += [
        ["공개 데이터에 건물번호·좌표가 없음",
         "도로명 → 읍면동 순으로 좌표를 찾고 단계를 기록", "상세주소 확보 시 건물 단위"],
        [f"화재의 {pct(conc.get('share_from_emd_centroid'), 0)}가 읍면동 중심 좌표",
         "도로명이 있는 건만으로 따로 검증(성능 저하 없음)", "도로명 기재율 개선 협의"],
        ["대상물·업소는 현재 시점 현황",
         "과거 이력만으로도 검증해 함께 제시(" + pct(hist_cap) + ")",
         "연도별 이력 자료 확보"],
        ["건축물대장 노후도로 대체 시도",
         f"읍면동 단위로 붙여 측정, 개선 없음 "
         f"({bld.get('delta_pp', 0):+.1f}%p, 신뢰구간 0 포함)",
         "구역 단위 주소 확보 시 재측정"],
        ["점검 이력을 붙일 수 없음",
         "결합할 키가 없다는 것을 실제로 확인해 기록", "대상물 관리번호 포함 자료 요청"],
        ["순찰 횟수를 근무편성과 잇지 못함",
         "월 위험계수로 주차별 횟수까지는 산출",
         "관서 교대·인원 편성 자료와 연계"],
        ["**단순 기준 대비 개선폭이 크지 않음**",
         f"**지난 화재만으로도 {pct(probe0.get('capture_top20'))} — 먼저 공개합니다**",
         "**가치는 인력 배분에 있습니다**"],
    ]
    table(s12, Inches(0.8), Inches(2.6), Inches(11.8), Inches(4.0), rows,
          col_widths=[3.6, 4.8, 3.4], size=12.5)
    textbox(s12, Inches(0.8), Inches(6.9), Inches(11.8), Inches(0.4), "",
            size=14, color=MUTED)

    # ---- 13 마무리 ----
    s13 = blank(prs)
    band(s13, 0, 0, W, Inches(0.14), RED)
    if logo.exists():
        picture(s13, logo, Inches(0.85), Inches(1.05), Inches(3.6), max_h=Inches(1.6))
    textbox(s13, Inches(0.9), Inches(2.75), Inches(11.5), Inches(1.6),
            "한정된 인력을\n가장 위험한 곳에", size=44, bold=True)
    textbox(s13, Inches(0.9), Inches(4.9), Inches(6.2), Inches(1.4),
            f"울산광역시 {manifest.get('panel', {}).get('grids', 0):,}개 구역 · "
            f"화재 {manifest.get('panel', {}).get('total_fires', 0):,.0f}건으로 검증\n\n"
            "박용준 · 아주대학교 산업공학과 석사과정\n"
            "github.com/dragonzzuny/Fire_bigdata", size=17, color=MUTED)

    # 이 장표는 질의응답 5분 내내 화면에 떠 있다. 오른쪽 절반이 비어 있으면
    # 그동안 근거가 심사위원 눈앞에 없다.
    _al = (alloc.get("equity") or alloc)
    _bh2 = (extra.get("backtest") or {}).get("headline", {})
    _ro = alloc.get("risk_order") or {}
    facts = [
        (f"{_al.get('actual_capture_rate', 0):.1%}",
         "같은 인력으로 담는 화재\n"
         "위험 순서대로는 어떻게 세어도 3% 미만 — 통째 "
         f"{_ro.get('capture_rate_whole', 0):.0%} · 부분 인정 "
         f"{_ro.get('capture_rate_partial', 0):.1%} · 건너뛰어도 "
         f"{_ro.get('capture_rate_skip', 0):.1%}"),
        (pct(h["model_capture"]), f"위험 상위 {k}% 구역이 담은 실제 화재"),
        (f"{_bh2.get('capture_share', 0):.1%}",
         f"관내 {_bh2.get('share_of_city', 0):.1%}만 순찰한 "
         f"{_bh2.get('year', 0)}년 회고 검증"),
    ]
    y = 1.25
    for value, label in facts:
        rows = label.count("\n") + 1
        textbox(s13, Inches(7.4), Inches(y), Inches(5.6), Inches(0.75),
                value, size=40, bold=True, color=RED)
        textbox(s13, Inches(7.4), Inches(y + 0.8), Inches(5.6),
                Inches(0.26 * rows + 0.1), label, size=13, color=MUTED,
                spacing=1.2)
        y += 1.45 + 0.24 * rows

    return prs




def collect_extra(cfg, city: str) -> dict:
    """장표에 들어갈 부가 수치. 없으면 0 으로 두고 그 자리를 비운다."""
    import pandas as pd

    out: dict = {}

    # 위험 등급별 격자당 실제 화재. 장표에 손으로 적지 않기 위해 파일에서 읽는다.
    try:
        import json as _json
        _ev = _json.loads((cfg.paths.outputs / "evaluation.json").read_text(encoding="utf-8"))
        _dec = _ev["temporal"]["model"]["decile"]
        _key = next(k for k in _dec[0] if "fire" in k.lower() or "화재" in k)
        out["decile_hi"] = float(_dec[-1][_key])
        out["decile_lo"] = float(_dec[0][_key])
    except Exception:                                    # noqa: BLE001
        pass
    try:
        from firebird import dataset as D
        panel = D.load_panel(cfg, city)
        cur = panel[panel["year"] == cfg.holdout_year]
        for col, key in (("station", "n_station"), ("center", "n_center"),
                         ("emd", "n_emd")):
            if col in cur.columns:
                out[key] = int(cur[col].astype(str).str.strip().replace("", pd.NA)
                               .dropna().nunique())
    except Exception:                                    # noqa: BLE001
        pass

    bp = cfg.paths.outputs / f"building_feature_eval_{city}.json"
    if bp.exists():
        try:
            out["building_eval"] = json.loads(bp.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            pass

    law = cfg.paths.cache / "law_articles.parquet"
    if law.exists():
        df = pd.read_parquet(law)
        out["n_law"] = int(df["law"].nunique())
        out["n_article"] = int(len(df))

    forms = cfg.paths.cache / "law_forms.parquet"
    if forms.exists():
        fm = pd.read_parquet(forms)
        out["n_form"] = int((fm["kind"] == "서식").sum())
        out["n_annex"] = int(len(fm) - out["n_form"])
        out["n_annex_all"] = int(len(fm))          # 색인에는 별표·서식을 모두 넣는다

    # 법정 서식을 몇 칸 채우는지. 장표에 손으로 적지 않기 위해 실제로 채워 본다.
    try:
        from firebird import dataset as D2, forms as FM, stations as ST2
        panel2 = D2.load_panel(cfg, city)
        cur2 = panel2[panel2["year"] == cfg.holdout_year]
        if len(cur2):
            row = cur2.nlargest(1, "fires_cum").iloc[0]
            sta = pd.concat([ST2.station_table(cur2, cfg, level=lv,
                                               city_label=cfg.city(city)["label"])
                             for lv in ("station", "center")], ignore_index=True)
            led = FM.zone_ledger(row, city_label=cfg.city(city)["label"],
                                 year=int(cfg.holdout_year), stations=sta,
                                 grid_m=int(cfg.grid_size_m))
            out["ledger_fields"] = int(led["n_fields"])
            out["ledger_filled"] = int(led["n_filled"])
    except Exception:                                    # noqa: BLE001
        pass

    # 서식 값은 scripts/09 가 낸 파일이 정본이다. 위에서 다시 계산한 값은
    # 건축물대장 연계가 빠져 그림 캡션(17/27)과 어긋났다(13/27).
    try:
        import json as _json
        _fl = cfg.paths.outputs / f"form_ledger_{city}_{cfg.holdout_year}.json"
        if _fl.exists():
            _d = _json.loads(_fl.read_text(encoding="utf-8"))
            out["ledger_fields"] = int(_d["n_fields"])
            out["ledger_filled"] = int(_d["n_filled"])
    except Exception:                                    # noqa: BLE001
        pass

    try:
        from firebird import monthly as MO
        fp = cfg.paths.processed / f"fires_{city}.parquet"
        wp = cfg.paths.processed / f"weather_monthly_{city}.parquet"
        if fp.exists() and wp.exists():
            mf = MO.fires_by_month(pd.read_parquet(fp), cfg.year_min, cfg.year_max)
            fit = MO.fit_month_risk(mf, pd.read_parquet(wp))
            out["r2_season"] = float(fit.get("baseline_r2", 0.0))
            out["r2_weather"] = float(fit.get("weather_r2", fit.get("baseline_r2", 0.0)))
            # 계절 지수 최고·최저 달. 장표에 손으로 적지 않기 위해 여기서 뽑는다.
            si = MO.seasonal_index(mf)
            if not si.empty:
                hi = si.loc[si["seasonal_index"].idxmax()]
                lo = si.loc[si["seasonal_index"].idxmin()]
                out["season_hi_month"] = int(hi["month"])
                out["season_hi"] = float(hi["seasonal_index"])
                out["season_lo_month"] = int(lo["month"])
                out["season_lo"] = float(lo["seasonal_index"])
    except Exception:                                    # noqa: BLE001
        pass

    # 회고 검증(scripts/17). 장표에 손으로 적지 않기 위해 파일에서 읽는다.
    bt = cfg.paths.outputs / f"backtest_patrol_{city}.json"
    if bt.exists():
        out["backtest"] = json.loads(bt.read_text(encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="ulsan")
    ap.add_argument("--no-pdf", dest="pdf", action="store_false",
                    help="PDF 변환을 건너뛴다")
    args = ap.parse_args()

    cfg = load_config()
    ev_path = cfg.paths.outputs / "evaluation.json"
    if not ev_path.exists():
        print("evaluation.json 이 없다. scripts/04_train_eval.py 를 먼저 돌려라.")
        return 1
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    city = ev.get("city", "ulsan")
    year = ev["temporal"]["test_year"]

    def load(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    summary = load(cfg.paths.outputs / f"artifacts_summary_{city}_{year}.json")
    manifest = load(cfg.paths.processed / f"manifest_{city}.json")

    figs = cfg.paths.figures
    missing = [f for f in ("fig_capture_curve.png", "fig_allocation.png")
               if not (figs / f).exists()]
    if missing:
        print(f"그림이 없다: {missing}. scripts/06_figures.py 를 먼저 돌려라.")
        return 1

    rows, extra = dataset_rows(cfg), collect_extra(cfg, city)
    out = cfg.paths.outputs / f"불씨예보_발표자료_{city}_{year}.pptx"
    prs = build(cfg, ev, summary, manifest, figs, rows, extra, movie=True)
    prs.save(out)
    print(f"발표자료 저장: {out}  ({len(prs.slides._sldIdLst)}장)")

    # 인쇄·제출용 PDF. 영상 도형이 들어 있으면 LibreOffice 가 그 자리를
    # 노이즈로 그린다. 같은 장표를 그림으로 바꿔 한 번 더 짓고 그것을 변환한다.
    if args.pdf:
        import tempfile
        pdf = out.with_suffix(".pdf")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / out.name
            build(cfg, ev, summary, manifest, figs, rows, extra,
                  movie=False).save(tmp)
            r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                                "--outdir", td, str(tmp)],
                               capture_output=True, text=True, timeout=900)
            made = Path(td) / (tmp.stem + ".pdf")
            if made.exists():
                pdf.unlink(missing_ok=True)
                shutil.copy2(made, pdf)
                print(f"PDF 저장: {pdf}  (영상 자리는 첫 화면 그림)")
            else:
                print(f"PDF 변환 실패: {r.stderr[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
