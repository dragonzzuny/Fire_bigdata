#!/usr/bin/env python
"""발표자료(PPTX) 생성 — 모든 수치는 outputs/evaluation.json 에서 읽는다.

장표에 손으로 적은 숫자를 올리면 질문 한 번에 무너진다. 여기서 만드는
숫자는 전부 파이프라인이 출력한 값이고, 파이프라인을 다시 돌리면
장표도 따라서 갱신된다.
"""
from __future__ import annotations

import json
import sys
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
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = FONT
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


def kpi(slide, x, y, w, value, label, sub="", color=RED):
    band(slide, x, y, w, Inches(1.5))
    textbox(slide, x, y + Inches(0.14), w, Inches(0.7), value,
            size=34, bold=True, color=color, align=PP_ALIGN.CENTER)
    textbox(slide, x, y + Inches(0.82), w, Inches(0.35), label,
            size=13, bold=True, align=PP_ALIGN.CENTER)
    if sub:
        textbox(slide, x, y + Inches(1.13), w, Inches(0.3), sub,
                size=10, color=MUTED, align=PP_ALIGN.CENTER)


def table(slide, x, y, w, h, rows: list[list[str]], *, col_widths=None,
          header_color=RGBColor(0x2B, 0x33, 0x40), size=12):
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
            run.font.size = Pt(size); run.font.name = FONT
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
    ("fire",     "화재발생현황",       "학습 라벨 (격자·연도별 화재 건수)"),
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
        im = Image.open(shot)
        h = int(im.size[1] * keep)
        im.crop((0, 0, im.size[0], h)).save(out)
        return out
    except Exception:                                    # noqa: BLE001
        return shot


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
          ds_rows: list[list[str]], extra: dict) -> Presentation:
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
    textbox(s1, Inches(8.1), Inches(4.92), Inches(4.0), Inches(1.2),
            "박용준\n아주대학교 산업공학과 석사과정", size=16, bold=True)
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
    band(s2, Inches(0.8), Inches(2.4), Inches(5.6), Inches(3.5))
    textbox(s2, Inches(1.1), Inches(2.68), Inches(5.0), Inches(3.1),
            "· 소방공무원 증원 정체 (2024년 전년 대비 +5명 수준)\n"
            "· 특정소방대상물·다중이용업소, 30층 이상 고층건축물(+8%) 지속 증가\n"
            "· 같은 법정 대상 안에서도 용도·업종·화재이력에 따라 실제 위험은 크게 다름\n"
            "· 그 차이를 데이터로 구분해 우선순위를 정하는 체계가 없음", size=16.5)
    band(s2, Inches(6.9), Inches(2.4), Inches(5.7), Inches(3.5),
         RGBColor(0xFD, 0xF0, 0xEC))
    textbox(s2, Inches(7.2), Inches(2.68), Inches(5.1), Inches(0.45),
            "해외는 데이터로 해결하고 있습니다", size=16.5, bold=True, color=RED)
    textbox(s2, Inches(7.2), Inches(3.2), Inches(5.1), Inches(2.6),
            "· 애틀랜타 소방 ‘Firebird’: 위험점수로 점검\n"
            "  우선순위 결정, 미국 NFPA 모범사례 선정\n"
            "· 뉴욕 FDNY: 위험기반 점검(RBIS) 운영\n\n"
            "국내 소방 정보화는 출동·신고 대응 중심이며,\n"
            "예방점검 대상 우선순위화 영역은 비어 있습니다.", size=15.5)
    textbox(s2, Inches(0.8), Inches(6.15), Inches(11.8), Inches(0.6),
            "목적: 지역별 화재위험을 예측해 한정된 인력을 "
            "가장 위험한 곳과 시기에 먼저 배치",
            size=15, bold=True)

    # ---- 7 화면③ 계획서 ----
    screen_slide(
        prs, "2. 결과물", "일별·월별·연간 순찰 계획서",
        "순찰 동선 · 중점 확인사항 · 법령 근거 포함",
        (figs / "shot_plan_result.png"
         if (figs / "shot_plan_result.png").exists() else figs / "shot_plan_doc.png"),
        [("일별 · 월별 · 연간",
          "· 월별: 그 달 화재위험으로 순찰 횟수 산정\n"
          "· 연간: 계절별 순찰 유형·법정 이행사항 배치"),
         ("공문 서식 그대로",
          "· 기관·수신·시행일·관련 근거·붙임·결재란 포함\n"
          "· 담당자가 옮겨 적을 항목 없음"),
         ("법정 서식 안내",
          "· 조치에 필요한 별지 서식을 함께 안내\n"
          "· 예) 화재예방강화지구 관리대장")],
        note="",
        # 계획서 원본은 세로로 매우 길다. 장표 비율에 맞게 머리 부분만 쓴다.
        keep=0.26)

    # ---- 3 무엇을 (구성) ----
    s3 = section(prs, "3. 제안 내용",
                 "예측부터 결재 문서까지 한 흐름으로",
                 "위험 예측 → 인력 기준 배분 → 관서별 순찰 동선 → 공문 서식 계획서")
    picture(s3, figs / "fig_pipeline.png", Inches(1.35), Inches(2.15), Inches(10.6))
    for i, (num, ttl, body) in enumerate([
            ("1", "예방점검 배분", "가용 인력 안에서\n가장 많이 잡히도록 배분"),
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
    rows = [["데이터셋", "제공", "역할", "적재 건수"]] + [list(r) for r in ds_rows]
    table(s4, Inches(0.8), Inches(2.3), Inches(11.8), Inches(3.1), rows,
          col_widths=[4.2, 2.4, 3.4, 1.8], size=11.5)
    textbox(s4, Inches(0.8), Inches(5.6), Inches(7.4), Inches(1.3),
            "· 카카오 로컬 API: 주소를 좌표로 변환 (좌표 확보 "
            + pct(manifest.get("coverage", {}).get("fire", {}).get("rate")) + ")\n"
            "· 기상청 API 허브: 일자료 8년치로 건조 정도 산출\n"
            f"· 국가법령정보센터: 소방 법령 {extra.get('n_law', 0)}종 "
            f"{extra.get('n_article', 0)}개 조문 · 별표 {extra.get('n_annex', 0)}건 "
            f"· 법정 서식 {extra.get('n_form', 0)}종\n"
            "· 개인정보를 다루지 않으며, 집계 단위 공공데이터만 사용",
            size=12.5, color=MUTED)
    band(s4, Inches(8.4), Inches(5.6), Inches(4.2), Inches(1.2),
         RGBColor(0xF4, 0xF6, 0xF8))
    textbox(s4, Inches(8.65), Inches(5.75), Inches(3.7), Inches(1.0),
            f"흩어진 자료를 하나의 표로 통합\n"
            f"→ {manifest.get('panel', {}).get('grids', 0):,}개 구역 × "
            f"{len(manifest.get('panel', {}).get('years', []))}개 연도 · "
            f"화재 {manifest.get('panel', {}).get('total_fires', 0):,.0f}건",
            size=12.5, bold=True)

    # ---- 5 지도: 어디가 위험하고 어디를 도는가 ----
    s_map = section(prs, "5. 서비스 화면 ①",
                    "화재위험 지도와 관서별 순찰 동선",
                    "예측 결과와 실제 이동 경로를 한 화면에")
    map_img = figs / "map_route.png"
    if map_img.exists():
        # 지도는 세로로 길다. 폭만 맞추면 장표 아래로 넘친다.
        from PIL import Image
        try:
            w_px, h_px = Image.open(map_img).size
            ratio = w_px / h_px
        except Exception:                                # noqa: BLE001
            ratio = 0.85
        max_h = 4.7
        width = min(6.5, max_h * ratio)
        pic = picture(s_map, map_img, Inches(0.7), Inches(2.15), Inches(width))
        if pic is not None:
            pic.width, pic.height = Inches(width), Inches(width / ratio)
    cards = [
        ("구역 단위 화재위험",
         f"· {grid_m}m 구역 단위 화재위험 예측\n"
         "· 과거 화재 · 주변 구역 확산 · 대상물 용도 · 업종 구성"),
        ("관서별 출발·복귀 동선",
         "· 119안전센터 출발·복귀\n"
         "· 선: 실제 도로 주행거리 · 번호: 방문 순서"),
        ("계획서에 그대로 첨부",
         "· 이 지도가 순찰계획서의 붙임으로 들어감"),
    ]
    y = 2.3
    for head, body in cards:
        band(s_map, Inches(7.6), Inches(y), Inches(5.0), Inches(1.45))
        textbox(s_map, Inches(7.85), Inches(y + 0.14), Inches(4.5), Inches(0.38),
                head, size=14.5, bold=True, color=RED)
        textbox(s_map, Inches(7.85), Inches(y + 0.58), Inches(4.5), Inches(0.8),
                body, size=12, color=MUTED, spacing=1.1)
        y += 1.62

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
          f"· 그 안의 점검 대상 {tk.get('cost_if_all', 0):,.0f}개소\n"
          f"· 1위 구역 한 곳 소요 "
          f"{(alloc.get('top1_grid') or {}).get('inspection_cost', 0):,.0f}건 > "
          f"가용 {alloc.get('budget_visits', 0):,}건"),
         ("소요와 효과의 동시 계산",
          "· 구역마다 점검 소요와 잡히는 화재를 함께 셈\n"
          f"· 소요 합계 {alloc.get('budget_visits', 0):,}건 이내에서\n"
          "  잡히는 화재가 가장 큰 묶음 선택\n"
          f"· {shown_grids:,}개 구역 · 화재 {pct(shown_cap)} 포착 "
          f"({alloc.get('gain_pp', 0):+.1f}%p)\n"
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
             ("채울 수 있는 칸만 채움",
              f"· {extra.get('ledger_fields', 0)}개 칸 중 건물동수·점포수·"
              "소방시설·\n   관서거리·취약요소 자동 입력"),
             ("빈칸은 비워 둠",
              "· 건축물대장·주민등록 연계 시 채워지는 칸\n"
              "· 개인정보라 넣지 않는 칸")],
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
    s9 = section(prs, "6. 검증 결과",
                 f"{tr[0]}~{tr[-1]}년 학습, {year}년 예측",
                 f"{year}년 자료는 학습에 미사용")
    picture(s9, figs / "fig_capture_curve.png", Inches(0.8), Inches(2.25), Inches(7.3))
    x = Inches(8.5)
    kpi(s9, x, Inches(2.3), Inches(4.0), pct(h["model_capture"]),
        f"위험 상위 {k}% 구역이 담은 실제 화재",
        (f"95% 신뢰구간 {m_ci['lo']:.1%} ~ {m_ci['hi']:.1%}"
         if m_ci and m_ci.get("lo") == m_ci.get("lo")
         else f"단순 기준 {pct(h['baseline_capture'])}"))
    kpi(s9, x, Inches(4.0), Inches(4.0), f"{h['model_lift']:.2f}배",
        "아무 데나 갔을 때 대비",
        (f"전년 화재 순으로 갈 때보다 {d['point_pp']:+.1f}%p"
         if d else ""), color=BLUE)
    kpi(s9, x, Inches(5.7), Inches(4.0), pct(t["model"]["pei"][key], 0),
        "도달 가능한 최선 대비",
        "실제 화재를 다 알고 줄 세운 값을 100으로 볼 때", color=GREEN)

    # ---- 10 어디까지 확인했나 ----
    s10 = section(prs, "6. 검증 결과 (계속)",
                  "타 지역·타 관할 적용 검증 4건",
                  "")
    rows = [["확인한 것", "질문", f"상위 {k}% 포착", "결과"]]
    rows.append(["미래 예측", f"{year}년을 맞히는가", pct(h["model_capture"]),
                 (f"전년 화재 순 대비 {d['point_pp']:+.1f}%p"
                  if d else f"{h['delta_pp']:+.1f}%p")])
    if "logo" in ev:
        lg = ev["logo"]
        rows.append(["관할 제외", "특정 구·군만 잘 맞는 것은 아닌가",
                     f"{pct(lg['capture_min'])} ~ {pct(lg['capture_max'])}",
                     f"평균 {pct(lg['capture_mean'])}"])
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
    table(s10, Inches(0.8), Inches(2.35), Inches(11.8), Inches(2.5), rows,
          col_widths=[2.8, 5.0, 2.4, 3.0], size=13.5)
    picture(s10, figs / "fig_monthly_risk.png", Inches(0.9), Inches(5.0), Inches(6.4))
    band(s10, Inches(7.7), Inches(5.0), Inches(4.9), Inches(1.9))
    textbox(s10, Inches(7.95), Inches(5.2), Inches(4.4), Inches(1.6),
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
        s_bt = section(prs, "6. 검증 결과 (계속)",
                       f"{bh.get('train_upto', 0)}년 자료 기준 "
                       f"{bh.get('year', 0)}년 회고 검증",
                       "그해 이전 자료만으로 계획했을 때의 포착 결과")
        rows = [["순찰 구역", "관내 비중", "우리 계획", "작년 화재 순", "무작위",
                 "그해 화재 중"]]
        for r in bt["years"][0]["by_patrol_size"]:
            rows.append([f"{r['patrol_grids']}개", f"{r['share_of_city']:.1%}",
                         f"{r['model_fires']:.0f}건", f"{r['baseline_fires']:.0f}건",
                         f"{r['random_fires']:.1f}건", f"{r['capture_share']:.1%}"])
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
        _g = bh.get("gain_ci", {})
        band(s_bt, Inches(0.8), Inches(4.85), Inches(11.8), Inches(0.55))
        textbox(s_bt, Inches(1.0), Inches(4.93), Inches(11.4), Inches(0.4),
                f"작년 화재 순으로 같은 {bh.get('patrol_grids', 0)}곳을 골랐다면 "
                f"{bh.get('baseline_fires', 0):.0f}건, 차이 "
                f"{bh.get('gain_over_baseline', 0):+.0f}건, 95% 신뢰구간 "
                f"{_g.get('lo', 0):+.0f} ~ {_g.get('hi', 0):+.0f}건입니다.",
                size=12.5)



    # ---- 8-2 차별성 ----
    s_diff = section(prs, "7. 차별성",
                     "예측 다음 단계: 배분 · 동선 · 문서",
                     "화재위험 예측은 2016년 애틀랜타 사례로 이미 존재. "
                     "그 다음 단계 세 가지")
    cards = [
        ("① 위험 순서 배열의 한계",
         "구역마다 점검 소요가 다릅니다.\n"
         f"· 상위 {k}% = {tk.get('n_grids_selected', 0):,}개 구역 · "
         f"{tk.get('cost_if_all', 0):,.0f}개소\n"
         f"· 1위 구역 소요 "
         f"{(alloc.get('top1_grid') or {}).get('inspection_cost', 0):,.0f}건 > "
         f"가용 {alloc.get('budget_visits', 0):,}건\n\n"
         "그래서 기준을 바꿨습니다.\n"
         f"· 소요 합계 {alloc.get('budget_visits', 0):,}건 이내에서\n"
         "· 잡히는 화재 합계가 가장 큰\n"
         "  구역 묶음을 선택\n\n"
         f"→ 같은 인력, {shown_grids:,}개 구역, "
         f"{alloc.get('gain_pp', 0):+.1f}%p",
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
         "  자료 밖이면 ‘확인 필요’ 표시",
         ""),
    ]
    x0, w_card = 0.8, 3.87
    for i, (head, body, foot) in enumerate(cards):
        x = Inches(x0 + i * (w_card + 0.16))
        # 카드가 짧아진 만큼 아래가 비어 위로 쏠려 보인다. 가운데로 내린다.
        top = 2.75
        band(s_diff, x, Inches(top), Inches(w_card), Inches(3.55))
        textbox(s_diff, x + Inches(0.22), Inches(top + 0.2), Inches(w_card - 0.44),
                Inches(0.7), head, size=14, bold=True, color=RED)
        textbox(s_diff, x + Inches(0.22), Inches(top + 0.85), Inches(w_card - 0.44),
                Inches(2.4), body, size=11.5, color=INK, spacing=1.24)
        if foot:
            textbox(s_diff, x + Inches(0.22), Inches(top + 2.8),
                    Inches(w_card - 0.44), Inches(0.5), foot, size=10.5,
                    color=MUTED)


    # ---- 미국 사례 비교 ----
    s_us = section(prs, "7. 차별성 (계속)",
                   "해외 사례 비교: 애틀랜타 Firebird",
                   "미국 NFPA 모범사례 선정 시스템 (KDD 2016)")
    rows = [["", "Firebird (애틀랜타, 2016)", "불씨예보 (울산, 2026)"],
            ["분석 단위", "상업용 건물 5,000여 개소",
             f"{grid_m}m 구역 {t['model']['n_grids']:,}개"],
            ["자료", "8종 결합 (건물대장·화재·인구 등)", "소방안전 빅데이터 8종 + 기상 + 법령"],
            ["예측 성능", "상업용 화재 70% 이상 예측\n(오경보율 20% 기준)",
             f"위험 상위 20% 구역이 화재 {pct(h['model_capture'])} 포착\n"
             f"(95% 신뢰구간 {m_ci.get('lo', 0):.0%} ~ {m_ci.get('hi', 0):.0%})"],
            ["검증 방식", "시간분할 (학습 이후 화재로 검증)",
             "시간분할 + 관할제외 + 타 지역 + 주소 정밀도"],
            ["산출물", "위험점수 · 지도 시각화",
             "위험지도 · 인력 제약 배분 · 관서별 동선 ·\n공문 계획서 · 법령 질의응답"],
            ["인력 제약", "**미해결**: 논문에 “19,397개는 현 인력이\n감당할 수 있는 수준을 훨씬 넘는다”고 기술",
             f"**해결**: 가용 인력 안에서 배분,\n동일 인력 대비 {alloc.get('gain_pp', 0):+.1f}%p 개선"]]
    table(s_us, Inches(0.8), Inches(2.3), Inches(11.8), Inches(3.9), rows,
          col_widths=[2.2, 4.8, 4.8], size=11.5)
    band(s_us, Inches(0.8), Inches(6.35), Inches(11.8), Inches(0.75),
         RGBColor(0xEC, 0xF8, 0xF2))
    textbox(s_us, Inches(1.05), Inches(6.48), Inches(11.3), Inches(0.5),
            "Firebird 논문도 “현 인력으로 감당할 수 없다”고 지적했으나 풀지는 "
            "않았습니다.", size=13, bold=True, color=GREEN)

    # ---- 11 기대효과 ----
    s11 = section(prs, "8. 기대효과 및 활용방안",
                  "순찰 효율 개선과 근거 기록", "")
    # 이 서비스의 결과물은 순찰 경로다. 머리 지표도 순찰에서 시작한다.
    _bh = (extra.get("backtest") or {}).get("headline", {})
    if _bh:
        kpi(s11, Inches(0.8), Inches(2.4), Inches(3.8),
            f"{_bh.get('capture_share', 0):.1%}", "순찰 구역 안에서 난 화재",
            f"관내 {_bh.get('share_of_city', 0):.1%}"
            f"({_bh.get('patrol_grids', 0)}개 구역)만 돌았을 때 · "
            f"{_bh.get('year', 0)}년 회고")
    elif alloc and "gain_pp" in alloc:
        kpi(s11, Inches(0.8), Inches(2.4), Inches(3.8),
            f"{alloc['gain_pp']:+.1f}%p", "같은 인력 기준 포착률 개선",
            f"인력에 맞춘 배분 vs 상위 {k}% 방식")
    if alloc and "gain_pp" in alloc:
        kpi(s11, Inches(4.85), Inches(2.4), Inches(3.8),
            f"{alloc['gain_pp']:+.1f}%p", "같은 인력 기준 점검 포착률",
            f"인력에 맞춘 배분 vs 상위 {k}% 방식 · "
            f"무작위 대비 {h['model_lift']:.2f}배", color=BLUE)
    hc = summary.get("hydrant_coverage", {})
    kpi(s11, Inches(8.9), Inches(2.4), Inches(3.7),
        f"{summary.get('n_blind_spots', 0)}개",
        "고위험 · 소방용수 사각 구역",
        f"전체 구역의 {pct(hc.get('share_without_hydrant'))}에 소화전 없음",
        color=GREEN)
    textbox(s11, Inches(0.8), Inches(4.3), Inches(11.8), Inches(2.4),
            "· 예방순찰   119안전센터별 출동 계획, 목적별 순찰 6종, 월별 순찰 강도\n"
            "· 예방점검   화재안전조사 대상 우선순위를 자동으로 정하고, 공문 서식 계획서로 바로 결재\n"
            "· 소방용수 정책   고위험인데 소화전이 없는 구역을 신설 우선순위의 객관적 근거로\n"
            "· 행정 지원   신규 대원·신규 부임지에서 위험 판단 근거와 법령 조문을 함께 제공\n"
            "· 확산   세종 적용으로 확인. 공개데이터만 쓰므로 별도 운영비가 들지 않음",
            size=15.5, spacing=1.65)

    # ---- 12 한계 ----
    s12 = section(prs, "9. 한계와 향후 계획",
                  "확인된 한계와 대응 방안",
                  "")
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
         "격자 단위 주소 확보 시 재측정"],
        ["점검 이력을 붙일 수 없음",
         "결합할 키가 없다는 것을 실제로 확인해 기록", "대상물 관리번호 포함 자료 요청"],
        ["순찰 횟수를 근무편성과 잇지 못함",
         "월 위험계수로 주차별 횟수까지는 산출",
         "관서 교대·인원 편성 자료와 연계"],
        ["단순 기준 대비 개선폭이 크지 않음",
         f"누적 화재만으로도 {pct(probe0.get('capture_top20'))}임을 먼저 공개",
         "가치는 설명·확장·인력배분에 있음"],
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
    textbox(s13, Inches(0.9), Inches(4.9), Inches(11.5), Inches(1.4),
            f"울산광역시 {manifest.get('panel', {}).get('grids', 0):,}개 구역 · "
            f"화재 {manifest.get('panel', {}).get('total_fires', 0):,.0f}건으로 검증\n\n"
            "박용준 · 아주대학교 산업공학과 석사과정\n"
            "github.com/dragonzzuny/Fire_bigdata", size=17, color=MUTED)

    return prs




def collect_extra(cfg, city: str) -> dict:
    """장표에 들어갈 부가 수치. 없으면 0 으로 두고 그 자리를 비운다."""
    import pandas as pd

    out: dict = {}
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

    prs = build(cfg, ev, summary, manifest, figs, dataset_rows(cfg),
                collect_extra(cfg, city))
    out = cfg.paths.outputs / f"불씨예보_발표자료_{city}_{year}.pptx"
    prs.save(out)
    print(f"발표자료 저장: {out}  ({len(prs.slides._sldIdLst)}장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
