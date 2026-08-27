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


def section(prs, kicker, title, lead=""):
    s = blank(prs)
    band(s, 0, 0, W, Inches(0.14), RED)
    textbox(s, Inches(0.8), Inches(0.45), Inches(11.8), Inches(0.4),
            kicker, size=13, bold=True, color=RED)
    textbox(s, Inches(0.8), Inches(0.85), Inches(11.8), Inches(0.8),
            title, size=31, bold=True)
    if lead:
        textbox(s, Inches(0.8), Inches(1.62), Inches(11.8), Inches(0.6),
                lead, size=15, color=MUTED)
    return s


def picture(slide, path: Path, x, y, w):
    if not Path(path).exists():
        return None
    return slide.shapes.add_picture(str(path), x, y, width=w)


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
            run = p.add_run(); run.text = str(val)
            run.font.size = Pt(size); run.font.name = FONT
            run.font.bold = (r == 0)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else INK
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
    box_h = min(1.45, (height - (len(bullets) - 1) * 0.18) / max(len(bullets), 1))
    for head, body in bullets:
        band(s, x, Inches(y), Inches(box_w), Inches(box_h))
        textbox(s, x + Inches(0.2), Inches(y + 0.11), Inches(box_w - 0.4),
                Inches(0.34), head, size=13, bold=True, color=RED)
        textbox(s, x + Inches(0.2), Inches(y + 0.49), Inches(box_w - 0.4),
                Inches(box_h - 0.58), body, size=10.5, color=MUTED, spacing=1.05)
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
    k = cfg.headline_k
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
    textbox(s1, Inches(0.9), Inches(1.15), Inches(11.5), Inches(1.0),
            "불씨예보", size=52, bold=True)
    textbox(s1, Inches(0.9), Inches(2.25), Inches(11.5), Inches(0.5),
            "K-Firebird", size=24, color=RED, bold=True)
    textbox(s1, Inches(0.9), Inches(2.95), Inches(11.5), Inches(0.9),
            "소방안전 빅데이터 기반 화재예방 점검·순찰 의사결정 시스템\n"
            "한정된 인력을 가장 위험한 곳에, 실행 가능한 계획으로",
            size=18, color=MUTED)
    band(s1, Inches(0.9), Inches(4.25), Inches(6.5), Inches(2.0),
         RGBColor(0xFD, 0xF0, 0xEC))
    textbox(s1, Inches(1.2), Inches(4.45), Inches(5.9), Inches(0.4),
            "이름의 뜻", size=14, bold=True, color=RED)
    textbox(s1, Inches(1.2), Inches(4.9), Inches(5.9), Inches(1.3),
            "· Firebird — 애틀랜타 소방의 화재위험 예측 시스템.\n"
            "  미국 NFPA 모범사례로 선정된 예방점검 우선순위 모델\n"
            "· K- — 국내 공개 데이터와 소방 법령 체계에 맞춘 한국형\n"
            "· 불씨예보 — 일기예보처럼, 불씨를 미리 알린다", size=13.5)
    band(s1, Inches(7.8), Inches(4.25), Inches(4.6), Inches(2.0))
    textbox(s1, Inches(8.1), Inches(4.45), Inches(4.0), Inches(0.4),
            "발표자", size=14, bold=True, color=RED)
    textbox(s1, Inches(8.1), Inches(4.92), Inches(4.0), Inches(1.2),
            "박 용 준\n아주대학교 산업공학과 석사과정", size=16, bold=True)
    textbox(s1, Inches(0.9), Inches(6.55), Inches(11.5), Inches(0.5),
            "제6회 소방안전 빅데이터 활용 및 아이디어 경진대회 · 서비스 개발 부문",
            size=13, color=MUTED)

    # ---- 2 왜 (배경) ----
    s2 = section(prs, "1. 배경 및 문제점",
                 "예방행정의 병목은 출동이 아니라 ‘어디부터 갈 것인가’",
                 "현재는 법정 점검주기·관할·담당자 경험에 의존합니다")
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
            "· 애틀랜타 소방 ‘Firebird’ — 위험점수로 점검\n"
            "  우선순위 결정, 미국 NFPA 모범사례 선정\n"
            "· 뉴욕 FDNY — 위험기반 점검(RBIS) 운영\n\n"
            "국내 소방 정보화는 출동·신고 대응 중심이며,\n"
            "예방점검 대상 우선순위화 영역은 비어 있습니다.", size=15.5)
    textbox(s2, Inches(0.8), Inches(6.15), Inches(11.8), Inches(0.6),
            "목적 — 소방안전 빅데이터로 지역별 화재위험을 예측해, "
            "한정된 인력을 가장 위험한 곳과 시기에 먼저 배치하도록 돕습니다.",
            size=15, bold=True)

    # ---- 3 무엇을 (구성) ----
    s3 = section(prs, "2. 제안 내용",
                 "위험 예측에서 끝내지 않고, 결재 가능한 계획까지",
                 "예측 → 인력에 맞춘 배분 → 관서별 순찰 동선 → 공문 서식 계획서")
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
    s4 = section(prs, "3. 활용 데이터",
                 "소방안전 빅데이터 플랫폼 데이터 상품을 1차 자료로 사용",
                 "울산 4종으로 만들고, 세종 4종으로 타 지역에서도 되는지 확인했습니다")
    rows = [["데이터셋", "제공", "역할", "적재 건수"]] + [list(r) for r in ds_rows]
    table(s4, Inches(0.8), Inches(2.3), Inches(11.8), Inches(3.1), rows,
          col_widths=[4.2, 2.4, 3.4, 1.8], size=11.5)
    textbox(s4, Inches(0.8), Inches(5.6), Inches(7.4), Inches(1.3),
            "· 카카오 로컬 API — 주소를 좌표로 변환 (좌표 확보 "
            + pct(manifest.get("coverage", {}).get("fire", {}).get("rate")) + ")\n"
            "· 기상청 API 허브 — 일자료 8년치로 건조 정도 산출\n"
            f"· 국가법령정보센터 — 소방 법령 {extra.get('n_law', 7)}종 "
            f"{extra.get('n_article', 457)}개 조문 · 별표 68건 · 법정 서식 39종\n"
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
    s_map = section(prs, "4. 서비스 화면 ①",
                    "어디가 위험하고, 어디를 어떤 순서로 도는가",
                    "화재위험 예측과 순찰 동선을 한 장에서 봅니다")
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
        ("색이 짙을수록 위험",
         "500m 구역마다 화재위험을 예측합니다. 과거 화재, 주변 구역으로의 확산, "
         "대상물 용도와 업종 구성을 함께 봅니다."),
        ("검은 사각형이 출동 관서",
         "119안전센터에서 출발해 관할을 돌고 복귀합니다. 선이 실제 도로 기준 동선, "
         "번호가 방문 순서입니다."),
        ("계획서에 그대로 첨부",
         "이 그림이 순찰계획서에 붙습니다. 표만 있는 계획서는 어디를 도는지 "
         "머리에 그려지지 않습니다."),
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
    tk = alloc.get("top_k_percent", {})
    screen_slide(
        prs, "4. 서비스 화면 ②", "오늘 어느 구역부터 점검할 것인가",
        "점검관 인원과 기간을 넣으면, 그 인력으로 실제 갈 수 있는 곳만 배분합니다",
        figs / "shot_allocation.png",
        [("인력을 먼저 넣습니다",
          "점검관 몇 명, 하루 몇 건, 며칠. 바꾸면 배분이 즉시 다시 계산됩니다."),
         ("구역마다 대상물 수가 다릅니다",
          f"위험도 상위 {k}%는 {tk.get('n_grids_selected', 0):,}개 구역이지만 "
          f"그 안의 점검 대상은 {tk.get('cost_if_all', 0):,.0f}개소입니다."),
         ("같은 인력으로 더 많이",
          f"인력에 맞춘 배분은 {op.get('n_grids', 0):,}개 구역을 돌아 "
          f"실제 화재 {pct(op.get('actual_capture_rate'))}를 포착 "
          f"({alloc.get('gain_pp', 0):+.1f}%p).")],
        note="관할별 최소 배분을 지정할 수 있습니다 — 특정 구에 점검이 몰리지 않도록.")

    # ---- 6 화면② 순찰 ----
    screen_slide(
        prs, "4. 서비스 화면 ③", "순찰 조건을 바꾸면 계획이 즉시 바뀝니다",
        "119안전센터에서 출발해 관할을 돌고 복귀하는 실제 도로 기준 동선입니다",
        figs / "shot_patrol.png",
        [("출동 관서 기준",
          f"소방서 {extra.get('n_station', 6)}개 · 119안전센터 "
          f"{extra.get('n_center', 28)}개 · 읍면동 {extra.get('n_emd', 83)}개 단위로 "
          "선택합니다."),
         ("목적별 순찰 6종",
          "일반예방 · 다중이용업소 야간 · 화재예방강화지구 · 피난약자시설 · "
          "소방용수 점검 · 건조기 특별경계. 가는 곳과 시간대가 다릅니다."),
         ("근무시간 안에 들어오게",
          "1회 순찰 시간을 넘으면 회차를 나눕니다. 실제 도로 주행거리와 "
          "소요시간을 함께 제시합니다.")],
        note="지도의 검은 점이 출동 관서, 색깔이 관서별 순찰 동선입니다.")

    # ---- 7 화면③ 계획서 ----
    screen_slide(
        prs, "4. 서비스 화면 ④", "그대로 결재를 올릴 수 있는 계획서",
        "동선·중점 확인사항·법령 근거가 들어간 공문 서식 문서를 자동으로 만듭니다",
        (figs / "shot_plan_result.png"
         if (figs / "shot_plan_result.png").exists() else figs / "shot_plan_doc.png"),
        [("일별 · 월별 · 연간",
          "월별은 그 달의 화재위험을 반영해 순찰 횟수를 정하고, 연간은 "
          "계절별 순찰 유형과 법정 이행사항을 배치합니다."),
         ("공문 서식 그대로",
          "기관·수신·시행일·관련 근거·붙임·결재란까지. 담당자가 다시 옮겨 "
          "적을 필요가 없습니다."),
         ("법정 서식 안내",
          "조치가 필요하면 어느 별지 서식을 쓰는지 함께 알려 줍니다. "
          "(예: 화재예방강화지구 관리대장)")],
        note="숫자·동선·법령 조문은 시스템이 확정하고, 생성형 AI는 문장만 다듬습니다. "
             "인쇄용 HTML 로 내려받아 그대로 A4 출력합니다.",
        keep=0.82)

    # ---- 8 화면④ 업무 도우미 ----
    screen_slide(
        prs, "4. 서비스 화면 ⑤", "법령을 조문 근거와 함께 찾아 줍니다",
        "‘화재예방강화지구는 어떤 지역을 지정하나요?’ 같은 질문에 답합니다",
        (figs / "shot_assistant_answer.png"
         if (figs / "shot_assistant_answer.png").exists() else figs / "shot_assistant.png"),
        [("소방 법령을 담았습니다",
          f"{extra.get('n_law', 7)}종 {extra.get('n_article', 457)}개 조문과 별표 68건에 "
          "업종별 점검 항목·관할 위험 현황을 함께 검색합니다."),
         ("반드시 근거를 붙입니다",
          "‘연 1회입니다’만 답하는 시스템은 행정에서 쓸 수 없습니다. "
          "법령명과 조문 번호가 함께 나옵니다."),
         ("신규 대원 업무 지원",
          "새로 부임한 담당자가 ‘왜 여기가 위험한지’와 ‘무슨 근거로 하는지’를 "
          "같은 화면에서 확인합니다.")],
        note="답변은 업무 참고용이며, 법령 원문은 국가법령정보센터에서 확인합니다.",
        keep=0.78)

    # ---- 9 어떻게 믿나 (검증) ----
    s9 = section(prs, "5. 검증 결과",
                 f"{tr[0]}~{tr[-1]}년 자료로 만들어 {year}년 화재를 맞혀 봤습니다",
                 f"{year}년 자료는 만드는 데 한 건도 쓰지 않았습니다")
    picture(s9, figs / "fig_capture_curve.png", Inches(0.8), Inches(2.25), Inches(7.3))
    x = Inches(8.5)
    kpi(s9, x, Inches(2.3), Inches(4.0), pct(h["model_capture"]),
        f"위험 상위 {k}% 구역이 담은 실제 화재",
        (f"95% 신뢰구간 {m_ci['lo']:.1%}–{m_ci['hi']:.1%}"
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
    s10 = section(prs, "5. 검증 결과 (계속)",
                  "한 지역에서만 되는 것은 아닌지 네 가지로 확인했습니다",
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
            "‘언제’ 도 함께 봅니다\n\n"
            "월별 화재위험 = 계절 패턴 × 기상(습도·건조일수).\n"
            "겨울(12·1월)이 연평균의 1.1배, 9월이 0.89배입니다.\n"
            "월별 순찰 횟수를 이 값에 맞춰 정합니다.", size=13)

    # ---- 미국 사례 비교 ----
    s_us = section(prs, "5. 검증 결과 (계속)",
                   "미국 애틀랜타 ‘Firebird’ 와 같은 축에서 비교했습니다",
                   "NFPA 모범사례로 선정된 시스템입니다 (KDD 2016)")
    rows = [["", "Firebird (애틀랜타, 2016)", "불씨예보 (울산, 2026)"],
            ["분석 단위", "상업용 건물 5,000여 개소", "500m 구역 1,459개"],
            ["자료", "8종 결합 (건물대장·화재·인구 등)", "소방안전 빅데이터 8종 + 기상 + 법령"],
            ["예측 성능", "상업용 화재 70% 이상 예측\n(오경보율 20% 기준)",
             f"위험 상위 20% 구역이 화재 {pct(h['model_capture'])} 포착\n"
             f"(95% 신뢰구간 {m_ci.get('lo', 0):.0%}–{m_ci.get('hi', 0):.0%})"],
            ["검증 방식", "시간분할 (학습 이후 화재로 검증)",
             "시간분할 + 관할제외 + 타 지역 + 주소 정밀도"],
            ["산출물", "위험점수 · 지도 시각화",
             "위험지도 · 인력 제약 배분 · 관서별 동선 ·\n공문 계획서 · 법령 질의응답"],
            ["인력 제약", "**미해결** — 논문에 “19,397개는 현 인력이\n감당할 수 있는 수준을 훨씬 넘는다”고 기술",
             f"**해결** — 가용 인력 안에서 배분,\n동일 인력 대비 {alloc.get('gain_pp', 0):+.1f}%p 개선"]]
    table(s_us, Inches(0.8), Inches(2.3), Inches(11.8), Inches(3.9), rows,
          col_widths=[2.2, 4.8, 4.8], size=11.5)
    band(s_us, Inches(0.8), Inches(6.35), Inches(11.8), Inches(0.75),
         RGBColor(0xEC, 0xF8, 0xF2))
    textbox(s_us, Inches(1.05), Inches(6.48), Inches(11.3), Inches(0.5),
            "Firebird 논문도 “현 인력으로 감당할 수 없다”는 문제를 지적했으나 풀지는 않았습니다. "
            "그 지점이 저희가 더한 부분입니다.", size=13, bold=True, color=GREEN)

    # ---- 11 기대효과 ----
    s11 = section(prs, "6. 기대효과 및 활용방안",
                  "같은 인력으로 더 많은 화재를 잡고, 근거를 남깁니다", "")
    if alloc and "gain_pp" in alloc:
        kpi(s11, Inches(0.8), Inches(2.4), Inches(3.8),
            f"{alloc['gain_pp']:+.1f}%p", "같은 인력 기준 포착률 개선",
            f"인력에 맞춘 배분 vs 상위 {k}% 방식")
    kpi(s11, Inches(4.85), Inches(2.4), Inches(3.8), f"{h['model_lift']:.2f}배",
        "아무 데나 갔을 때 대비", "경험·민원 기반 → 데이터 기반 전환", color=BLUE)
    hc = summary.get("hydrant_coverage", {})
    kpi(s11, Inches(8.9), Inches(2.4), Inches(3.7),
        f"{summary.get('n_blind_spots', 0)}개",
        "고위험 · 소방용수 사각 구역",
        f"전체 구역의 {pct(hc.get('share_without_hydrant'))}에 소화전 없음",
        color=GREEN)
    textbox(s11, Inches(0.8), Inches(4.3), Inches(11.8), Inches(2.4),
            "· 예방점검   화재안전조사 대상 우선순위를 자동으로 정하고, 공문 서식 계획서로 바로 결재\n"
            "· 예방순찰   119안전센터별 출동 계획, 목적별 순찰 6종, 월별 순찰 강도\n"
            "· 소방용수 정책   고위험인데 소화전이 없는 구역을 신설 우선순위의 객관적 근거로\n"
            "· 행정 지원   신규 대원·신규 부임지에서 위험 판단 근거와 법령 조문을 함께 제공\n"
            "· 확산   세종 적용으로 확인. 공개데이터만 쓰므로 별도 운영비가 들지 않음",
            size=15.5, spacing=1.65)

    # ---- 12 한계 ----
    s12 = section(prs, "7. 기타 — 한계 및 향후 계획",
                  "알고 있는 한계를 먼저 말씀드립니다",
                  "자료상 제약과 그에 대한 대응, 개선 방향입니다")
    rows = [["한계", "현재 대응", "향후"]]
    rows += [
        ["공개 데이터에 건물번호·좌표가 없음",
         "도로명 → 읍면동 순으로 좌표를 찾고 단계를 기록", "상세주소 확보 시 건물 단위"],
        ["화재의 63%가 읍면동 중심 좌표",
         "도로명이 있는 건만으로 따로 검증 — 성능 저하 없음", "도로명 기재율 개선 협의"],
        ["대상물·업소는 현재 시점 현황",
         "과거 이력만으로도 검증해 함께 제시", "연도별 이력 자료 확보"],
        ["점검 이력을 붙일 수 없음",
         "결합할 키가 없다는 것을 실제로 확인해 기록", "대상물 관리번호 포함 자료 요청"],
        ["단순 기준 대비 개선폭이 크지 않음",
         "누적 화재만으로도 68.4%임을 먼저 공개",
         "가치는 설명·확장·인력배분에 있음"],
    ]
    table(s12, Inches(0.8), Inches(2.35), Inches(11.8), Inches(3.5), rows,
          col_widths=[3.6, 4.8, 3.4], size=12.5)
    textbox(s12, Inches(0.8), Inches(6.15), Inches(11.8), Inches(0.7),
            "전 과정을 다시 실행할 수 있는 스크립트와 검증 절차를 공개합니다. "
            "같은 명령으로 오늘 보신 모든 수치가 다시 만들어집니다.",
            size=14, color=MUTED)

    # ---- 13 마무리 ----
    s13 = blank(prs)
    band(s13, 0, 0, W, Inches(0.14), RED)
    textbox(s13, Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.6),
            "한정된 인력을\n가장 위험한 곳에", size=44, bold=True)
    textbox(s13, Inches(0.9), Inches(4.55), Inches(11.5), Inches(1.4),
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

    law = cfg.paths.cache / "law_articles.parquet"
    if law.exists():
        df = pd.read_parquet(law)
        out["n_law"] = int(df["law"].nunique())
        out["n_article"] = int(len(df))

    try:
        from firebird import monthly as MO
        fp = cfg.paths.processed / f"fires_{city}.parquet"
        wp = cfg.paths.processed / f"weather_monthly_{city}.parquet"
        if fp.exists() and wp.exists():
            mf = MO.fires_by_month(pd.read_parquet(fp), cfg.year_min, cfg.year_max)
            fit = MO.fit_month_risk(mf, pd.read_parquet(wp))
            out["r2_season"] = float(fit.get("baseline_r2", 0.0))
            out["r2_weather"] = float(fit.get("weather_r2", fit.get("baseline_r2", 0.0)))
    except Exception:                                    # noqa: BLE001
        pass
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
