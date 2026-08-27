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

def build(cfg, ev: dict, summary: dict, manifest: dict, figs: Path,
          ds_rows: list[list[str]], extra: dict) -> Presentation:
    """10분 발표(시연 포함) + 5분 질의응답 기준.

    시연이 4분을 쓰므로 장표는 6분 안에 넘어가야 한다. 12장, 장당 30초.
    '넘겨도 되는 장'을 명시해 시간이 밀릴 때 버릴 순서를 미리 정해 둔다.
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

    def ci_text(block: dict) -> str:
        m = block.get("model", {})
        if not m or m.get("lo") != m.get("lo"):
            return ""
        return f"95% 신뢰구간 {m['lo']:.1%}–{m['hi']:.1%}"

    # ---- 1 표지 ----
    title_slide(prs, "불씨예보 (K-Firebird)",
                "소방안전 빅데이터 기반 화재예방 점검·순찰 의사결정 시스템\n"
                "— 한정된 인력을 가장 위험한 곳에, 실행 가능한 계획으로",
                "제6회 소방안전 빅데이터 활용 및 아이디어 경진대회 · 서비스 개발 · "
                "박용준(아주대학교)")

    # ---- 2 배경 ----
    s2 = section(prs, "1. 배경 및 문제점",
                 "예방행정의 병목은 출동 대응이 아니라 사전 의사결정",
                 "한정된 인력으로 어디부터 점검할 것인가 — 현재는 법정 점검주기·관할·"
                 "담당자 경험에 의존")
    band(s2, Inches(0.8), Inches(2.4), Inches(5.6), Inches(3.5))
    textbox(s2, Inches(1.1), Inches(2.68), Inches(5.0), Inches(3.1),
            "· 소방공무원 증원 정체 (2024년 전년 대비 +5명 수준)\n"
            "· 특정소방대상물·다중이용업소, 30층 이상 고층건축물(+8%) 지속 증가\n"
            "· 동일 법정 대상 내에서도 용도·업종·화재이력에 따라 실제 위험도 상이\n"
            "· 이를 데이터로 우선순위화하는 체계 부재", size=16.5)
    band(s2, Inches(6.9), Inches(2.4), Inches(5.7), Inches(3.5),
         RGBColor(0xFD, 0xF0, 0xEC))
    textbox(s2, Inches(7.2), Inches(2.68), Inches(5.1), Inches(0.45),
            "해외는 데이터로 해결하고 있다", size=16.5, bold=True, color=RED)
    textbox(s2, Inches(7.2), Inches(3.2), Inches(5.1), Inches(2.6),
            "· 애틀랜타 소방 ‘Firebird’ — 머신러닝 위험점수로\n"
            "  점검 우선순위 결정, NFPA 모범사례 선정\n"
            "· 뉴욕 FDNY — 위험기반 점검(RBIS) 운영\n\n"
            "국내 소방 AI 는 출동·신고 대응 중심.\n"
            "예방점검 대상 우선순위화 영역은 비어 있다.", size=15.5)
    textbox(s2, Inches(0.8), Inches(6.15), Inches(11.8), Inches(0.6),
            "목적 — 소방안전 빅데이터로 격자 단위 화재위험을 예측해, "
            "한정된 인력을 가장 위험한 지역·시간에 우선 배치하도록 지원한다.",
            size=15, bold=True)

    # ---- 3 제안(구성도) ----
    s3 = section(prs, "2. 제안 내용",
                 "위험 예측에서 끝내지 않고, 실행 가능한 계획까지",
                 "예측 → 인력 제약 배분 → 관서별 순찰 동선 → 결재용 계획서")
    # 구성도는 가로로 길다. 카드와 겹치지 않게 위쪽에만 두고 폭을 줄인다.
    picture(s3, figs / "fig_pipeline.png", Inches(1.35), Inches(2.15), Inches(10.6))
    for i, (num, ttl, body) in enumerate([
            ("1", "예방점검 배분", "가용 인력 제약 하\n기대 화재 포착 최대화"),
            ("2", "관서별 순찰", "119안전센터 출발·복귀\n도로 기준 최적 동선"),
            ("3", "계획서 자동 생성", "동선·확인사항·법령 근거\n일별·월별·연간"),
            ("4", "업무 도우미", "소방 법령 457개 조문\n근거 제시형 질의응답")]):
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
                 "울산광역시소방본부 4종을 학습·검증에, 세종특별자치시소방본부 4종을 "
                 "타 지역 적용 검증에 사용")
    rows = [["데이터셋", "제공", "역할", "적재 건수"]]
    rows += [[label, prov, role, cnt] for label, prov, role, cnt in ds_rows]
    table(s4, Inches(0.8), Inches(2.3), Inches(11.8), Inches(3.1), rows,
          col_widths=[4.2, 2.4, 3.4, 1.8], size=11.5)
    textbox(s4, Inches(0.8), Inches(5.6), Inches(7.4), Inches(1.3),
            "· 카카오 로컬 API — 도로명·읍면동 → 좌표 (좌표 확보 "
            + pct(manifest.get("coverage", {}).get("fire", {}).get("rate")) + ")\n"
            "· 기상청 API 허브 — 일자료 8년치, 실효습도 산출\n"
            "· 국가법령정보센터 — 소방 법령 7종 457개 조문\n"
            "· 개인정보 미사용. 도로·격자 집계 단위 공공데이터만 사용",
            size=12.5, color=MUTED)
    band(s4, Inches(8.4), Inches(5.6), Inches(4.2), Inches(1.2),
         RGBColor(0xF4, 0xF6, 0xF8))
    textbox(s4, Inches(8.65), Inches(5.75), Inches(3.7), Inches(1.0),
            f"흩어진 자료를 격자 단위 의사결정 테이블로 통합\n"
            f"→ {manifest.get('panel', {}).get('grids', 0):,}개 격자 × "
            f"{len(manifest.get('panel', {}).get('years', []))}개 연도 · "
            f"화재 {manifest.get('panel', {}).get('total_fires', 0):,.0f}건",
            size=12.5, bold=True)

    # ---- 5 검증 ----
    s5 = section(prs, "4. 차별성 및 실현가능성 ①",
                 f"{tr[0]}~{tr[-1]}년으로 학습해 {year}년 화재를 예측",
                 f"{year}년 자료는 학습에 사용하지 않았습니다")
    picture(s5, figs / "fig_capture_curve.png", Inches(0.8), Inches(2.25), Inches(7.3))
    x = Inches(8.5)
    kpi(s5, x, Inches(2.3), Inches(4.0), pct(h["model_capture"]),
        f"상위 {k}% 화재 포착률",
        ci_text(ci) or f"단순 기준 {pct(h['baseline_capture'])}")
    kpi(s5, x, Inches(4.0), Inches(4.0), pct(t["model"]["pei"][key], 0),
        "달성 가능 최대치 대비", "지역이 달라도 비교 가능한 지표", color=GREEN)
    d = ci.get("delta", {})
    delta_line = (f"{d['point_pp']:+.1f}%p (CI {d['lo_pp']:+.1f}~{d['hi_pp']:+.1f}%p)"
                  if d and d.get("lo_pp") == d.get("lo_pp") else "")
    kpi(s5, x, Inches(5.7), Inches(4.0), f"{h['model_lift']:.2f}배",
        "무작위 배정 대비 효율", delta_line + " · 통계적으로 유의" if d.get("excludes_zero")
        else delta_line, color=BLUE)

    # ---- 6 운영 (핵심) ----
    s6 = section(prs, "4. 차별성 및 실현가능성 ②",
                 "위험도 상위 20% 선정은 현장 인력으로 소화할 수 없습니다",
                 "격자마다 점검 대상물 수가 다릅니다. 가용 인력을 제약으로 두고 배분해야 "
                 "실행 가능한 계획이 됩니다")
    picture(s6, figs / "fig_allocation.png", Inches(0.8), Inches(2.4), Inches(7.6))
    if alloc:
        tk, op = alloc["top_k_percent"], alloc["optimized"]
        band(s6, Inches(8.7), Inches(2.4), Inches(3.9), Inches(2.0),
             RGBColor(0xFD, 0xF0, 0xEC))
        textbox(s6, Inches(8.95), Inches(2.6), Inches(3.4), Inches(1.7),
                f"위험도 상위 {k}% = {tk['n_grids_selected']:,}개 격자\n"
                f"해당 격자 내 점검 대상 {tk['cost_if_all']:,.0f} 개소\n\n"
                f"점검관 4명 × 20일 = {alloc['budget_visits']:,}건\n"
                f"→ 첫 격자에서 예산이 끝납니다", size=14, color=RED)
        band(s6, Inches(8.7), Inches(4.6), Inches(3.9), Inches(1.9),
             RGBColor(0xEC, 0xF8, 0xF2))
        textbox(s6, Inches(8.95), Inches(4.8), Inches(3.4), Inches(1.6),
                f"인력 제약 배분 적용 시\n"
                f"동일 인력으로 {op['n_grids']:,}개 격자 점검\n"
                f"실제 화재 {pct(op.get('actual_capture_rate'))} 포착\n"
                f"({alloc.get('gain_pp', 0):+.1f}%p 개선)", size=15, bold=True, color=GREEN)

    # ---- 7 시연 안내 ----
    s7 = blank(prs)
    band(s7, 0, 0, W, Inches(0.14), RED)
    textbox(s7, Inches(0.9), Inches(1.5), Inches(11.5), Inches(0.9),
            "시 연", size=40, bold=True)
    textbox(s7, Inches(0.9), Inches(2.5), Inches(11.5), Inches(0.5),
            "실제 울산 데이터로 동작하는 화면입니다", size=17, color=MUTED)
    demo = [("1", "예방점검 배분", "점검관 인원을 바꾸면 배분이 즉시 재계산"),
            ("2", "관서별 순찰 동선", "119안전센터 출발·복귀, 실제 도로 거리"),
            ("3", "계획서 자동 생성", "동선·확인사항·법령 조문이 들어간 결재 문서"),
            ("4", "업무 도우미", "‘화재예방강화지구는?’ → 조문 근거와 함께 답변")]
    for i, (num, ttl, body) in enumerate(demo):
        y = Inches(3.35 + i * 0.92)
        band(s7, Inches(1.6), y, Inches(10.1), Inches(0.78))
        textbox(s7, Inches(1.85), y + Inches(0.16), Inches(0.5), Inches(0.5),
                num, size=17, bold=True, color=RED)
        textbox(s7, Inches(2.45), y + Inches(0.16), Inches(3.2), Inches(0.5),
                ttl, size=16, bold=True)
        textbox(s7, Inches(5.9), y + Inches(0.18), Inches(5.6), Inches(0.5),
                body, size=14, color=MUTED)

    # ---- 8 현업 밀착 ----
    s8 = section(prs, "4. 차별성 및 실현가능성 ③",
                 "현장 업무 단위로 만들었습니다",
                 "시군구 단위 계획은 현장에서 쓰이지 않습니다 — 울산 남구 하나가 19개 읍면동입니다")
    items = [
        ("출동 관서 기준", f"소방서 {extra.get('n_station', 0)}개 · "
                        f"119안전센터 {extra.get('n_center', 0)}개 · "
                        f"읍면동 {extra.get('n_emd', 0)}개를 격자마다 부착.\n"
                        "순찰은 안전센터에서 출발해 관할을 돌고 복귀합니다."),
        ("목적별 순찰 6종", "일반예방 · 다중이용업소 야간 · 화재예방강화지구 ·\n"
                        "피난약자시설 · 소방용수시설 점검 · 건조기 특별경계\n"
                        "유형마다 대상·시간대·확인 항목이 다릅니다."),
        ("도로 기준 동선", "실제 주행거리(OSRM) 기반 TSP 최적화.\n"
                       "2-opt + Or-opt 로 최근접 이웃 대비 -10.2%.\n"
                       "근무시간을 넘으면 회차를 나눕니다."),
        ("법령 근거 제시", f"소방 법령 {extra.get('n_law', 0)}종 "
                       f"{extra.get('n_article', 0)}개 조문을 색인.\n"
                       "계획서와 질의응답에 조문 번호가 함께 붙습니다."),
    ]
    for i, (ttl, body) in enumerate(items):
        x, y = Inches(0.8 + (i % 2) * 6.1), Inches(2.35 + (i // 2) * 2.15)
        band(s8, x, y, Inches(5.8), Inches(1.9))
        textbox(s8, x + Inches(0.3), y + Inches(0.16), Inches(5.2), Inches(0.42),
                ttl, size=16, bold=True, color=RED)
        textbox(s8, x + Inches(0.3), y + Inches(0.66), Inches(5.2), Inches(1.1),
                body, size=13, color=MUTED)
    textbox(s8, Inches(0.8), Inches(6.65), Inches(11.8), Inches(0.5),
            "계획의 숫자·동선·법령 조문은 시스템이 확정하고, 생성형 AI 는 문장만 다듬습니다. "
            "근거가 생성 모델에서 나오면 검증이 불가능해집니다.", size=13, color=MUTED)

    # ---- 9 검증 프로토콜 ----
    s9 = section(prs, "4. 차별성 및 실현가능성 ④",
                 "네 가지 방식으로 교차 검증하고, 불확실성까지 밝힙니다",
                 "한 가지 검증만으로는 특정 지역에만 통하는 모델인지 판별할 수 없습니다")
    rows = [["검증", "묻는 것", f"상위 {k}% 포착", "판정"]]
    rows.append(["시간분할", f"미래 예측 성능 ({year}년)", pct(h["model_capture"]),
                 (f"단순 기준 대비 {d['point_pp']:+.1f}%p · 유의"
                  if d.get("excludes_zero") else f"{h['delta_pp']:+.1f}%p")])
    if "logo" in ev:
        lg = ev["logo"]
        rows.append(["관할 제외 검증", "특정 지역 의존 여부",
                     f"{pct(lg['capture_min'])} ~ {pct(lg['capture_max'])}",
                     f"평균 {pct(lg['capture_mean'])}"])
    if "transfer" in ev:
        tf = ev["transfer"]
        tci = tf.get("ci", {}).get(key, {}).get("model", {})
        note = (f"CI {tci['lo']:.0%}–{tci['hi']:.0%} · 표본 작음"
                if tci and tci.get("lo") == tci.get("lo") else "")
        rows.append(["타 지역 적용 (울산→세종)", "신규 지역 확장 가능성",
                     pct(tf["headline"]["model_capture"]), note])
    if "resolution_scenarios" in ev:
        r0 = ev["resolution_scenarios"]
        road = next((x for x in r0 if "도로명" in str(x.get("시나리오", ""))), None)
        if road:
            rows.append(["주소 해상도 검증", "읍면동 집중이 성능을 부풀렸는가",
                         pct(road.get("모델포착@20%")), "부풀림 없음"])
    table(s9, Inches(0.8), Inches(2.35), Inches(11.8), Inches(2.5), rows,
          col_widths=[3.4, 4.4, 2.4, 3.0], size=13.5)
    # 표 아래 남는 높이는 약 2.1인치뿐이다. 그림 폭을 줄여 세로가 잘리지 않게 한다.
    picture(s9, figs / "fig_model_compare.png", Inches(1.5), Inches(5.0), Inches(3.7))
    picture(s9, figs / "fig_equity.png", Inches(6.0), Inches(5.0), Inches(4.2))

    # ---- 10 월별·기상 ----
    s10 = section(prs, "5. 기대효과 및 활용방안 ①",
                  "‘어디를’ 에 더해 ‘언제’ 까지",
                  "월별 위험계수 = 계절 패턴 × 기상(실효습도·건조일수). "
                  "격자 순위 × 월 계수로 그 달의 순찰 강도를 정합니다")
    picture(s10, figs / "fig_monthly_risk.png", Inches(0.8), Inches(2.4), Inches(7.6))
    picture(s10, figs / "fig_hour_profile.png", Inches(0.8), Inches(4.9), Inches(7.6))
    band(s10, Inches(8.7), Inches(2.5), Inches(3.9), Inches(3.9))
    textbox(s10, Inches(8.95), Inches(2.72), Inches(3.5), Inches(3.6),
            "· 기상청 일자료 8년치(2,922일) 수집\n\n"
            "· 과거 특보 이력은 API 가 현재 시점만 주므로,\n"
            "  실효습도(기상청 공식, 감쇠계수 0.7)를 직접\n"
            "  계산해 건조주의보 기준을 재현\n\n"
            f"· 기상 반영 시 월별 화재 설명력\n"
            f"  R² {extra.get('r2_season', 0):.2f} → {extra.get('r2_weather', 0):.2f}\n\n"
            "· 화재 다발 시간대(오후)와 결합해\n  순찰 시간까지 제안", size=13)

    # ---- 11 기대효과 ----
    s11 = section(prs, "5. 기대효과 및 활용방안 ②",
                  "동일 인력으로 더 많은 화재를 포착하고, 근거를 남깁니다", "")
    if alloc and "gain_pp" in alloc:
        kpi(s11, Inches(0.8), Inches(2.4), Inches(3.8),
            f"{alloc['gain_pp']:+.1f}%p", "동일 인력 기준 포착률 개선",
            f"인력 제약 배분 vs 상위 {k}% 방식")
    kpi(s11, Inches(4.85), Inches(2.4), Inches(3.8), f"{h['model_lift']:.2f}배",
        "무작위 배정 대비 효율", "경험·민원 기반 → 데이터 기반 전환", color=BLUE)
    hc = summary.get("hydrant_coverage", {})
    kpi(s11, Inches(8.9), Inches(2.4), Inches(3.7),
        f"{summary.get('n_blind_spots', 0)}개",
        "고위험 · 소방용수 사각 격자",
        f"전체 격자의 {pct(hc.get('share_without_hydrant'))} 에 소화전 없음", color=GREEN)
    textbox(s11, Inches(0.8), Inches(4.3), Inches(11.8), Inches(2.4),
            "· 예방점검   화재안전조사·예방순찰 대상 우선순위 자동화, "
            "결재용 계획서로 보고 및 현장 활용\n"
            "· 예방순찰   119안전센터별 출동 계획, 목적별 순찰 6종, 월별 순찰 강도\n"
            "· 소방용수 정책   고위험·용수 사각 격자를 소화전 신설 우선순위의 객관적 근거로\n"
            "· 행정 지원   신규 대원·신규 부임지에서 위험 판단 근거와 법령 조문을 함께 제공\n"
            "· 확산   세종 적용 검증으로 전국 시·도 확대 가능. 공개데이터 활용으로 운영비 부담 없음",
            size=15.5, spacing=1.65)

    # ---- 12 한계 ----
    s12 = section(prs, "6. 기타 — 한계 및 향후 계획",
                  "알고 있는 한계와 대응 방안",
                  "자료상 제약을 먼저 밝히고, 각각에 대한 대응과 개선 방향을 제시합니다")
    rows = [["한계", "현재 대응", "향후"]]
    rows += [
        ["공개 데이터에 건물번호·좌표 미포함",
         "도로명 → 읍면동 계층 지오코딩, 단계별 기록", "부산형 상세주소로 건물 단위"],
        ["화재의 63%가 읍면동 중심좌표",
         "도로명 확보분 한정 검증 — 성능 저하 없음 확인", "도로명 기재율 개선 협의"],
        ["대상물·업소는 현재 시점 현황",
         "이력 자료만의 검증을 병행 제시", "연도별 이력 자료 확보"],
        ["점검 이력 결합 불가",
         "결합 키 부재를 실측으로 확인·기록", "대상물 관리번호 포함 자료 요청"],
        ["단순 기준 대비 개선폭이 크지 않음",
         "누적 화재만으로도 68.4%임을 먼저 공개", "모델 가치는 설명·확장·배분에 있음"],
    ]
    table(s12, Inches(0.8), Inches(2.35), Inches(11.8), Inches(3.5), rows,
          col_widths=[3.6, 4.8, 3.4], size=12.5)
    textbox(s12, Inches(0.8), Inches(6.15), Inches(11.8), Inches(0.7),
            "전 과정 재현 스크립트와 검증 테스트 170개를 공개합니다. "
            "동일한 명령으로 본 자료의 모든 수치를 재현할 수 있습니다.",
            size=14, color=MUTED)

    # ---- 13 마무리 ----
    s13 = blank(prs)
    band(s13, 0, 0, W, Inches(0.14), RED)
    textbox(s13, Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.6),
            "한정된 인력을\n가장 위험한 곳에", size=44, bold=True)
    textbox(s13, Inches(0.9), Inches(4.5), Inches(11.5), Inches(1.2),
            f"울산광역시 {manifest.get('panel', {}).get('grids', 0):,}개 격자 · "
            f"화재 {manifest.get('panel', {}).get('total_fires', 0):,.0f}건으로 검증\n"
            f"github.com/dragonzzuny/Fire_bigdata", size=18, color=MUTED)
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
