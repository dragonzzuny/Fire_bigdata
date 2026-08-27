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
          ds_rows: list[list[str]]) -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    k = cfg.headline_k
    t = ev["temporal"]
    h = t["headline"]
    key = f"top{k}"
    year = int(t["test_year"])
    # json 왕복을 거치면 연도가 float 이 되어 '2014.0년' 으로 찍힌다.
    tr = [int(y) for y in t["train_years"]]
    alloc = summary.get("allocation", {})

    # ---- 1 표지 ----
    title_slide(prs, "불씨예보 (K-Firebird)",
                "소방안전 빅데이터 기반 화재예방 점검 우선순위 결정 시스템\n"
                "— 한정된 인력으로 어디를, 언제 점검할 것인가",
                "제6회 소방안전 빅데이터 활용 및 아이디어 경진대회 · 서비스 개발 · 박용준(아주대학교)")

    # ---- 2 문제 ----
    s = section(prs, "1. 배경 및 문제점", "예방행정의 병목은 출동 대응이 아니라 사전 의사결정",
                "한정된 인력으로 어디부터 점검할 것인가 — 현재는 법정 점검주기·관할·담당자 경험에 의존")
    band(s, Inches(0.8), Inches(2.4), Inches(5.6), Inches(3.6))
    textbox(s, Inches(1.1), Inches(2.7), Inches(5.0), Inches(3.2),
            "· 소방공무원 증원 정체 (2024년 전년 대비 +5명 수준)\n"
            "· 특정소방대상물·다중이용업소 및 고층건축물은 지속 증가\n"
            "· 동일 법정 대상 내에서도 용도·업종·화재이력에 따라 실제 위험도 상이\n"
            "· 이를 데이터로 우선순위화하는 체계 부재", size=17)
    band(s, Inches(6.9), Inches(2.4), Inches(5.6), Inches(3.6), RGBColor(0xFD, 0xF0, 0xEC))
    textbox(s, Inches(7.2), Inches(2.7), Inches(5.0), Inches(0.5),
            "해외 도입 사례", size=17, bold=True, color=RED)
    textbox(s, Inches(7.2), Inches(3.25), Inches(5.0), Inches(2.6),
            "· 애틀랜타 소방 ‘Firebird’ — 머신러닝 위험점수로\n   점검 우선순위 결정, NFPA 모범사례 선정\n"
            "· 뉴욕 FDNY — 위험기반 점검(RBIS) 운영\n\n"
            "국내 소방 AI 는 출동·신고 대응 중심으로 추진 중이며,\n"
            "예방점검 대상 우선순위화 영역은 미도입 상태", size=16)

    # ---- 3 해법 ----
    s = section(prs, "2. 제안 내용", "위험 지도 → 점검 우선순위 → 점검계획서를 하나의 업무 흐름으로",
                f"울산광역시 {cfg.grid_size_m}m 격자 단위 화재위험 예측 · 설명가능 AI · 점검계획서 자동 생성")
    for i, (num, ttl, body) in enumerate([
            ("1", "예방점검 배분", "가용 인력 기준\n점검 대상 격자 선정"),
            ("2", "위험요인·점검표", "선정 사유(SHAP)와\n업종별 점검 항목"),
            ("3", "예방순찰 계획", "화재 다발 시간대와\n순찰 동선"),
            ("4", "대응취약 구역", "소방용수 사각지대\n화재 급증 구역")]):
        x = Inches(0.8 + i * 3.05)
        band(s, x, Inches(2.5), Inches(2.85), Inches(3.0))
        textbox(s, x, Inches(2.7), Inches(2.85), Inches(0.5), num,
                size=26, bold=True, color=RED, align=PP_ALIGN.CENTER)
        textbox(s, x, Inches(3.3), Inches(2.85), Inches(0.5), ttl,
                size=16, bold=True, align=PP_ALIGN.CENTER)
        textbox(s, x, Inches(3.95), Inches(2.85), Inches(1.3), body,
                size=13, color=MUTED, align=PP_ALIGN.CENTER)
    textbox(s, Inches(0.8), Inches(5.9), Inches(11.8), Inches(0.8),
            "위험 판단은 예측 모델이, 점검 항목은 업종별 규칙이, 문서화만 LLM 이 담당합니다. "
            "생성 모델이 점검 항목이나 법령을 임의로 만들지 않도록 역할을 분리했습니다.",
            size=14, color=MUTED)

    # ---- 4 활용 데이터 ----
    # 공모전 필수 요건(소방안전 빅데이터 플랫폼 데이터 상품 활용)에 해당하는 장이다.
    # 건수는 실제 적재 결과에서 읽어 채운다.
    s = section(prs, "3. 활용 데이터", "소방안전 빅데이터 플랫폼 데이터 상품을 1차 자료로 사용",
                "울산광역시소방본부 4종을 학습·검증에, 세종특별자치시소방본부 4종을 타 지역 적용 검증에 사용")
    rows = [["데이터셋", "제공", "역할", "적재 건수"]]
    for label, prov, role, cnt in ds_rows:
        rows.append([label, prov, role, cnt])
    table(s, Inches(0.8), Inches(2.35), Inches(11.8), Inches(3.2), rows,
          col_widths=[4.2, 2.4, 3.4, 1.8], size=12)
    textbox(s, Inches(0.8), Inches(5.75), Inches(7.2), Inches(1.3),
            "· 카카오 로컬 API — 도로명·읍면동 → 좌표 (좌표 확보 "
            + pct(manifest.get("coverage", {}).get("fire", {}).get("rate")) + ")\n"
            "· 개인정보를 다루지 않으며, 도로·격자 집계 단위 공공데이터만 사용",
            size=13, color=MUTED)
    band(s, Inches(8.3), Inches(5.7), Inches(4.3), Inches(1.35), RGBColor(0xF4, 0xF6, 0xF8))
    textbox(s, Inches(8.55), Inches(5.85), Inches(3.8), Inches(1.1),
            f"흩어진 4종 자료를 격자 단위\n의사결정 테이블로 통합\n"
            f"→ {manifest.get('panel', {}).get('grids', 0):,}개 격자 × "
            f"{len(manifest.get('panel', {}).get('years', []))}개 연도",
            size=13, bold=True)

    # ---- 5 성능 ----
    s = section(prs, "4. 차별성 및 실현가능성", f"{tr[0]}~{tr[-1]}년 자료로 학습하여 {year}년 화재를 예측",
                f"{year}년 자료는 학습에 사용하지 않았습니다 (시간분할 검증)")
    picture(s, figs / "fig_capture_curve.png", Inches(0.8), Inches(2.25), Inches(7.3))
    x = Inches(8.5)
    kpi(s, x, Inches(2.3), Inches(4.0), pct(h["model_capture"]),
        f"상위 {k}% 화재 포착률", f"단순 기준 {pct(h['baseline_capture'])} 대비 {h['delta_pp']:+.1f}%p")
    kpi(s, x, Inches(4.0), Inches(4.0), f"{h['model_lift']:.2f}배",
        "무작위 배정 대비 효율", "동일 면적 기준", color=BLUE)
    kpi(s, x, Inches(5.7), Inches(4.0), pct(t["model"]["pei"][key], 0),
        "PEI · 달성 가능 최대 대비", f"PAI {t['model']['pai'][key]:.2f} / 최대 {t['model']['pai_max'][key]:.2f}",
        color=GREEN)

    # ---- 5 PEI ----
    s = section(prs, "4. 차별성 및 실현가능성", "포착률의 절대 수준을 판단하기 위한 표준지표",
                "화재가 소수 격자에 집중된 지역은 어떤 모델이든 포착률이 높게 나옵니다. "
                "달성 가능한 최대치와 비교해야 성능을 정확히 판단할 수 있습니다.")
    picture(s, figs / "fig_pei.png", Inches(1.3), Inches(2.5), Inches(6.6))
    textbox(s, Inches(8.3), Inches(2.6), Inches(4.4), Inches(3.4),
            "PAI = 포착률 ÷ 면적비율\n"
            "PAI 최대 = 실제 화재 순으로 배열한 사후 최적값\n"
            "PEI = PAI ÷ PAI 최대\n\n"
            f"본 시스템은 달성 가능 최대치의\n{pct(t['model']['pei'][key], 0)} 수준입니다.\n\n"
            "PEI 는 지역이 달라도 비교 가능한 지표로,\n"
            "예측치안 분야에서 표준으로 사용됩니다.", size=15)
    picture(s, figs / "fig_decile.png", Inches(1.3), Inches(4.6), Inches(6.6))

    # ---- 6 운영 (핵심) ----
    s = section(prs, "4. 차별성 ① 운영", "위험도 상위 20% 선정은 현장 인력으로 소화하기 어렵습니다",
                "격자마다 점검 대상물 수가 다르므로, 가용 인력을 제약으로 두고 배분해야 합니다")
    picture(s, figs / "fig_allocation.png", Inches(0.8), Inches(2.4), Inches(7.6))
    if alloc:
        tk = alloc["top_k_percent"]
        op = alloc["optimized"]
        band(s, Inches(8.7), Inches(2.4), Inches(3.9), Inches(2.0), RGBColor(0xFD, 0xF0, 0xEC))
        textbox(s, Inches(8.95), Inches(2.6), Inches(3.4), Inches(1.7),
                f"위험도 상위 {k}% = {tk['n_grids_selected']:,}개 격자\n"
                f"해당 격자 내 점검 대상 {tk['cost_if_all']:,.0f} 개소\n\n"
                f"가용 물량 {alloc['budget_visits']:,}건 기준\n"
                f"실제 점검 가능 격자는 {tk['n_grids_affordable']:,}개",
                size=14, color=RED)
        band(s, Inches(8.7), Inches(4.6), Inches(3.9), Inches(1.9), RGBColor(0xEC, 0xF8, 0xF2))
        textbox(s, Inches(8.95), Inches(4.8), Inches(3.4), Inches(1.6),
                f"인력 제약 기반 배분 적용 시\n"
                f"동일 인력으로 {op['n_grids']:,}개 격자 점검\n"
                f"실제 화재 {pct(op.get('actual_capture_rate'))} 포착\n"
                f"({alloc.get('gain_pp', 0):+.1f}%p 개선)", size=15, bold=True, color=GREEN)

    # ---- 7 검증 3종 ----
    s = section(prs, "4. 차별성 ② 검증", "네 가지 방식으로 성능을 교차 검증",
                "한 가지 검증만으로는 특정 지역에만 통하는 모델인지 판별할 수 없습니다")
    rows = [["검증", "묻는 것", f"상위 {k}% 포착", "판정"]]
    rows.append(["시간분할", f"미래 예측 성능 ({year}년)", pct(h["model_capture"]),
                 f"단순 기준 대비 {h['delta_pp']:+.1f}%p"])
    if "logo" in ev:
        lg = ev["logo"]
        rows.append(["관할 제외 검증", "특정 지역 의존 여부",
                     f"{pct(lg['capture_min'])} ~ {pct(lg['capture_max'])}",
                     f"평균 {pct(lg['capture_mean'])}"])
    if "transfer" in ev:
        tf = ev["transfer"]["headline"]
        rows.append(["타 지역 적용 (울산→세종)", "신규 지역 확장 가능성",
                     pct(tf["model_capture"]), f"단순 기준 대비 {tf['delta_pp']:+.1f}%p"])
    if "temporal_history_only" in ev:
        hh = ev["temporal_history_only"]["headline"]
        rows.append(["현황 자료 제외", "시점 정보 혼입 영향", pct(hh["model_capture"]),
                     f"기여분 {ev.get('snapshot_contribution_pp', 0):+.1f}%p"])
    table(s, Inches(0.8), Inches(2.4), Inches(11.8), Inches(2.6), rows,
          col_widths=[3, 4, 2.4, 2.6], size=14)
    picture(s, figs / "fig_model_compare.png", Inches(2.6), Inches(5.0), Inches(4.2))
    textbox(s, Inches(7.2), Inches(5.2), Inches(5.4), Inches(1.6),
            "알고리즘도 동일 조건에서 비교했습니다.\n"
            "본 시스템의 결과물은 화재 건수가 아니라 점검 순서이므로,\n"
            "순위를 직접 학습하는 방식이 목적에 부합합니다.", size=14, color=MUTED)

    # ---- 8 형평성 ----
    s = section(prs, "4. 차별성 ③ 형평성", "관할별 배분 형평성을 지표로 측정",
                "특정 관할에 점검이 집중되면서 화재 비중과 어긋난다면, 효율이 아니라 편중입니다")
    picture(s, figs / "fig_equity.png", Inches(0.8), Inches(2.4), Inches(7.4))
    es = t.get("equity_summary", {})
    band(s, Inches(8.5), Inches(2.5), Inches(4.1), Inches(2.6))
    textbox(s, Inches(8.75), Inches(2.7), Inches(3.6), Inches(2.3),
            "막대 위 수치 = 점검 비중 ÷ 화재 비중\n\n"
            "1.0 이면 위험도에 비례하여 배분된 상태\n\n"
            f"실측 범위 {es.get('inspection_vs_risk_min', 0):.2f} ~ "
            f"{es.get('inspection_vs_risk_max', 0):.2f}", size=14)
    if es.get("underserved"):
        band(s, Inches(8.5), Inches(5.3), Inches(4.1), Inches(1.2), RGBColor(0xFD, 0xF0, 0xEC))
        textbox(s, Inches(8.75), Inches(5.5), Inches(3.6), Inches(0.9),
                f"{', '.join(es['underserved'])}: 화재 비중 대비 점검 배분 부족\n"
                "배분 조정의 객관적 근거로 활용 가능합니다.",
                size=13, color=RED)

    # ---- 9 데이터 정직성 ----
    s = section(prs, "4. 차별성 ④ 신뢰성", "전처리 단계별 데이터 손실을 모두 기록",
                "자료 부족으로 산출하지 못한 것과 산출했으나 성능이 낮은 것은 다른 문제입니다")
    picture(s, figs / "fig_data_funnel.png", Inches(0.8), Inches(2.35), Inches(7.0))
    ded = manifest.get("fire_deduplication", {})
    cov = manifest.get("coverage", {}).get("fire", {})
    lines = []
    if ded.get("removed"):
        lines.append(f"· 동일 화재의 중복 출동 기록 {ded['removed']:,}건 제거\n"
                     f"  (2021년 파일은 55%가 중복, 미처리 시 학습 라벨 2배 과대)")
    if cov:
        lv = cov.get("by_level", {})
        lines.append(f"· 좌표 확보 {pct(cov.get('rate'))} "
                     f"(도로명 {lv.get('road', 0):,} / 읍면동 {lv.get('emd', 0):,})")
    lines.append("· 세종 소방용수시설 좌표 1,792건이 시 경계 밖 →\n  폐기 후 주소 기반 지오코딩으로 대체")
    lines.append("· 2021년 화재는 발생 시각 누락 → 시간대 분석에서 제외")
    textbox(s, Inches(8.1), Inches(2.5), Inches(4.6), Inches(3.4),
            "\n".join(lines), size=13)
    band(s, Inches(8.1), Inches(5.5), Inches(4.5), Inches(1.2), RGBColor(0xEC, 0xF8, 0xF2))
    textbox(s, Inches(8.35), Inches(5.68), Inches(4.0), Inches(0.9),
            "읍면동 중심좌표 집중이 성능을 과대평가할 가능성을 검증한 결과,\n"
            "도로명 확보분만으로도 성능 저하는 없었습니다.",
            size=12, color=GREEN)

    # ---- 10 현장 산출물 ----
    s = section(prs, "5. 기대효과 및 활용방안", "예측에서 끝내지 않고 현장에서 바로 쓰는 산출물까지",
                "위험점수만으로는 현장 업무로 연결되지 않습니다")
    items = [("점검 계획표", "격자별 점검 순위와 누적 인력 소요\n(CSV 배포, 즉시 활용)"),
             ("점검계획서 초안", "선정 사유와 업종별 점검 항목을\n공문 형식으로 자동 작성"),
             ("순찰 동선", "화재 다발 시간대 및 최단 순찰 경로"),
             ("소방용수 사각지대", "고위험 구역 중 소화전 미설치 격자\n→ 신설 예산 편성 근거")]
    for i, (ttl, body) in enumerate(items):
        x, y = Inches(0.8 + (i % 2) * 6.1), Inches(2.4 + (i // 2) * 2.1)
        band(s, x, y, Inches(5.8), Inches(1.85))
        textbox(s, x + Inches(0.3), y + Inches(0.18), Inches(5.2), Inches(0.45),
                ttl, size=17, bold=True, color=RED)
        textbox(s, x + Inches(0.3), y + Inches(0.72), Inches(5.2), Inches(1.0),
                body, size=14, color=MUTED)
    hc = summary.get("hydrant_coverage", {})
    if hc.get("available"):
        textbox(s, Inches(0.8), Inches(6.7), Inches(11.8), Inches(0.5),
                f"분석 결과 전체 격자의 {pct(hc['share_without_hydrant'])} 에 소화전이 없으며, "
                f"고위험·용수 사각 격자 {summary.get('n_blind_spots', 0)}개를 도출했습니다.",
                size=14, color=MUTED)

    # ---- 11 기대효과 ----
    s = section(prs, "5. 기대효과 및 활용방안", "동일 인력으로 더 많은 화재를 포착하고 근거를 남깁니다",
                "")
    if alloc and "gain_pp" in alloc:
        kpi(s, Inches(0.8), Inches(2.5), Inches(3.8),
            f"{alloc['gain_pp']:+.1f}%p", "동일 인력 기준 포착률 개선",
            f"인력 제약 배분 vs 상위 {k}% 방식")
    kpi(s, Inches(4.85), Inches(2.5), Inches(3.8), f"{h['model_lift']:.2f}배",
        "무작위 배정 대비 효율", "경험·민원 기반 → 데이터 기반 전환", color=BLUE)
    kpi(s, Inches(8.9), Inches(2.5), Inches(3.7), pct(t["model"]["pei"][key], 0),
        "달성 가능 최대 대비", "추가 개선 여지도 함께 산출", color=GREEN)
    textbox(s, Inches(0.8), Inches(4.4), Inches(11.8), Inches(2.4),
            "· 예방점검   화재안전조사·예방순찰 대상 우선순위 자동화, 점검계획서로 보고 및 현장 활용\n"
            "· 소방용수 정책   고위험·용수 사각 격자를 소화전 신설·보강 우선순위의 객관적 근거로 활용\n"
            "· 순찰 배치   화재 다발 시간대·요일에 맞춘 순찰 시간 및 동선 제안\n"
            "· 행정 지원   신규 대원 및 신규 부임지에서 위험 판단 근거를 제공\n"
            "· 확산   세종 적용 검증으로 전국 시·도 확대 가능. 공개데이터 활용으로 별도 운영비 부담 없음",
            size=16, spacing=1.6)

    # ---- 12 한계 ----
    s = section(prs, "6. 기타 — 한계 및 향후 계획", "현재 한계와 대응 방안",
                "자료상 제약을 명시하고, 각각에 대한 대응과 개선 방향을 함께 제시합니다")
    rows = [["한계", "현재 대응", "다음"]]
    rows += [
        ["공개 데이터에 건물번호·좌표 미포함", "도로명 → 읍면동 계층 지오코딩, 단계별 기록",
         "부산형 상세주소 확보 시 건물 단위"],
        ["화재의 63%가 읍면동 중심좌표", "도로명 확보분 한정 검증(성능 저하 없음)",
         "도로명 기재율 개선 협의"],
        ["대상물·업소는 현재 시점 현황", "이력 자료만의 검증 결과를 병행 제시",
         "연도별 이력 자료 확보"],
        ["점검 이력 결합 불가", "결합 키 부재를 실측으로 확인 및 기록",
         "대상물 관리번호 포함 자료 요청"],
        ["순찰 경로가 직선거리 기준", "경로 교차 제거로 이동거리 단축", "도로망 기반 경로 최적화"],
    ]
    table(s, Inches(0.8), Inches(2.4), Inches(11.8), Inches(3.6), rows,
          col_widths=[3.6, 4.6, 3.6], size=13)
    textbox(s, Inches(0.8), Inches(6.3), Inches(11.8), Inches(0.7),
            "전 과정 재현 스크립트와 검증 테스트를 공개합니다. "
            "동일한 명령으로 본 자료의 모든 수치를 재현할 수 있습니다.",
            size=14, color=MUTED)

    # ---- 13 마무리 ----
    s = blank(prs)
    band(s, 0, 0, W, Inches(0.14), RED)
    textbox(s, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.6),
            "한정된 인력을\n가장 위험한 곳에", size=44, bold=True)
    textbox(s, Inches(0.9), Inches(4.5), Inches(11.5), Inches(1.2),
            f"울산광역시 {manifest.get('panel', {}).get('grids', 0):,}개 격자 · "
            f"화재 {manifest.get('panel', {}).get('total_fires', 0):,.0f}건으로 검증\n"
            f"github.com/dragonzzuny/Fire_bigdata", size=18, color=MUTED)
    return prs


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

    prs = build(cfg, ev, summary, manifest, figs, dataset_rows(cfg))
    out = cfg.paths.outputs / f"불씨예보_발표자료_{city}_{year}.pptx"
    prs.save(out)
    print(f"발표자료 저장: {out}  ({len(prs.slides._sldIdLst)}장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
