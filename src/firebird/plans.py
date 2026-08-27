"""순찰·점검 계획서 생성 — 일별 / 월별 / 연간.

현장에 필요한 것은 위험 점수표가 아니라 **결재를 올릴 수 있는 문서**다.
어느 관서가, 언제, 어디를, 어떤 순서로, 무엇을 보고, 무슨 근거로 하는가가
한 장에 있어야 한다.

이 모듈은 문서를 규칙으로 먼저 완성한다. LLM 은 그 뒤에 문체만 다듬는다.
계획의 숫자·동선·법령 근거가 생성 모델에서 나오면 검증이 불가능해진다.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import llm as L
from .patrol_modes import MODES, PatrolMode, recommended_hours

log = logging.getLogger(__name__)

#: 공문 서식 기본값. 실제 배포 시 소방서명·기안자를 설정에서 받아 채운다.
DOC_DEFAULTS = {
    "기관명": "○○소방서",
    "부서": "예방과",
    "수신": "내부결재",
    "기안자": "",
    "연락처": "",
}


@dataclass
class DocMeta:
    """공문 머리·꼬리에 들어가는 정보.

    계획서는 결재를 받는 문서다. 제목만 있고 수신·근거·붙임·결재란이 없으면
    현장에서 그대로 올릴 수 없어 결국 담당자가 다시 옮겨 적게 된다.
    """
    기관명: str = DOC_DEFAULTS["기관명"]
    부서: str = DOC_DEFAULTS["부서"]
    수신: str = DOC_DEFAULTS["수신"]
    기안자: str = ""
    연락처: str = ""
    문서번호: str = ""
    시행일: date | None = None


#: 공문 항목 번호. chr(0xAC00 + n) 으로 만들면 '가, 각, 갂' 이 나온다.
#: 한글 완성형은 초성·중성·종성이 곱해진 배열이라 1씩 더하면 종성이 붙는다.
HANGUL_ORDINALS = ("가", "나", "다", "라", "마", "바", "사", "아", "자", "차",
                   "카", "타", "파", "하")


def _hangul_ordinal(i: int) -> str:
    return HANGUL_ORDINALS[i] if i < len(HANGUL_ORDINALS) else f"({i + 1})"


def _doc_header(meta: DocMeta, title: str, basis: list[str]) -> list[str]:
    """공문 머리.

    「행정업무의 운영 및 혁신에 관한 규정」 별지 제1호서식(일반기안문)의
    배열을 따른다. 기관명 → 수신 → (경유) → 제목 → 본문 순이다.
    '항목 / 내용' 같은 표 머리를 붙이면 그 순간 공문이 아니라 보고서 표가 된다.

    줄 끝의 공백 두 칸은 마크다운의 줄바꿈 표시다. 이것이 없으면 수신·경유·
    제목이 한 문단으로 붙어 버린다.
    """
    d = meta.시행일 or date.today()
    out = [f"# {title}", "", f"**{meta.기관명}**", ""]
    out.append(f"수신　　{meta.수신}  ")
    out.append("(경유)  ")
    out.append(f"제목　　{title}  ")
    out.append("")

    sub = []
    if meta.문서번호:
        sub.append(f"문서번호 {meta.문서번호}")
    sub.append(f"시행일 {d.year}. {d.month}. {d.day}.")
    if meta.기안자:
        sub.append(f"담당 {meta.부서} {meta.기안자}"
                   + (f" ({meta.연락처})" if meta.연락처 else ""))
    elif meta.부서:
        sub.append(f"담당 {meta.부서}")
    out.append("　　".join(sub))
    out.append("")

    if basis:
        out.append("1. 관련")
        out.append("")
        for i, b in enumerate(basis):
            out.append(f"   {_hangul_ordinal(i)}. {b}")
        out.append("")
    return out


def _doc_footer(meta: DocMeta, attachments: list[str]) -> list[str]:
    """붙임, 종결 표시, 발신 명의, 결재란.

    공문은 마지막 붙임 뒤에 두 칸 띄우고 '끝.' 을 찍는다. 붙임이 없으면
    본문 마지막 글자 뒤에 찍는다. 그 아래에 발신 명의를 둔다.
    """
    out = ["", "---", ""]
    if attachments:
        out.append("붙임")
        out.append("")
        for i, a in enumerate(attachments, 1):
            tail = "  끝." if i == len(attachments) else ""
            out.append(f"   {i}. {a} 1부.{tail}")
    else:
        out.append("   끝.")
    out += ["", f"**{meta.기관명}장**", ""]
    out += ["| 기안 | 검토 | 결재 |", "|---|---|---|",
            "| | | |", "| | | |", ""]
    return out


#: 문서에 반드시 들어가야 하는 격자 정의.
#: 계획서를 처음 받는 사람은 '격자 2331_3456' 이 무엇인지 모른다.
GRID_DEFINITION = (
    "- **격자**: 지역을 가로·세로 500m 정사각형으로 나눈 분석 단위입니다. "
    "공개 데이터에 건물 번호와 좌표가 없어 개별 건물을 특정할 수 없으므로, "
    "주소(도로명 또는 읍면동)를 좌표로 변환해 500m 칸에 모아 집계했습니다. "
    "격자 번호(예: `2331_3456`)는 UTM-K 좌표계의 가로·세로 칸 번호이며, "
    "한 격자는 대략 도보 5~7분 거리의 한 블록 범위에 해당합니다."
)

#: 계절별 권장 순찰 유형. 화재 발생 특성과 법정 업무 시기를 함께 본다.
SEASONAL_MODES = {
    1: "dry_season", 2: "dry_season", 3: "dry_season",     # 봄철 건조·산불
    4: "market", 5: "market", 6: "general",
    7: "general", 8: "general", 9: "vulnerable",
    10: "market", 11: "dry_season", 12: "night_business",  # 연말 다중이용업소
}

#: 연간 계획에 반드시 들어가야 하는 법정 사항. 조문은 실제로 확인한 것만 적는다.
STATUTORY_ITEMS = [
    ("화재예방강화지구 화재안전조사", "연 1회 이상",
     "화재의 예방 및 안전관리에 관한 법률 제18조제3항, 같은 법 시행령 제20조제1항"),
    ("화재예방강화지구 소방훈련·교육", "연 1회 이상 (10일 전 통보)",
     "같은 법 제18조제5항, 같은 법 시행령 제20조제2항·제3항"),
    ("화재예방강화지구 관리대장 작성·관리", "상시",
     "같은 법 시행규칙 제8조"),
]


@dataclass
class PlanContext:
    """계획 한 건을 만드는 데 필요한 모든 것."""
    city_label: str
    year: int
    mode: PatrolMode
    targets: pd.DataFrame                 # 순찰 대상 격자 (출동관서 배정 포함)
    routes: list[pd.DataFrame] = field(default_factory=list)
    #: 계획서에 넣을 지도 이미지 경로(순찰 동선 + 화재위험). 없으면 생략된다.
    map_path: str = ""
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    distance_source: str = ""
    month_plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    law_index: object | None = None
    fires: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------- 법령 근거

#: 순찰·점검 계획의 근거가 될 수 있는 조문인지 가리는 낱말.
#: 이게 없으면 BM25 가 어휘만 보고 엉뚱한 조문을 가져온다. 실제로
#: '일반 예방순찰' 질의에 화재예방안전진단 절차 규정(시행규칙 제41조)이 1순위로 왔다.
RELEVANCE_TERMS = ("순찰", "점검", "화재안전조사", "예방", "소방시설", "피난",
                   "비상구", "방염", "안전관리", "훈련", "교육", "강화지구",
                   "다중이용업", "소방용수", "화기")

#: 근거로 인정할 최소 검색 점수. 낮은 점수는 '우연히 낱말이 겹친' 조문이다.
MIN_LEGAL_SCORE = 8.0


def _sentence_cut(text: str, limit: int = 320) -> str:
    """문장 경계에서 자른다.

    '…다음 각 호의 절차에 따라 ' 처럼 문장 중간에서 끊긴 인용은 결재 문서에
    그대로 나가면 안 된다. 한국어 종결어미(다./함./한다.) 뒤에서 자르고,
    경계를 못 찾으면 인용 자체를 짧게 줄인다.
    """
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return t
    head = t[:limit]
    for end in ("다. ", "함. ", "다.", "함.", ". "):
        pos = head.rfind(end)
        if pos > limit * 0.4:
            return head[:pos + len(end)].strip()
    pos = head.rfind(" ")
    return (head[:pos] if pos > limit * 0.5 else head).strip() + " …"


def find_legal_basis(ctx: PlanContext, extra_terms: list[str] | None = None,
                     top_k: int = 4, *, min_score: float = MIN_LEGAL_SCORE) -> list[dict]:
    """이 계획과 관련된 법령 조문을 찾는다.

    조문 번호 없는 계획서는 결재가 안 난다. 다만 **아무 조문이나 붙이면
    더 나쁘다** — 담당자가 근거를 확인하는 순간 신뢰를 잃는다. 그래서
    (1) 최소 점수, (2) 순찰·점검 관련 낱말 포함 두 조건을 모두 만족한 조문만 쓴다.
    """
    if ctx.law_index is None:
        return []

    out: list[dict] = []
    seen: set[str] = set()

    # 1단계: 순찰 유형이 명시한 근거 조문을 먼저 넣는다.
    #        검색은 어휘만 보므로 '소방시설·피난' 이 나온다는 이유로
    #        화재예방안전진단 절차 규정을 일반 순찰의 근거로 끌어오기도 한다.
    wanted = list(getattr(ctx.mode, "legal_refs", ()) or ())
    if wanted:
        by_ref = {}
        for doc in getattr(ctx.law_index, "docs", []):
            # 같은 조문이 항·호로 나뉘어 여러 문서로 들어올 수 있다.
            # 그대로 두면 계획서에 같은 조문이 두 번 인용된다.
            if getattr(doc, "source", "") == "법령" and doc.ref in wanted:
                if doc.ref not in by_ref or len(doc.text) > len(by_ref[doc.ref].text):
                    by_ref[doc.ref] = doc
        for ref in wanted:
            doc = by_ref.get(ref)
            if doc is None:
                log.info("명시한 근거 조문을 법령 자료에서 찾지 못함: %s", ref)
                continue
            out.append({"ref": doc.ref, "title": doc.title,
                        "excerpt": _sentence_cut(doc.text), "score": None})
            seen.add(doc.ref)

    # 명시 근거가 충분하면 검색으로 더 채우지 않는다.
    # 어휘 검색이 채운 조문은 '있으면 좋은 것'이 아니라 '틀리면 해로운 것'이다.
    # 결재자가 조문을 열어 보고 무관하면 문서 전체의 신뢰가 무너진다.
    if len(out) >= 2 or (wanted and len(out) == len(wanted)):
        return out[:top_k]

    # 2단계: 명시 근거가 없거나 부족할 때만 검색으로 보충한다.
    query = " ".join([ctx.mode.label, ctx.mode.purpose,
                      *ctx.mode.checks, *(extra_terms or [])])
    hits = ctx.law_index.search(query, top_k=top_k * 5)
    for doc, score in hits:
        if doc.ref in seen:
            continue
        if getattr(doc, "source", "") != "법령" or score < min_score:
            continue
        blob = f"{doc.title} {doc.text}"
        if not any(term in blob for term in RELEVANCE_TERMS):
            continue
        out.append({"ref": doc.ref, "title": doc.title,
                    "excerpt": _sentence_cut(doc.text), "score": round(score, 1)})
        seen.add(doc.ref)
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------- 일별

def daily_plan(ctx: PlanContext, plan_date: date, *, shift_hours: tuple[int, int] | None = None
               ) -> dict:
    """하루치 순찰 계획."""
    hours = shift_hours or recommended_hours(ctx.mode, ctx.fires)
    weekday = "월화수목금토일"[plan_date.weekday()]

    teams = []
    for r in ctx.routes:
        if r.empty:
            continue
        depot = r.attrs.get("depot", {})
        row = ctx.summary[(ctx.summary.get("출동관서") == depot.get("name"))] \
            if "출동관서" in ctx.summary.columns else pd.DataFrame()
        teams.append({
            "출동관서": depot.get("name", r.get("출동관서", pd.Series([""])).iloc[0]),
            "격자수": int(len(r)),
            "총_km": float(row["총_km"].iloc[0]) if len(row) else
                     float(r["누적거리_m"].iloc[-1]) / 1000,
            "총_분": float(row["총_분"].iloc[0]) if len(row) else
                     float(r["누적시간_분"].iloc[-1]),
            "동선": r,
        })

    return {
        "kind": "daily",
        "date": plan_date,
        "weekday": weekday,
        "hours": hours,
        "mode": ctx.mode,
        "teams": teams,
        "legal": find_legal_basis(ctx),
        "distance_source": ctx.distance_source,
    }


# ---------------------------------------------------------------- 월별

def month_weeks(year: int, month: int) -> list[tuple[date, date]]:
    """그 달을 주 단위로 나눈다 (월요일 시작)."""
    first = date(year, month, 1)
    last = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
    weeks, cur = [], first
    while cur <= last:
        end = min(cur + timedelta(days=6 - cur.weekday()), last)
        weeks.append((cur, end))
        cur = end + timedelta(days=1)
    return weeks


def monthly_plan(ctx: PlanContext, month: int, *, days_per_week: int = 2) -> dict:
    """한 달치 순찰 계획 — 주차별로 관서·구역을 배분한다."""
    weeks = month_weeks(ctx.year, month)
    mult, grade = 1.0, "보통"
    if not ctx.month_plan.empty:
        row = ctx.month_plan[ctx.month_plan["month"] == month]
        if len(row):
            mult = float(row["위험계수"].iloc[0])
            grade = str(row["등급"].iloc[0]) if "등급" in row else "보통"

    # 위험계수가 높은 달은 순찰 횟수를 늘린다 — 계수를 그대로 운영에 반영한다.
    rounds = max(1, int(round(days_per_week * len(weeks) * mult)))

    by_station = []
    if ctx.routes:
        for r in ctx.routes:
            depot = r.attrs.get("depot", {})
            by_station.append({"출동관서": depot.get("name", ""), "격자수": int(len(r))})
    alloc = pd.DataFrame(by_station)

    schedule = []
    for i, (a, b) in enumerate(weeks):
        assigned = alloc.iloc[i % len(alloc)]["출동관서"] if len(alloc) else ""
        schedule.append({
            "주차": f"{i + 1}주",
            "기간": f"{a.month}/{a.day}–{b.month}/{b.day}",
            "중점 관서": assigned,
            "순찰 횟수": max(1, round(rounds / max(len(weeks), 1))),
        })

    return {
        "kind": "monthly",
        "year": ctx.year, "month": month,
        "risk_multiplier": mult, "risk_grade": grade,
        "mode": ctx.mode,
        "weeks": pd.DataFrame(schedule),
        "station_alloc": alloc,
        "total_rounds": rounds,
        "legal": find_legal_basis(ctx),
    }


# ---------------------------------------------------------------- 연간

def annual_plan(ctx: PlanContext) -> dict:
    """연간 순찰·점검 계획 — 계절별 유형 배치와 법정 사항."""
    rows = []
    for m in range(1, 13):
        key = SEASONAL_MODES.get(m, "general")
        mode = MODES[key]
        mult = 1.0
        if not ctx.month_plan.empty:
            r = ctx.month_plan[ctx.month_plan["month"] == m]
            if len(r):
                mult = float(r["위험계수"].iloc[0])
        h = recommended_hours(mode, ctx.fires)
        rows.append({
            "월": f"{m}월",
            "month": m,
            "중점 순찰": mode.label,
            "권장 시간대": f"{h[0]:02d}–{h[1]:02d}시",
            "위험계수": round(mult, 2),
            "비고": "집중" if mult >= 1.05 else ("완화" if mult <= 0.95 else ""),
        })
    cal = pd.DataFrame(rows)

    quarters = []
    for q in range(4):
        ms = cal.iloc[q * 3:(q + 1) * 3]
        quarters.append({
            "분기": f"{q + 1}분기",
            "기간": f"{q*3+1}–{q*3+3}월",
            "평균 위험계수": round(float(ms["위험계수"].mean()), 2),
            "중점 순찰": " · ".join(dict.fromkeys(ms["중점 순찰"])),
        })

    return {
        "kind": "annual",
        "year": ctx.year,
        "calendar": cal,
        "quarters": pd.DataFrame(quarters),
        "statutory": STATUTORY_ITEMS,
        "legal": find_legal_basis(ctx, ["화재예방강화지구", "화재안전조사", "소방훈련"]),
    }


# ---------------------------------------------------------------- 문서

def _legal_block(legal: list[dict], n: int) -> list[str]:
    """법령 근거 절. 번호를 인자로 받아 앞 절과 이어지게 한다.

    번호 없는 절이 4장과 5장 사이에 끼면 문서 체계가 무너진다.
    """
    if not legal:
        return []
    out = ["", f"## {n}. 법령 근거", ""]
    for i, l in enumerate(legal, 1):
        out.append(f"{i}) **{l['ref']}**")
        out.append(f"   > {l['excerpt']}")
        out.append("")
    return out


def render_daily(plan: dict, ctx: PlanContext, meta: DocMeta | None = None) -> str:
    meta = meta or DocMeta()
    d, mode = plan["date"], plan["mode"]
    h = plan["hours"]
    legal = plan.get("legal", [])
    basis = [_law_citation(l["ref"]) for l in legal[:2]] or \
            ["「화재의 예방 및 안전관리에 관한 법률」 제7조(화재안전조사)"]
    basis.append(f"{ctx.city_label} 화재위험 예측 결과({ctx.year}년 기준)")

    out = _doc_header(meta, f"{d.year}년 {d.month}월 {d.day}일 예방순찰 계획(안)", basis)
    out += [
        "2. 위 호와 관련하여 아래와 같이 예방순찰을 실시하고자 합니다.", "",
        "**가. 순찰 개요**", "",
        "| 구분 | 내용 |", "|---|---|",
        f"| 순찰 종류 | {mode.label} |",
        f"| 일시 | {d.year}. {d.month}. {d.day}.({plan['weekday']}) "
        f"{h[0]:02d}:00 ~ {h[1]:02d}:00 |",
        f"| 대상 지역 | {ctx.city_label} |",
        f"| 투입 관서 | {len(plan['teams'])}개 관서 |",
        f"| 순찰 구역 | {sum(t['격자수'] for t in plan['teams'])}개 격자 |",
        "",
        "**나. 순찰 목적**", "", f"   {mode.purpose}", "",
        "**다. 관서별 순찰 구역**", "",
        "| 출동관서 | 순찰 격자 | 이동거리 | 소요시간 |", "|---|---|---|---|",
    ]
    for t in plan["teams"]:
        out.append(f"| {t['출동관서']} | {t['격자수']}개 | "
                   f"{t['총_km']:.1f} km | {t['총_분']:.0f}분 |")
    tot_km = sum(t["총_km"] for t in plan["teams"])
    out.append(f"| **계** | **{sum(t['격자수'] for t in plan['teams'])}개** | "
               f"**{tot_km:.1f} km** | |")

    out += ["", "**라. 순찰 동선**", ""]
    if ctx.map_path:
        # 표만 있는 계획서는 '어디를 도는지'가 머리에 안 그려진다.
        # 위험 분포와 동선이 겹친 지도를 함께 넣는다.
        out += [f"![순찰 동선도]({ctx.map_path})", "",
                "   ※ 색이 짙을수록 화재위험이 높은 구역. 검은 사각형이 출동 관서, "
                "선이 순찰 동선입니다.", ""]
    for t in plan["teams"]:
        out.append(f"○ {t['출동관서']} (관서 출발·복귀 기준 총 {t['총_km']:.1f} km)")
        out.append("")
        out.append("| 순번 | 격자 | 읍면동 | 이동거리 | 누적시간 |")
        out.append("|---|---|---|---|---|")
        for row in t["동선"].itertuples():
            emd = getattr(row, "emd", "") or getattr(row, "sgg", "")
            out.append(f"| {row.순번} | {row.grid_id} | {emd} | "
                       f"{row.이동거리_m:,.0f} m | {row.누적시간_분:.0f}분 |")
        out.append("")

    out += ["**마. 중점 확인 사항**", ""]
    for c in mode.checks:
        out.append(f"   □ {c}")

    if legal:
        out += ["", "**바. 세부 근거**", ""]
        for l in legal:
            out.append(f"○ {l['ref']}")
            out.append(f"   > {l['excerpt']}")
            out.append("")

    out += ["", "**사. 용어 및 유의사항**", "", GRID_DEFINITION, "",
            "- 본 계획은 공개 데이터 기반 화재위험 예측 결과이며, 법정 점검주기 및 "
            "관할 판단을 대체하지 않습니다.",
            "- 순찰 중 발견한 위험요인은 화재안전조사 대상으로 별도 보고합니다.",
            f"- 이동거리는 "
            f"{'실제 도로 주행거리' if plan['distance_source'] == 'osrm' else '직선거리 환산값'}"
            f"이며 교통 상황에 따라 달라질 수 있습니다."]
    forms = getattr(mode, "official_forms", ()) or ()
    if forms:
        out += ["", "**아. 관련 법정 서식**", "",
                "   순찰 결과 조치가 필요한 경우 아래 법정 서식을 사용합니다."
                " (「화재의 예방 및 안전관리에 관한 법률 시행규칙」 별지)", ""]
        for f in forms:
            out.append(f"   - {f}")
    out += _doc_footer(meta, ["순찰 대상 격자 목록", "관서별 순찰 동선도"])
    return "\n".join(out)


def render_monthly(plan: dict, ctx: PlanContext, meta: DocMeta | None = None) -> str:
    meta = meta or DocMeta()
    legal = plan.get("legal", [])
    basis = [_law_citation(l["ref"]) for l in legal[:2]] or \
            ["「화재의 예방 및 안전관리에 관한 법률」 제7조(화재안전조사)"]
    basis.append(f"{ctx.city_label} 월별 화재위험 분석 결과")

    out = _doc_header(meta, f"{plan['year']}년 {plan['month']}월 예방순찰 계획(안)", basis)
    out += [
        "2. 위 호와 관련하여 아래와 같이 월간 예방순찰 계획을 수립하고자 합니다.", "",
        "**가. 계획 개요**", "",
        "| 구분 | 내용 |", "|---|---|",
        f"| 중점 순찰 | {plan['mode'].label} |",
        f"| 월 화재위험 | 연평균 대비 {plan['risk_multiplier']:.2f}배 "
        f"({plan['risk_grade']}) |",
        f"| 계획 순찰 횟수 | {plan['total_rounds']}회 |",
        "",
        "**나. 주차별 계획**", "",
        "| 주차 | 기간 | 중점 관서 | 순찰 횟수 |", "|---|---|---|---|",
    ]
    for r in plan["weeks"].to_dict("records"):
        out.append(f"| {r['주차']} | {r['기간']} | {r.get('중점 관서', '')} | "
                   f"{r.get('순찰 횟수', 1)}회 |")

    if not plan["station_alloc"].empty:
        out += ["", "**다. 관서별 순찰 구역**", "",
                "| 출동관서 | 순찰 격자 |", "|---|---|"]
        for r in plan["station_alloc"].to_dict("records"):
            out.append(f"| {r['출동관서']} | {r['격자수']}개 |")

    out += ["", "**라. 중점 확인 사항**", ""]
    for c in plan["mode"].checks:
        out.append(f"   □ {c}")

    if legal:
        out += ["", "**마. 세부 근거**", ""]
        for l in legal:
            out.append(f"○ {l['ref']}")
            out.append(f"   > {l['excerpt']}")
            out.append("")

    out += ["", "**바. 용어**", "", GRID_DEFINITION,
            "- **위험계수**: 최근 8년 화재 발생과 기상(습도·건조일수)을 반영한 값으로, "
            "1.0이 연평균 수준입니다."]
    out += _doc_footer(meta, ["주차별 순찰 배정표", "관서별 순찰 구역도"])
    return "\n".join(out)


def render_annual(plan: dict, ctx: PlanContext, meta: DocMeta | None = None) -> str:
    meta = meta or DocMeta()
    legal = plan.get("legal", [])
    basis = ["「화재의 예방 및 안전관리에 관한 법률」 제7조(화재안전조사)",
             "「화재의 예방 및 안전관리에 관한 법률」 제18조(화재예방강화지구의 지정 등)",
             f"{ctx.city_label} 화재위험 예측 및 월별 발생 특성 분석 결과"]

    out = _doc_header(meta, f"{plan['year']}년 화재예방 순찰·점검 연간계획(안)", basis)
    out += [
        "2. 위 호와 관련하여 아래와 같이 연간계획을 수립하고자 합니다.", "",
        "**가. 분기별 운영 방향**", "",
        "| 분기 | 기간 | 평균 위험계수 | 중점 순찰 |", "|---|---|---|---|",
    ]
    for r in plan["quarters"].to_dict("records"):
        out.append(f"| {r['분기']} | {r['기간']} | {r['평균 위험계수']} | {r['중점 순찰']} |")

    out += ["", "**나. 월별 계획**", "",
            "| 월 | 중점 순찰 | 권장 시간대 | 위험계수 | 비고 |", "|---|---|---|---|---|"]
    for r in plan["calendar"].to_dict("records"):
        out.append(f"| {r['월']} | {r['중점 순찰']} | {r['권장 시간대']} | "
                   f"{r['위험계수']} | {r['비고']} |")

    out += ["", "**다. 법정 이행 사항**", "",
            "| 사항 | 주기 | 근거 |", "|---|---|---|"]
    for name, cycle, ref in plan["statutory"]:
        out.append(f"| {name} | {cycle} | {ref} |")

    if legal:
        out += ["", "**라. 세부 근거**", ""]
        for l in legal:
            out.append(f"○ {l['ref']}")
            out.append(f"   > {l['excerpt']}")
            out.append("")

    out += ["", "**마. 용어 및 유의사항**", "", GRID_DEFINITION, "",
            "- **위험계수**: 최근 8년 화재 발생과 기상(습도·건조일수)을 반영한 값으로, "
            "1.0이 연평균 수준입니다.",
            "- 본 계획은 법정 점검주기를 대체하지 않으며, 그 위에 우선순위를 더하는 것입니다."]
    out += _doc_footer(meta, ["월별 순찰 계획표", "관서별 관할 구역 현황"])
    return "\n".join(out)


def _law_citation(ref: str) -> str:
    """'화재의 예방 및 안전관리에 관한 법률 제7조' -> '「화재의 … 법률」 제7조'.

    공문은 법령명을 낫표로 감싸고 조문은 밖에 둔다.
    낫표를 호출부와 나눠 붙이면 짝이 어긋나므로 여기서 한 번에 만든다.
    """
    parts = str(ref).rsplit(" ", 1)
    if len(parts) == 2 and parts[1].startswith("제"):
        return f"「{parts[0]}」 {parts[1]}"
    return f"「{ref}」"


_law_short = _law_citation                    # 이전 이름 호환


RENDERERS = {"daily": render_daily, "monthly": render_monthly, "annual": render_annual}


def render(plan: dict, ctx: PlanContext, meta: DocMeta | None = None) -> str:
    return RENDERERS[plan["kind"]](plan, ctx, meta)


POLISH_RULE = (
    "아래는 소방서 예방과에서 결재를 올릴 순찰계획서입니다. "
    "문장 표현만 공문 문체로 다듬어 **완성된 문서 그 자체**를 출력하십시오.\n\n"
    "지켜야 할 것:\n"
    "1. 표의 숫자, 격자 번호, 관서명, 법령 조문, 거리·시간 값을 절대 바꾸지 마십시오.\n"
    "2. 없는 내용을 새로 만들지 마십시오.\n"
    "3. 마크다운 형식과 절 번호 체계를 그대로 유지하십시오.\n\n"
    "**절대 넣지 말 것 (매우 중요):**\n"
    "- 작업 내용에 대한 설명, 수정 사항 요약, 검토 의견, 확인 요청\n"
    "- '다듬었습니다', '확인이 필요합니다', '검토를 권합니다' 같은 문장\n"
    "- 인사말, 머리말, 맺음말, 코드블록 표시(```)\n"
    "- 원문에 없던 주석·각주·별표 설명\n\n"
    "출력은 문서 본문만 포함해야 하며, 첫 줄은 반드시 '# ' 로 시작하는 제목이어야 합니다."
)

#: LLM 이 문서 끝에 덧붙이기 쉬운 메타 발언. 결재 문서에 나가면 안 된다.
_META_HEADINGS = (
    "확인이 필요한", "검토가 필요한", "수정 사항", "변경 사항", "다듬은 내용",
    "작업 내용", "참고 사항(작성자", "AI 참고", "보완 의견", "검토 의견",
)


def strip_meta(text: str) -> str:
    """모델이 덧붙인 자기 작업 설명을 걷어낸다.

    프롬프트로 금지해도 모델은 종종 문서 끝에 '확인이 필요한 사항 3건' 같은
    코멘트를 붙인다. 그건 계획서가 아니라 작업 보고다. 결재 문서에 섞이면
    문서 전체의 신뢰가 떨어지므로 기계적으로도 한 번 더 막는다.
    """
    lines = str(text).splitlines()

    # 코드블록 울타리 제거
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    # 첫 제목(# ) 앞의 잡담 제거
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            lines = lines[i:]
            break

    # 메타 제목이 나오면 그 지점부터 끝까지 버린다
    cut = len(lines)
    for i, ln in enumerate(lines):
        bare = ln.lstrip("#* ").strip()
        if ln.lstrip().startswith(("#", "**")) and any(k in bare for k in _META_HEADINGS):
            cut = i
            break
    lines = lines[:cut]

    # 꼬리에 남은 메타 문장 정리
    while lines and (not lines[-1].strip() or
                     any(k in lines[-1] for k in _META_HEADINGS)):
        lines.pop()
    return "\n".join(lines).strip()


# ---------------------------------------------------------------- 사실성 검사

#: 문서에서 절대 바뀌면 안 되는 것들. 문체는 바뀌어도 이 값들은 그대로여야 한다.
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_LEGAL_RE = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?|제\s*\d+\s*항|제\s*\d+\s*호|별표\s*\d+")
_LAWNAME_RE = re.compile(r"「([^」]{2,60})」")


def _numbers(text: str) -> Counter:
    """문서에 나온 수를 센다. 자릿점은 표기 차이일 뿐이므로 지운다."""
    return Counter(_NUM_RE.findall(str(text).replace(",", "")))


def _legal_tokens(text: str) -> Counter:
    """조문·항·호·별표 번호와 법령명. 공백 표기 차이는 무시한다."""
    t = str(text)
    items = [re.sub(r"\s+", "", x) for x in _LEGAL_RE.findall(t)]
    items += [re.sub(r"\s+", "", x) for x in _LAWNAME_RE.findall(t)]
    return Counter(items)


def check_fidelity(original: str, candidate: str) -> dict:
    """다듬은 문서가 원문의 사실을 그대로 담고 있는지 기계적으로 확인한다.

    LLM 은 문체를 고치라고 하면 숫자도 '보기 좋게' 바꾼다. 68.5% 가 70% 가 되고
    제7조가 제17조가 되는 일이 실제로 일어난다. 결재 문서에서는 그 한 글자가
    문서 전체를 무효로 만든다. 그래서 프롬프트로 금지하는 데서 그치지 않고,
    나온 결과를 원문과 대조해 **새로 생긴 수·조문이 하나라도 있으면 버린다.**

    반환: {ok, added_numbers, dropped_numbers, added_legal, dropped_legal}
    """
    on, cn = _numbers(original), _numbers(candidate)
    ol, cl = _legal_tokens(original), _legal_tokens(candidate)
    added_n = sorted((cn - on).elements())
    dropped_n = sorted((on - cn).elements())
    added_l = sorted((cl - ol).elements())
    dropped_l = sorted((ol - cl).elements())
    # 새로 생긴 값은 무조건 차단. 사라진 수는 표 한 줄이 통째로 빠진 신호일 수
    # 있으므로, 몇 개 안 되는 표기 차이는 넘기고 덩어리로 빠지면 차단한다.
    total_n = max(sum(on.values()), 1)
    lost_badly = len(dropped_n) > 2 and len(dropped_n) / total_n > 0.05
    return {
        "ok": not (added_n or added_l or dropped_l or lost_badly),
        "dropped_too_many": bool(lost_badly),
        "added_numbers": added_n,
        "dropped_numbers": dropped_n,
        "added_legal": added_l,
        "dropped_legal": dropped_l,
    }


def polish(cfg, markdown: str, *, use_llm: bool = True) -> dict:
    """문체만 다듬는다. 숫자와 근거는 규칙이 만든 것을 그대로 둔다.

    다듬은 결과가 원문보다 눈에 띄게 짧아지면 채택하지 않는다 —
    모델이 표를 통째로 날렸다는 뜻이고, 그런 문서는 쓸 수 없다.
    """
    if not use_llm or not L.is_available(cfg):
        return {"text": markdown, "polished": False}
    text, backend, _tried = L.generate(cfg, f"{POLISH_RULE}\n\n---\n\n{markdown}")
    if not text:
        return {"text": markdown, "polished": False, "reason": "생성 실패"}

    cleaned = strip_meta(text)
    if len(cleaned) < len(markdown) * 0.6:
        log.warning("다듬은 문서가 원문의 %.0f%% 로 줄어 원문을 유지한다",
                    len(cleaned) / max(len(markdown), 1) * 100)
        return {"text": markdown, "polished": False, "reason": "내용 손실"}

    fid = check_fidelity(markdown, cleaned)
    if not fid["ok"]:
        log.warning("다듬은 문서가 원문에 없는 값을 담아 원문을 유지한다 "
                    "(새로 생긴 수 %s · 새로 생긴 조문 %s · 사라진 조문 %s)",
                    fid["added_numbers"][:5], fid["added_legal"][:5],
                    fid["dropped_legal"][:5])
        return {"text": markdown, "polished": False, "reason": "사실 불일치",
                "fidelity": fid}
    if fid["dropped_numbers"]:
        log.info("다듬는 과정에서 빠진 수: %s", fid["dropped_numbers"][:8])
    return {"text": cleaned, "polished": True, "backend": backend, "fidelity": fid}
