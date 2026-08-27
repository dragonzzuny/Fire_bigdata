"""순찰·점검 계획서 생성 — 일별 / 월별 / 연간.

현장에 필요한 것은 위험 점수표가 아니라 **결재를 올릴 수 있는 문서**다.
어느 관서가, 언제, 어디를, 어떤 순서로, 무엇을 보고, 무슨 근거로 하는가가
한 장에 있어야 한다.

이 모듈은 문서를 규칙으로 먼저 완성한다. LLM 은 그 뒤에 문체만 다듬는다.
계획의 숫자·동선·법령 근거가 생성 모델에서 나오면 검증이 불가능해진다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import llm as L
from .patrol_modes import MODES, PatrolMode, recommended_hours

log = logging.getLogger(__name__)

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
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    distance_source: str = ""
    month_plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    law_index: object | None = None
    fires: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------- 법령 근거

def find_legal_basis(ctx: PlanContext, extra_terms: list[str] | None = None,
                     top_k: int = 4) -> list[dict]:
    """이 계획과 관련된 법령 조문을 찾는다.

    조문 번호 없는 계획서는 결재가 안 난다. 순찰 유형의 목적과 점검 항목을
    질의어로 삼아 근거를 자동으로 붙인다.
    """
    if ctx.law_index is None:
        return []
    query = " ".join([ctx.mode.label, ctx.mode.purpose,
                      *ctx.mode.checks, *(extra_terms or [])])
    hits = ctx.law_index.search(query, top_k=top_k * 2)
    out = []
    for doc, score in hits:
        if getattr(doc, "source", "") != "법령":
            continue
        out.append({"ref": doc.ref, "title": doc.title,
                    "excerpt": doc.text[:220].replace("\n", " "), "score": round(score, 1)})
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

def _legal_block(legal: list[dict]) -> list[str]:
    if not legal:
        return []
    out = ["", "## 법령 근거", ""]
    for i, l in enumerate(legal, 1):
        out.append(f"{i}. **{l['ref']}**")
        out.append(f"   > {l['excerpt']}")
    return out


def render_daily(plan: dict, ctx: PlanContext) -> str:
    d, mode = plan["date"], plan["mode"]
    h = plan["hours"]
    out = [
        f"# 예방순찰 계획서 ({d.year}. {d.month}. {d.day}. {plan['weekday']})",
        "",
        f"- **순찰 종류**: {mode.label}",
        f"- **순찰 시간**: {h[0]:02d}:00 – {h[1]:02d}:00",
        f"- **대상 지역**: {ctx.city_label}",
        f"- **투입 관서**: {len(plan['teams'])}개 관서",
        f"- **거리 기준**: "
        f"{'실제 도로 주행거리' if plan['distance_source'] == 'osrm' else '직선거리 환산'}",
        "",
        "## 1. 순찰 목적", "", f"{mode.purpose}", "",
        "## 2. 관서별 순찰 구역", "",
        "| 출동관서 | 순찰 격자 | 이동거리 | 소요시간 |",
        "|---|---|---|---|",
    ]
    for t in plan["teams"]:
        out.append(f"| {t['출동관서']} | {t['격자수']}개 | "
                   f"{t['총_km']:.1f} km | {t['총_분']:.0f}분 |")

    out += ["", "## 3. 순찰 동선", ""]
    for t in plan["teams"]:
        out.append(f"### {t['출동관서']}")
        out.append("")
        out.append("| 순번 | 격자 | 읍면동 | 이동거리 | 누적시간 |")
        out.append("|---|---|---|---|---|")
        r = t["동선"]
        for row in r.itertuples():
            emd = getattr(row, "emd", "") or getattr(row, "sgg", "")
            out.append(f"| {row.순번} | {row.grid_id} | {emd} | "
                       f"{row.이동거리_m:,.0f} m | {row.누적시간_분:.0f}분 |")
        out.append("")
        out.append(f"※ 순찰 종료 후 {t['출동관서']}로 복귀 (총 {t['총_km']:.1f} km)")
        out.append("")

    out += ["## 4. 중점 확인 사항", ""]
    for c in mode.checks:
        out.append(f"- [ ] {c}")

    out += _legal_block(plan.get("legal", []))
    out += ["", "## 5. 유의사항", "",
            "- 본 계획은 공개 데이터 기반 화재위험 예측 결과이며, "
            "법정 점검주기 및 관할 판단을 대체하지 않습니다.",
            "- 격자는 500m 단위 집계로, 개별 건물을 특정하지 않습니다.",
            "- 순찰 중 발견한 위험요인은 화재안전조사 대상으로 별도 보고합니다."]
    return "\n".join(out)


def render_monthly(plan: dict, ctx: PlanContext) -> str:
    out = [
        f"# {plan['year']}년 {plan['month']}월 예방순찰 계획",
        "",
        f"- **중점 순찰**: {plan['mode'].label}",
        f"- **월 화재위험**: 연평균 대비 **{plan['risk_multiplier']:.2f}배** "
        f"({plan['risk_grade']})",
        f"- **계획 순찰 횟수**: {plan['total_rounds']}회",
        "",
        "## 1. 주차별 계획", "",
        "| 주차 | 기간 | 중점 관서 | 순찰 횟수 |", "|---|---|---|---|",
    ]
    # 컬럼명에 공백이 있으면 itertuples 의 속성 접근이 안 된다. dict 로 받는다.
    for r in plan["weeks"].to_dict("records"):
        out.append(f"| {r['주차']} | {r['기간']} | {r.get('중점 관서', '')} | "
                   f"{r.get('순찰 횟수', 1)}회 |")

    if not plan["station_alloc"].empty:
        out += ["", "## 2. 관서별 순찰 구역", "",
                "| 출동관서 | 순찰 격자 |", "|---|---|"]
        for r in plan["station_alloc"].to_dict("records"):
            out.append(f"| {r['출동관서']} | {r['격자수']}개 |")

    out += ["", "## 3. 중점 확인 사항", ""]
    for c in plan["mode"].checks:
        out.append(f"- [ ] {c}")
    out += _legal_block(plan.get("legal", []))
    return "\n".join(out)


def render_annual(plan: dict, ctx: PlanContext) -> str:
    out = [
        f"# {plan['year']}년 화재예방 순찰·점검 연간계획",
        "",
        f"- **대상**: {ctx.city_label}",
        "- **수립 근거**: 화재위험 예측 결과 및 월별 화재 발생 특성",
        "",
        "## 1. 분기별 운영 방향", "",
        "| 분기 | 기간 | 평균 위험계수 | 중점 순찰 |", "|---|---|---|---|",
    ]
    for r in plan["quarters"].to_dict("records"):
        out.append(f"| {r['분기']} | {r['기간']} | {r['평균 위험계수']} | "
                   f"{r['중점 순찰']} |")

    out += ["", "## 2. 월별 계획", "",
            "| 월 | 중점 순찰 | 권장 시간대 | 위험계수 | 비고 |",
            "|---|---|---|---|---|"]
    for r in plan["calendar"].to_dict("records"):
        out.append(f"| {r['월']} | {r['중점 순찰']} | "
                   f"{r['권장 시간대']} | {r['위험계수']} | {r['비고']} |")

    out += ["", "## 3. 법정 이행 사항", "",
            "| 사항 | 주기 | 근거 |", "|---|---|---|"]
    for name, cycle, ref in plan["statutory"]:
        out.append(f"| {name} | {cycle} | {ref} |")

    out += _legal_block(plan.get("legal", []))
    out += ["", "## 4. 유의사항", "",
            "- 위험계수는 최근 8년 화재 발생과 기상(습도·건조일수)을 반영한 값으로, "
            "1.0이 연평균 수준입니다.",
            "- 본 계획은 법정 점검주기를 대체하지 않으며, 그 위에 우선순위를 더하는 것입니다."]
    return "\n".join(out)


RENDERERS = {"daily": render_daily, "monthly": render_monthly, "annual": render_annual}


def render(plan: dict, ctx: PlanContext) -> str:
    return RENDERERS[plan["kind"]](plan, ctx)


POLISH_RULE = (
    "아래는 소방서 예방과에서 결재를 올릴 순찰계획서 초안입니다. "
    "표의 숫자, 격자 번호, 관서명, 법령 조문은 **절대 바꾸지 마십시오**. "
    "문장 표현만 공문 문체로 다듬고, 목적과 유의사항을 자연스럽게 보완하십시오. "
    "없는 내용을 새로 만들지 마십시오. 마크다운 형식을 유지하십시오."
)


def polish(cfg, markdown: str, *, use_llm: bool = True) -> dict:
    """문체만 다듬는다. 숫자와 근거는 규칙이 만든 것을 그대로 둔다."""
    if not use_llm or not L.is_available(cfg):
        return {"text": markdown, "polished": False}
    text, backend, _tried = L.generate(cfg, f"{POLISH_RULE}\n\n---\n\n{markdown}")
    if not text:
        return {"text": markdown, "polished": False}
    return {"text": text, "polished": True, "backend": backend}
