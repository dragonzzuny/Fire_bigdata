"""순찰 목적별 유형.

소방의 예방순찰은 하나가 아니다. 목적이 다르면 **가야 할 곳도, 시간도, 볼 것도**
다르다. 위험 상위 격자를 한 줄로 세워 주는 것만으로는 현장 업무에 안 맞는다.

근거 (화재의 예방 및 안전관리에 관한 법률, 소방공무원 당직 및 비상업무규칙):
  · 제18조 화재예방강화지구 — 시장지역, 공장·창고 밀집지역 등을 지정하고
    소방관서장이 그 지구의 소방대상물에 화재안전조사를 실시한다.
  · 특별경계근무 — 비상상황에 이르지 않은 경우 목적과 기간을 명시해
    "출동대비 인원 보강을 최소화하고 순찰 등 예방활동 위주로" 편성한다.

여기서 정의하는 유형은 그 업무 구분을 데이터 쪽에 옮긴 것이다.
각 유형은 (대상 선정 기준, 권장 시간대, 중점 확인 항목)이 다르다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatrolMode:
    key: str
    label: str
    purpose: str
    #: 대상 격자를 고를 때 가중치를 줄 컬럼과 배수
    weights: dict[str, float] = field(default_factory=dict)
    #: 권장 시간대 (시작시, 끝시). None 이면 데이터에서 산출한 피크를 쓴다.
    hours: tuple[int, int] | None = None
    #: 현장에서 볼 것
    checks: tuple[str, ...] = ()
    #: 기본 순찰 격자 수
    default_k: int = 15
    #: 이 순찰의 법적 근거. 검색으로 찾은 조문보다 이쪽을 먼저 쓴다.
    #: 어휘 검색은 '소방시설·피난' 같은 낱말만 보고 화재예방안전진단 절차 규정을
    #: 일반 순찰의 근거로 끌어오기도 한다. 근거가 어긋나면 결재 때 신뢰를 잃는다.
    legal_refs: tuple[str, ...] = ()


MODES: dict[str, PatrolMode] = {
    "general": PatrolMode(
        key="general", label="일반 예방순찰",
        purpose="화재위험이 높은 구역을 정기적으로 돌며 위험요인을 확인한다.",
        weights={"pred": 1.0},
        hours=None,
        checks=("소화기·유도등 등 기본 소방시설 상태",
                "비상구 적치물 및 폐쇄 여부",
                "옥외 가연물 방치",
                "노후 전기설비·문어발 배선"),
        legal_refs=("화재의 예방 및 안전관리에 관한 법률 제7조",
                    "화재의 예방 및 안전관리에 관한 법률 제17조"),
        default_k=15),

    "night_business": PatrolMode(
        key="night_business", label="다중이용업소 야간순찰",
        purpose="심야 영업 중 비상구 폐쇄·피난 장애를 현장에서 확인한다. "
                "영업 중이어야 확인되는 항목이라 낮 점검으로는 대체되지 않는다.",
        weights={"biz_n_유흥주점": 3.0, "biz_n_노래연습장": 2.5,
                 "biz_n_PC방": 1.5, "biz_n_일반음식점": 1.0, "pred": 0.5},
        hours=(22, 2),
        checks=("영업 중 비상구 잠금·폐쇄 여부 (중점)",
                "피난통로 폭 확보 및 적치물",
                "유도등 점등 및 시인성",
                "화재감지기 임의 차단 여부",
                "내부 마감재 방염성능"),
        legal_refs=("다중이용업소의 안전관리에 관한 특별법 제13조",
                    "화재의 예방 및 안전관리에 관한 법률 제17조"),
        default_k=12),

    "market": PatrolMode(
        key="market", label="화재예방강화지구 순찰",
        purpose="시장지역·공장 밀집지역 등 법 제18조 지정 지구를 집중 관리한다.",
        weights={"usage_n_판매영업": 3.0, "usage_n_공장": 2.0,
                 "usage_n_창고": 1.5, "usage_n_복합건축물": 1.5, "pred": 1.0},
        hours=None,
        checks=("점포 간 방화구획·연소확대 위험",
                "소방차 진입로 확보 및 불법 주정차",
                "전기 분전반·노후 배선",
                "소화기·옥외소화전 접근성",
                "심야 무인 상태의 화기 관리"),
        legal_refs=("화재의 예방 및 안전관리에 관한 법률 제18조",
                    "화재의 예방 및 안전관리에 관한 법률 시행령 제20조",
                    "화재의 예방 및 안전관리에 관한 법률 시행규칙 제8조"),
        default_k=12),

    "vulnerable": PatrolMode(
        key="vulnerable", label="피난약자시설 순찰",
        purpose="자력 피난이 어려운 수용자가 있는 시설을 확인한다. "
                "화재 건수는 적어도 발생 시 인명피해가 크다.",
        weights={"usage_n_노유자": 3.0, "usage_n_의료": 2.5,
                 "usage_n_숙박": 1.5, "pred": 0.5},
        hours=(20, 23),
        checks=("야간 근무 인력 및 대피 계획 숙지",
                "피난기구·완강기 상태",
                "간이스프링클러 헤드 장애물",
                "객실·병실 감지기 작동",
                "피난안내도 게시"),
        legal_refs=("화재의 예방 및 안전관리에 관한 법률 제36조",
                    "화재의 예방 및 안전관리에 관한 법률 제37조"),
        default_k=10),

    "water_supply": PatrolMode(
        key="water_supply", label="소방용수시설 점검순찰",
        purpose="소화전이 없거나 먼 구역의 대체 수리(水利)를 사전 확인한다. "
                "화재가 나고 나서 찾으면 늦다.",
        weights={"dist_hydrant_m": 1.0, "pred": 1.0},
        hours=(9, 17),
        checks=("소화전 개폐 상태 및 주변 적치물",
                "소화전 표지 시인성, 불법 주정차",
                "대체 수리(저수조·하천) 접근로",
                "겨울철 동결 여부"),
        legal_refs=("소방기본법 제10조",
                    "소방기본법 제28조"),
        default_k=15),

    "dry_season": PatrolMode(
        key="dry_season", label="건조기 특별경계순찰",
        purpose="건조·강풍 시기에 임야 인접·옥외 가연물 구역을 집중한다. "
                "특별경계근무는 목적과 기간을 정해 예방활동 위주로 편성한다.",
        weights={"usage_n_동식물": 2.0, "usage_n_창고": 1.5, "pred": 1.0},
        hours=(13, 18),
        checks=("옥외 소각·화기 취급 행위",
                "임야 인접 가연물 적치",
                "건축 공사장 화기 관리",
                "쓰레기 집하장·폐기물 야적"),
        legal_refs=("화재의 예방 및 안전관리에 관한 법률 제17조",
                    "화재의 예방 및 안전관리에 관한 법률 시행령 제16조"),
        default_k=15),
}


def mode_score(panel_year: pd.DataFrame, mode: PatrolMode) -> pd.Series:
    """순찰 유형에 맞는 대상 선정 점수.

    위험 예측값만 쓰지 않는다. 야간 순찰이라면 유흥주점이 많은 격자가,
    피난약자 순찰이라면 노유자시설이 있는 격자가 앞으로 와야 한다.
    각 항목을 백분위로 바꿔 더하므로 단위가 달라도 섞을 수 있다.
    """
    score = pd.Series(0.0, index=panel_year.index)
    total_w = 0.0
    for col, w in mode.weights.items():
        if col not in panel_year.columns:
            continue
        v = pd.to_numeric(panel_year[col], errors="coerce").fillna(0.0)
        if col == "dist_hydrant_m":
            # 소화전이 멀수록 높은 점수. 무한대(소화전 없음)는 최댓값으로.
            v = v.replace([np.inf, -np.inf], np.nan)
            v = v.fillna(v.max() if v.notna().any() else 0.0)
        if v.nunique() <= 1:
            continue
        score = score + v.rank(pct=True) * w
        total_w += w
    if total_w > 0:
        return score / total_w * 100.0
    # 가중 컬럼이 하나도 없으면 예측값으로 물러난다.
    base = pd.to_numeric(panel_year.get("pred", pd.Series(0.0, index=panel_year.index)),
                         errors="coerce").fillna(0.0)
    return base.rank(pct=True) * 100.0


def select_targets(panel_year: pd.DataFrame, mode: PatrolMode, k: int | None = None,
                   *, sgg: list[str] | None = None) -> pd.DataFrame:
    """유형·지역에 맞는 순찰 대상 격자 상위 k개."""
    df = panel_year
    if sgg:
        df = df[df["sgg"].astype(str).isin([str(s) for s in sgg])]
    if df.empty:
        return df
    out = df.copy()
    out["순찰점수"] = mode_score(out, mode)
    n = int(k or mode.default_k)
    return out.nlargest(min(n, len(out)), "순찰점수").reset_index(drop=True)


def recommended_hours(mode: PatrolMode, fires: pd.DataFrame | None = None) -> tuple[int, int]:
    """권장 시간대. 유형이 시간을 정해 두면 그것을, 아니면 데이터의 피크를 쓴다."""
    if mode.hours is not None:
        return mode.hours
    if fires is None or fires.empty or "hour" not in fires.columns:
        return (13, 18)
    from .patrol import peak_windows
    peaks = peak_windows(fires, top_n=1, window=3)
    if peaks:
        return (int(peaks[0]["start_hour"]), int(peaks[0]["end_hour"]))
    return (13, 18)
