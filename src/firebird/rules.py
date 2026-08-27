"""업종별 점검 체크리스트 규칙.

위험 '판단'은 모델이 하고, 점검 '항목'은 규칙이 만든다.
LLM 은 이 둘을 문장으로 옮기기만 한다 — 환각이 점검 항목을 만들어내면
그건 행정 문서의 오류가 되기 때문이다.
"""
from __future__ import annotations

import pandas as pd

# 업종 -> (근거, 점검 항목들)
BUSINESS_RULES: dict[str, dict] = {
    "일반음식점": {
        "근거": "주방 화기·튀김기 과열, 덕트 기름때 축적",
        "항목": ["주방 자동소화장치 작동 상태", "덕트·후드 기름때 제거 주기 확인",
                "가스 누설 차단장치 및 배관 상태", "비상구 적치물 여부", "소화기 위치·압력"],
    },
    "노래연습장": {
        "근거": "밀실 구조·다량의 내장재, 피난 지연 위험",
        "항목": ["복도·계단 피난 유도등 점등", "각 실 비상구 개방 상태", "내부 마감재 방염성능 확인",
                "영업 중 비상방송 청취 가능 여부", "간이스프링클러 정상 여부"],
    },
    "유흥주점": {
        "근거": "심야 영업·음주 상태 이용객, 피난 능력 저하",
        "항목": ["비상구 잠금·폐쇄 여부(중점)", "피난통로 폭 확보", "방염 커튼·의자 성능 확인",
                "화재감지기 임의 차단 여부", "심야 시간대 안전요원 배치"],
    },
    "PC방": {
        "근거": "다수 전기기기 상시 가동, 문어발 배선·과부하",
        "항목": ["콘센트 문어발 접속·과부하 여부", "본체 후면 먼지 축적", "칸막이 방염 성능",
                "비상구 유도등 시야 확보", "누전차단기 동작시험"],
    },
    "숙박업": {
        "근거": "취침 중 화재 인지 지연, 객실 단위 피난",
        "항목": ["객실 내 피난기구·완강기 상태", "객실 감지기 작동", "복도 유도등·비상조명",
                "고시원 간이스프링클러 헤드 상태", "피난안내도 게시"],
    },
    "학원": {
        "근거": "미성년자 다수 이용, 피난 유도 필요",
        "항목": ["피난계단 적치물", "비상구 표시등", "소화기 비치 수량", "대피훈련 실시 기록"],
    },
    "산후조리원": {
        "근거": "자력 피난 불가 대상(신생아) 수용",
        "항목": ["신생아실 대피 계획·인력", "간이스프링클러 정상 여부", "감지기 오동작 이력",
                "야간 근무자 대피 숙지"],
    },
    "실내체육시설": {
        "근거": "대공간·다중 이용, 전기설비 밀집",
        "항목": ["실내 마감재 방염", "비상구 다중 확보", "전기 분전반 상태"],
    },
    "영화관": {
        "근거": "암전 상태 다중 수용, 피난 유도 의존도 높음",
        "항목": ["상영 중 유도등 시인성", "피난통로 폭", "비상방송 설비"],
    },
    "기타": {
        "근거": "일반 다중이용업소 공통 위험",
        "항목": ["소화기 비치·압력", "비상구 개방", "유도등 점등", "감지기 작동"],
    },
}

# 특정소방대상물 등급 -> 추가 확인 항목
GRADE_RULES: dict[str, list[str]] = {
    "특급": ["소방안전관리자(특급) 선임·상주 확인", "자체점검 결과 및 조치 이력", "제연설비 작동시험"],
    "1급": ["소방안전관리자(1급) 선임 확인", "스프링클러 밸브 개방 상태", "옥내소화전 방수압"],
    "2급": ["소방안전관리자 선임 확인", "자동화재탐지설비 수신기 이상표시"],
    "3급": ["소화기·유도등 등 기본 소방시설 상태"],
}

# 격자 상황 -> 상황별 항목
CONTEXT_RULES: list[tuple[str, str]] = [
    ("no_hydrant", "관할 소화전 부재 — 인접 수리(水利) 위치 사전 확인 및 중계송수 계획 수립"),
    ("far_hydrant", "최근접 소화전이 멀다 — 소방용수 확보 경로 사전 답사"),
    ("neighbor_fires", "인접 격자에서 최근 화재 발생 — 동일 위험요인(노후 배선·화기 취급) 확산 여부 점검"),
    ("repeat_fires", "최근 반복 화재 발생 구역 — 이전 조사 지적사항 이행 여부 재확인"),
]


def checklist_for_grid(row: pd.Series, cfg, top_biz: list[str] | None = None) -> dict:
    """한 격자의 점검 체크리스트를 조립한다."""
    items: list[dict] = []

    biz = top_biz if top_biz is not None else dominant_business_types(row)
    for b in biz:
        rule = BUSINESS_RULES.get(b, BUSINESS_RULES["기타"])
        items.append({"구분": f"업종:{b}", "근거": rule["근거"], "항목": list(rule["항목"])})

    for grade, extra in GRADE_RULES.items():
        col = f"target_n_{grade}"
        if float(row.get(col, 0) or 0) > 0:
            items.append({"구분": f"대상물:{grade}",
                          "근거": f"{grade} 특정소방대상물 {int(row[col])}개소 소재",
                          "항목": list(extra)})

    ctx = []
    if float(row.get("n_hydrant", 0) or 0) == 0:
        ctx.append(CONTEXT_RULES[0][1])
    elif float(row.get("dist_hydrant_m", 0) or 0) > cfg["hydrant"]["max_dist_m"]:
        ctx.append(CONTEXT_RULES[1][1])
    if float(row.get("neigh_fires_lag1", 0) or 0) > 0:
        ctx.append(CONTEXT_RULES[2][1])
    if float(row.get("fires_cum", 0) or 0) >= 2:
        ctx.append(CONTEXT_RULES[3][1])
    if ctx:
        items.append({"구분": "현장여건", "근거": "격자 대응환경·이력", "항목": ctx})

    return {"grid_id": row.get("grid_id"), "sections": items,
            "n_items": sum(len(s["항목"]) for s in items)}


def dominant_business_types(row: pd.Series, top_n: int = 3) -> list[str]:
    """이 격자에서 가장 많은 업종 상위 N개."""
    counts = {c[len("biz_n_"):]: float(row[c] or 0)
              for c in row.index if str(c).startswith("biz_n_")}
    ranked = [k for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v > 0]
    return ranked[:top_n] or ["기타"]
