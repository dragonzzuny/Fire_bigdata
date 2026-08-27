"""법정 서식 채우기.

계획서를 우리 마음대로 만들면 결재에서 되돌아온다. 법에 서식이 정해져 있는
업무는 그 서식을 써야 한다. 여기서는 「화재의 예방 및 안전관리에 관한 법률
시행규칙」 [별지 제11호서식] **화재예방강화지구 관리대장**을 다룬다.

채울 수 있는 칸만 채운다. 없는 자료는 빈칸으로 두고 '왜 비었는지'를 함께
표시한다. 빈칸을 그럴듯한 값으로 메우는 순간 이 문서는 결재 문서가 아니라
추정치 모음이 되고, 그 사실을 읽는 사람은 알 수 없게 된다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FORM_LAW = "화재의 예방 및 안전관리에 관한 법률 시행규칙"
FORM_NO = "별지 제11호서식"
FORM_TITLE = "화재예방강화지구 관리대장"

#: 서식의 '소방시설' 칸 ↔ 우리가 가진 대상물 설비 열.
#: 서식은 설비 '개수'를 묻는다. 우리 자료는 그 설비를 갖춘 대상물 수이므로,
#: 그대로 쓰지 않고 무엇을 센 값인지 표기와 함께 넘긴다.
FACILITY_MAP = {
    "소화설비": ("fac_n_옥내소화전", "옥내소화전 설치 대상물"),
    "경보설비": ("fac_n_자동화재탐지", "자동화재탐지설비 설치 대상물"),
    "소화활동설비": ("fac_n_스프링클러", "스프링클러 설치 대상물"),
    "소방용수": ("n_hydrant", "격자 내 소방용수시설"),
}

#: 공개 자료로는 채울 수 없는 칸과 그 이유. 화면과 문서에 그대로 적힌다.
NOT_AVAILABLE = {
    "대표자": "공개 데이터에 개인정보가 포함되지 않음",
    "전화번호": "공개 데이터에 개인정보가 포함되지 않음",
    "지정일자": "화재예방강화지구 지정 이력은 소방본부 내부 자료",
    "건축연도": "건축물대장 연계 필요",
    "연면적": "건축물대장 연계 필요",
    "건축면적": "건축물대장 연계 필요",
    "유동인구": "생활인구 자료 미연계",
    "상주인구": "주민등록 자료 미연계",
    "소방조직": "의용소방대 편성 자료 미연계",
    "화재안전조사 결과 및 조치명령": "점검이력 데이터에 결합 키가 없어 미연계",
    "소방설비등의 설치지원": "설치지원 집행 자료 미연계",
    "소방훈련 및 소방교육": "훈련·교육 실시 기록 미연계",
}


@dataclass(frozen=True)
class Field:
    """서식 한 칸. value 가 비면 blank_reason 이 왜 비었는지 말한다."""
    label: str
    value: str = ""
    unit: str = ""
    blank_reason: str = ""
    source: str = ""            # 이 값이 어느 자료에서 나왔는지


def _station_distances(row: pd.Series, stations: pd.DataFrame) -> dict:
    """본서·관할 119안전센터까지의 거리(km).

    서식이 묻는 '소방관서거리'다. 관서 좌표와 격자 좌표가 모두 있으므로
    계산할 수 있다. 도로거리가 아니라 직선거리임을 표기와 함께 넘긴다.
    """
    out = {}
    need = {"name", "lon", "lat"}
    if (stations is None or stations.empty
            or not need.issubset(set(stations.columns))):
        return out
    lon0, lat0 = float(row.get("lon", np.nan)), float(row.get("lat", np.nan))
    if not (np.isfinite(lon0) and np.isfinite(lat0)):
        return out
    s = stations.dropna(subset=["lon", "lat"]).set_index("name")
    for key, col in (("본서", "station"), ("관할119안전센터", "center")):
        name = str(row.get(col, "") or "").strip()
        if not name or name not in s.index:
            continue
        lon1, lat1 = float(s.loc[name, "lon"]), float(s.loc[name, "lat"])
        dlat = np.radians(lat1 - lat0)
        dlon = np.radians(lon1 - lon0)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat0)) * np.cos(np.radians(lat1)) * np.sin(dlon / 2) ** 2)
        out[key] = (name, 6371.0 * 2 * float(np.arcsin(np.sqrt(min(a, 1.0)))))
    return out


def zone_ledger(row: pd.Series, *, city_label: str, year: int,
                stations: pd.DataFrame | None = None,
                drivers: list[dict] | None = None,
                grid_m: int = 500) -> dict:
    """격자 하나를 [별지 제11호서식] 관리대장 칸에 맞춰 채운다.

    row 는 그 해 격자 한 줄(패널). drivers 는 SHAP 상위 요인이며 '지구특징'에 쓴다.
    """
    def num(col, default=0):
        v = row.get(col, default)
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    gid = str(row.get("grid_id", ""))
    emd = str(row.get("emd", "") or "").strip()
    sgg = str(row.get("sgg", "") or "").strip()
    where = " ".join(x for x in (city_label, sgg, emd) if x)

    fields: dict[str, Field] = {}
    fields["명칭"] = Field("명칭", f"{emd or sgg} 화재위험 {gid} 구역",
                          source=f"{year}년 위험도 분석 결과")
    fields["위치"] = Field("위치", f"{where} (격자 {gid}, {grid_m}m×{grid_m}m)",
                          source="UTM-K(EPSG:5179) 격자")
    for k in ("대표자", "전화번호", "지정일자", "건축연도", "연면적", "건축면적",
              "유동인구", "상주인구", "소방조직"):
        fields[k] = Field(k, blank_reason=NOT_AVAILABLE[k])

    fields["건물동수"] = Field("건물동수", f"{num('target_total'):,}", "개소",
                             source="특정소방대상물 현황")
    fields["점포수"] = Field("점포수", f"{num('biz_total'):,}", "개소",
                           source="다중이용업소 현황")
    fields["지구면적"] = Field("지구면적", f"{grid_m * grid_m:,}", "㎡",
                             source="격자 정의")

    # 건물구조: 용도 구성 상위 3개
    usage = {c[len("usage_n_"):]: num(c) for c in row.index
             if str(c).startswith("usage_n_")}
    top_use = sorted((v, k) for k, v in usage.items() if v > 0)[::-1][:3]
    fields["건물구조"] = Field(
        "건물구조",
        ", ".join(f"{k} {v}개소" for v, k in top_use) if top_use else "",
        blank_reason="" if top_use else "격자 내 등록 대상물 없음",
        source="특정소방대상물 용도 구성")

    for label, (col, note) in FACILITY_MAP.items():
        n = num(col)
        fields[f"소방시설:{label}"] = Field(label, f"{n:,}", "개", source=note)
    fields["소방시설:피난구조설비"] = Field(
        "피난구조설비", blank_reason="설비별 세부 현황 자료 미연계")
    fields["소방시설:기타설비"] = Field(
        "기타설비", blank_reason="설비별 세부 현황 자료 미연계")

    dists = _station_distances(row, stations if stations is not None else pd.DataFrame())
    for key in ("본서", "관할119안전센터"):
        if key in dists:
            name, km = dists[key]
            fields[f"소방관서거리:{key}"] = Field(
                key, f"{km:.1f}", "km", source=f"{name} 좌표 기준 직선거리")
        else:
            fields[f"소방관서거리:{key}"] = Field(key, blank_reason="관서 좌표 미확인")

    # 지구특징 — 모델이 위험도를 올린 요인
    feats = []
    for d in (drivers or [])[:4]:
        nm = d.get("label") or d.get("feature") or ""
        val = d.get("value")
        if not nm:
            continue
        feats.append(f"{nm}" + (f" {val}" if val not in (None, "") else ""))
    if not feats:
        feats = [f"{year}년 화재 {num('fires')}건, 누적 {num('fires_cum')}건"]
    fields["지구특징"] = Field("지구특징", " · ".join(feats),
                             source="모델 기여요인(SHAP) 및 화재 이력")

    # 취약요소 — 소화전 사각, 화재 재발
    weak = []
    hyd = num("n_hydrant")
    dist_h = row.get("dist_hydrant_m", np.nan)
    if hyd <= 0:
        weak.append("격자 내 소방용수시설 없음"
                    + (f" (최근접 {float(dist_h):,.0f}m)"
                       if pd.notna(dist_h) and float(dist_h) < 1e6 else ""))
    if num("fires_lag1") > 0 and num("fires") > 0:
        weak.append(f"2년 연속 화재 발생({year - 1}년 {num('fires_lag1')}건, "
                    f"{year}년 {num('fires')}건)")
    if num("biz_n_유흥주점") + num("biz_n_노래연습장") > 0:
        weak.append(f"심야 영업 업소 {num('biz_n_유흥주점') + num('biz_n_노래연습장')}개소")
    fields["취약요소"] = Field("취약요소", " · ".join(weak),
                             blank_reason="" if weak else "해당 사항 없음",
                             source="소방용수시설 현황 및 화재발생현황")

    fields["도로여건"] = Field("도로여건", blank_reason="도로 폭·소방차 진입로 자료 미연계")
    fields["현대화사업"] = Field("현대화사업", blank_reason="사업 집행 자료 미연계")

    sections = {
        "화재안전조사 결과 및 조치명령": NOT_AVAILABLE["화재안전조사 결과 및 조치명령"],
        "소방설비등의 설치지원": NOT_AVAILABLE["소방설비등의 설치지원"],
        "소방훈련 및 소방교육": NOT_AVAILABLE["소방훈련 및 소방교육"],
    }
    filled = sum(1 for f in fields.values() if f.value)
    return {
        "law": FORM_LAW, "form_no": FORM_NO, "title": FORM_TITLE,
        "grid_id": gid, "fields": fields, "empty_sections": sections,
        "n_fields": len(fields), "n_filled": filled,
        "fill_rate": filled / max(len(fields), 1),
    }


# ---------------------------------------------------------------- 렌더링

def _cell(f: Field) -> str:
    import html as _h
    if f.value:
        unit = f" <span class='u'>{_h.escape(f.unit)}</span>" if f.unit else ""
        tip = f" title=\"{_h.escape(f.source)}\"" if f.source else ""
        return f"<td class='v'{tip}>{_h.escape(f.value)}{unit}</td>"
    reason = _h.escape(f.blank_reason or "")
    return (f"<td class='e' title=\"{reason}\">"
            f"<span class='why'>{reason}</span></td>")


LEDGER_CSS = """
<style>
.form { font-family:'Malgun Gothic','맑은 고딕',sans-serif; color:#111; }
.form .hdr { font-size:11px; letter-spacing:.02em; margin:0 0 6px; }
.form h2 { text-align:center; font-size:21px; letter-spacing:.14em;
           margin:2px 0 10px; font-weight:700; }
.form .side { text-align:right; font-size:11px; margin:0 0 4px; }
.form .sec { border:1px solid #111; border-bottom:0; padding:4px 8px;
             font-weight:700; font-size:12.5px; background:#f2f2f2; }
.form table { border-collapse:collapse; width:100%; table-layout:fixed;
              font-size:12px; margin-bottom:10px; }
.form th, .form td { border:1px solid #111; padding:5px 7px; height:26px;
                     vertical-align:middle; word-break:break-all; }
.form th { background:#fafafa; font-weight:500; text-align:center; width:110px; }
.form td.v { background:#fff7ed; }
.form td.e { background:#fbfbfb; }
.form td .u { color:#666; font-size:10.5px; margin-left:2px; }
.form td .why { color:#b8b8b8; font-size:10px; font-style:italic; }
.form .foot { text-align:right; font-size:10.5px; color:#333; margin-top:6px; }
.form .legend { font-size:11px; color:#555; margin:8px 0 0; }
.form .legend b { background:#fff7ed; padding:1px 6px; border:1px solid #ddd; }
</style>
"""


def render_ledger_html(led: dict, *, city_label: str = "", year: int = 0) -> str:
    """관리대장을 서식 배치 그대로 HTML 로 만든다.

    칸 순서·묶음은 법제처 PDF 서식과 같게 두었다. 채운 칸은 배경으로 구분하고,
    빈칸에는 왜 비었는지를 작은 글씨로 적는다.
    """
    F = led["fields"]

    def row2(a, b):
        return (f"<tr><th>{F[a].label}</th>{_cell(F[a])}"
                f"<th>{F[b].label}</th>{_cell(F[b])}</tr>")

    parts = [LEDGER_CSS, "<div class='form'>",
             f"<div class='hdr'>■ {led['law']} [{led['form_no']}]</div>",
             f"<h2>{led['title']}</h2>",
             "<div class='side'>(앞쪽)</div>",
             "<div class='sec'>1. 화재예방강화지구 지정현황</div>",
             "<table><colgroup><col width='110'><col><col width='110'><col></colgroup>"]
    parts.append(f"<tr><th>{F['명칭'].label}</th>{_cell(F['명칭'])}"
                 f"<th>{F['위치'].label}</th>{_cell(F['위치'])}</tr>")
    parts.append(row2("대표자", "전화번호"))
    parts.append(row2("지정일자", "건축연도"))
    parts.append(row2("연면적", "건축면적"))
    parts.append(row2("건물동수", "점포수"))
    parts.append(row2("유동인구", "상주인구"))
    parts.append(f"<tr><th>소방조직</th>{_cell(F['소방조직'])}<th></th><td></td></tr>")
    parts.append(f"<tr><th>건물구조</th>{_cell(F['건물구조'])}"
                 f"<th>지구면적</th>{_cell(F['지구면적'])}</tr>")
    parts.append("</table>")

    parts.append("<table><colgroup><col width='110'><col width='120'><col>"
                 "<col width='120'><col></colgroup>")
    fac = [("소방시설:소화설비", "소방시설:경보설비"),
           ("소방시설:피난구조설비", "소방시설:소방용수"),
           ("소방시설:소화활동설비", "소방시설:기타설비")]
    for i, (a, b) in enumerate(fac):
        head = "<th rowspan='3'>소방시설</th>" if i == 0 else ""
        parts.append(f"<tr>{head}<th>[&nbsp;&nbsp;]{F[a].label}</th>{_cell(F[a])}"
                     f"<th>[&nbsp;&nbsp;]{F[b].label}</th>{_cell(F[b])}</tr>")
    parts.append("</table>")

    parts.append("<table><colgroup><col width='110'><col width='120'><col>"
                 "<col width='140'><col></colgroup>")
    parts.append(f"<tr><th>소방관서거리</th><th>본서</th>"
                 f"{_cell(F['소방관서거리:본서'])}<th>관할119안전센터</th>"
                 f"{_cell(F['소방관서거리:관할119안전센터'])}</tr>")
    parts.append("</table>")

    parts.append("<table><colgroup><col width='110'><col></colgroup>")
    for k in ("지구특징", "취약요소", "도로여건", "현대화사업"):
        parts.append(f"<tr><th>{F[k].label}</th>{_cell(F[k])}</tr>")
    parts.append("</table>")

    for i, (name, reason) in enumerate(led["empty_sections"].items(), 2):
        parts.append(f"<div class='sec'>{i}. {name}</div>")
        parts.append("<table><colgroup><col width='130'><col>"
                     "<col width='130'></colgroup>")
        heads = {"화재안전조사 결과 및 조치명령": ("일자", "조치사항", "보완일자"),
                 "소방설비등의 설치지원": ("일자", "설치내용", "완료일자"),
                 "소방훈련 및 소방교육": ("훈련일자", "참석인원", "교육일자")}[name]
        parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr>")
        parts.append(f"<tr><td class='e' colspan='3'>"
                     f"<span class='why'>{reason}</span></td></tr>")
        parts.append("<tr>" + "<td class='e'></td>" * 3 + "</tr>")
        parts.append("</table>")

    parts.append("<div class='foot'>210mm×297mm[백상지(80g/㎡) 또는 중질지(80g/㎡)]</div>")
    parts.append(f"<div class='legend'><b>주황색 칸</b> = 불씨예보가 채운 칸 "
                 f"({led['n_filled']}/{led['n_fields']}, {led['fill_rate']:.0%}). "
                 f"나머지는 연계 자료가 없어 비워 두었으며, 칸마다 사유를 적었습니다."
                 + (f" · 기준 {city_label} {year}년" if city_label else "") + "</div>")
    parts.append("</div>")
    return "\n".join(parts)
