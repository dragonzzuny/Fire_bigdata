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

#: 지금 자료로 못 채우는 칸. (사유, 채울 수 있는 경로)
#:
#: '자료가 없다'로 끝내면 못 하는 일처럼 읽힌다. 실제로는 대부분 **연계만 하면
#: 채워지는** 칸이고, 개인정보라서 애초에 넣으면 안 되는 칸은 따로 있다.
#: 둘을 구분해 적는다 — 심사에서도 현장에서도 이 구분이 답이다.
NOT_AVAILABLE = {
    "대표자": ("공개 데이터에 개인정보가 포함되지 않음", "관서 내부 대장에서 기입"),
    "전화번호": ("공개 데이터에 개인정보가 포함되지 않음", "관서 내부 대장에서 기입"),
    "지정일자": ("화재예방강화지구 지정 이력은 소방본부 내부 자료",
               "소방본부 지정 대장 연계"),
    "건축연도": ("건축물대장 미연계", "국토부 건축물대장 표제부 API(사용승인일)"),
    "연면적": ("건축물대장 미연계", "국토부 건축물대장 표제부 API(연면적)"),
    "건축면적": ("건축물대장 미연계", "국토부 건축물대장 표제부 API(건축면적)"),
    "유동인구": ("생활인구 자료 미연계", "통신사 유동인구(유상) 또는 카드 매출 추정"),
    "상주인구": ("주민등록 자료 미연계", "행안부 주민등록 인구현황(읍면동 단위)"),
    "소방조직": ("의용소방대 편성 자료 미연계", "소방본부 의용소방대 명부 연계"),
    "화재안전조사 결과 및 조치명령": ("점검이력 데이터에 결합 키가 없어 미연계",
                            "대상물 관리번호가 포함된 점검이력 제공 시 연계"),
    "소방설비등의 설치지원": ("설치지원 집행 자료 미연계", "소방본부 사업 집행 자료 연계"),
    "소방훈련 및 소방교육": ("훈련·교육 실시 기록 미연계", "소방본부 훈련 기록 연계"),
}


def blank_reason(key: str) -> str:
    v = NOT_AVAILABLE.get(key)
    return v[0] if isinstance(v, tuple) else str(v or "")


def fill_route(key: str) -> str:
    v = NOT_AVAILABLE.get(key)
    return v[1] if isinstance(v, tuple) and len(v) > 1 else ""


@dataclass(frozen=True)
class Field:
    """서식 한 칸. value 가 비면 blank_reason 이 왜 비었는지 말한다."""
    label: str
    value: str = ""
    unit: str = ""
    blank_reason: str = ""
    source: str = ""            # 이 값이 어느 자료에서 나왔는지
    route: str = ""             # 비었다면, 무엇을 연계하면 채워지는가


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
                grid_m: int = 500,
                building: dict | None = None) -> dict:
    """격자 하나를 [별지 제11호서식] 관리대장 칸에 맞춰 채운다.

    row 는 그 해 격자 한 줄(패널). drivers 는 SHAP 상위 요인이며 '지구특징'에 쓴다.
    building 은 건축물대장 집계(buildings.stats_for)이며, 있으면 연면적·건축면적·
    건축연도 칸이 채워진다. 없으면 그 칸은 비워 두고 연계 경로만 남는다.
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
    bd = building or {}
    n_bld = int(bd.get("_n", 0) or 0)
    for k in ("대표자", "전화번호", "지정일자", "건축연도", "연면적", "건축면적",
              "유동인구", "상주인구", "소방조직"):
        unit = {"연면적": "㎡", "건축면적": "㎡", "유동인구": "명",
                "상주인구": "명", "소방조직": "명"}.get(k, "")
        val = str(bd.get(k, "") or "")
        src = ""
        if val:
            src = (f"건축물대장 표제부 {n_bld:,}동 "
                   + ("중앙값" if k == "건축연도" else "합계"))
        fields[k] = Field(k, value=val, unit=unit, source=src,
                          blank_reason="" if val else blank_reason(k),
                          route="" if val else fill_route(k))

    # 서식의 '건물동수' 는 말 그대로 건축물 동수다. 특정소방대상물 수는 대상물
    # 단위라 한 건물에 여러 건이 등록될 수 있어 동수보다 크게 나온다
    # (실측: 한 격자에서 대상물 413건 vs 건축물 206동). 둘을 같은 칸에 넣으면
    # 서식이 묻는 값과 다른 값이 들어간다. 건축물대장이 있으면 그것을 쓴다.
    n_target = num("target_total")
    if n_bld:
        fields["건물동수"] = Field("건물동수", f"{n_bld:,}", "개",
                                 source="건축물대장 표제부 동수")
    else:
        fields["건물동수"] = Field(
            "건물동수", blank_reason="건축물대장 미연계",
            route="국토부 건축물대장 표제부 API(동수)")
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
    if n_target:
        feats.append(f"특정소방대상물 {n_target:,}개소")
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

    sections = {k: blank_reason(k) for k in
                ("화재안전조사 결과 및 조치명령", "소방설비등의 설치지원",
                 "소방훈련 및 소방교육")}
    filled = sum(1 for f in fields.values() if f.value)
    return {
        "law": FORM_LAW, "form_no": FORM_NO, "title": FORM_TITLE,
        "grid_id": gid, "fields": fields, "empty_sections": sections,
        "n_fields": len(fields), "n_filled": filled,
        "fill_rate": filled / max(len(fields), 1),
    }


def ledger_values(led: dict) -> tuple[dict, dict]:
    """zone_ledger 결과를 서식 채우기용으로 옮긴다.

    빈칸에는 아무것도 쓰지 않는다. 서식은 정부 원본 그대로 두고, 비운 칸과
    그 사유는 서식 **바깥**에 적는다 — 칸 안에 회색 글씨를 넣는 순간
    원본과 다른 문서가 된다.
    """
    vals, bl = {}, {}
    for key, f in led["fields"].items():
        if not f.value:
            continue
        if key in BULLET_KEYS:
            bl[key] = [x.strip() for x in str(f.value).split(" · ") if x.strip()]
        else:
            vals[key] = f.value
    return vals, bl


#: 가운뎃점이 인쇄된 서술 칸.
BULLET_KEYS = ("지구특징", "취약요소", "도로여건", "현대화사업")


def fill_official(led: dict, pdf_path, out_png, *, dpi: int = 200):
    """법제처 서식 PDF 위에 값을 얹어 그림을 만든다."""
    from . import formfill

    vals, bl = ledger_values(led)
    return formfill.fill(pdf_path, vals, bl, out_png, dpi=dpi)


# ---------------------------------------------------------------- 렌더링

def _val(f: Field) -> str:
    """칸 안에 들어갈 내용.

    채운 값은 검은 글씨 그대로 둔다 — 실제로 손으로 채운 대장과 같아야 한다.
    빈칸에는 왜 비었는지를 아주 옅게 적는다. 빈칸을 그럴듯한 값으로 메우면
    결재 문서가 아니라 추정치가 되고, 아무 표시도 없으면 '아직 안 썼다'와
    '쓸 수 없다'가 구분되지 않는다.
    """
    import html as _h
    if f.value:
        tip = f' title="{_h.escape(f.source)}"' if f.source else ""
        return f'<span class="v"{tip}>{_h.escape(f.value)}</span>'
    tip = f' title="연계 경로: {_h.escape(f.route)}"' if f.route else ""
    return f'<span class="why"{tip}>{_h.escape(f.blank_reason or "")}</span>'


def _unit(f: Field) -> str:
    return f'<span class="u">{f.unit}</span>' if f.unit else ""


LEDGER_CSS = """
<style>
.form { font-family:'Malgun Gothic','맑은 고딕','Noto Sans CJK KR',sans-serif;
        color:#000; background:#fff; }
.form .hdr { font-size:11.5px; margin:0 0 8px; letter-spacing:-.2px; }
.form h2 { text-align:center; font-size:22px; letter-spacing:.32em;
           margin:0 0 6px; font-weight:700; }
.form .side { text-align:right; font-size:11px; margin:0 0 3px; }
.form table { border-collapse:collapse; width:100%; table-layout:fixed;
              font-size:12px; margin:0 0 9px; }
.form th, .form td { border:1px solid #000; padding:3px 6px; height:23px;
                     vertical-align:middle; word-break:break-all; }
.form th { font-weight:400; text-align:center; }
.form td { text-align:left; }
.form td.r { text-align:right; }
.form .sec td { text-align:left; font-weight:700; font-size:12.5px;
                padding:4px 8px; }
.form .v { color:#111; }
.form .u { color:#000; font-size:11px; }
.form .why { color:#c9ccd1; font-size:9.5px; font-style:italic; }
.form .dot { color:#000; }
.form .foot { text-align:right; font-size:10.5px; margin-top:4px; }
.form .legend { font-size:11px; color:#555; margin:10px 0 0; line-height:1.6;
                border-top:1px dashed #ccc; padding-top:8px; }
</style>
"""


def render_ledger_html(led: dict, *, city_label: str = "", year: int = 0) -> str:
    """[별지 제11호서식] 관리대장을 **원본 서식 배치 그대로** 만든다.

    칸 순서·묶음·단위 표기·괘선을 법제처 PDF 와 같게 둔다. 색을 칠하거나
    칸을 합치면 그 순간 '비슷하게 만든 표'가 되고, 결재선에서 되돌아온다.
    채운 값과 빈칸은 배경색이 아니라 글자로 구분한다.
    """
    F = led["fields"]

    def row2(a, b, ua="", ub=""):
        return (f"<tr><th>{F[a].label}</th><td>{_val(F[a])}</td>"
                f"<th>{F[b].label}</th><td>{_val(F[b])}</td></tr>")

    def row2u(a, b):
        """단위가 칸 오른쪽에 인쇄되어 있는 행(연면적 ㎡, 건물동수 개 …)."""
        return (f"<tr><th>{F[a].label}</th>"
                f"<td class='r'>{_val(F[a])} {_unit(F[a])}</td>"
                f"<th>{F[b].label}</th>"
                f"<td class='r'>{_val(F[b])} {_unit(F[b])}</td></tr>")

    def sec(no, name):
        return (f"<table><tr class='sec'><td>{no}. {name}</td></tr></table>")

    p = [LEDGER_CSS, "<div class='form'>",
         f"<div class='hdr'>■ {led['law']} [{led['form_no']}]</div>",
         f"<h2>{led['title']}</h2>",
         "<div class='side'>(앞쪽)</div>",
         sec(1, "화재예방강화지구 지정현황"),
         "<table><colgroup><col width='96'><col><col width='96'><col></colgroup>",
         row2("명칭", "위치"),
         row2("대표자", "전화번호"),
         row2("지정일자", "건축연도"),
         row2u("연면적", "건축면적"),
         row2u("건물동수", "점포수"),
         row2u("유동인구", "상주인구"),
         f"<tr><th>{F['소방조직'].label}</th>"
         f"<td class='r'>{_val(F['소방조직'])} {_unit(F['소방조직'])}</td>"
         f"<th></th><td></td></tr>",
         f"<tr><th>{F['건물구조'].label}</th><td>{_val(F['건물구조'])}</td>"
         f"<th>{F['지구면적'].label}</th>"
         f"<td class='r'>{_val(F['지구면적'])} {_unit(F['지구면적'])}</td></tr>",
         "</table>"]

    # 소방시설 — 원본은 3행 2열, 각 칸이 [ ]설비명 / 값 / 개 로 나뉜다.
    p.append("<table><colgroup><col width='96'><col width='118'><col>"
             "<col width='34'><col width='118'><col><col width='34'>"
             "</colgroup>")
    rows = [("소방시설:소화설비", "소방시설:경보설비"),
            ("소방시설:피난구조설비", "소방시설:소방용수"),
            ("소방시설:소화활동설비", "소방시설:기타설비")]
    for i, (a, b) in enumerate(rows):
        head = "<th rowspan='3'>소방시설</th>" if i == 0 else ""
        p.append(f"<tr>{head}"
                 f"<th>[&nbsp;&nbsp;]{F[a].label}</th>"
                 f"<td class='r'>{_val(F[a])}</td><th>개</th>"
                 f"<th>[&nbsp;&nbsp;]{F[b].label}</th>"
                 f"<td class='r'>{_val(F[b])}</td><th>개</th></tr>")
    p.append("</table>")

    p.append("<table><colgroup><col width='96'><col width='60'><col>"
             "<col width='34'><col width='130'><col><col width='34'>"
             "</colgroup>")
    p.append(f"<tr><th>소방관서거리</th><th>본서</th>"
             f"<td class='r'>{_val(F['소방관서거리:본서'])}</td><th>km</th>"
             f"<th>관할119안전센터</th>"
             f"<td class='r'>{_val(F['소방관서거리:관할119안전센터'])}</td>"
             f"<th>km</th></tr>")
    p.append("</table>")

    # 지구특징·취약요소·도로여건·현대화사업 — 원본은 각 칸에 가운뎃점 두 줄.
    p.append("<table><colgroup><col width='96'><col></colgroup>")
    for k in ("지구특징", "취약요소", "도로여건", "현대화사업"):
        f = F[k]
        if f.value:
            items = [x.strip() for x in str(f.value).split(" · ") if x.strip()]
        else:
            items = []
        body = "".join(
            f"<div><span class='dot'>·</span> "
            f"<span class='v'>{__import__('html').escape(it)}</span></div>"
            for it in items) or f"<div><span class='dot'>·</span> {_val(f)}</div>"
        if len(items) < 2:
            body += "<div><span class='dot'>·</span></div>"
        p.append(f"<tr><th>{f.label}</th><td>{body}</td></tr>")
    p.append("</table>")

    heads = {"화재안전조사 결과 및 조치명령": ("일자", "조치사항", "보완일자"),
             "소방설비등의 설치지원": ("일자", "설치내용", "완료일자"),
             "소방훈련 및 소방교육": ("훈련일자", "참석인원", "교육일자", "참석인원")}
    widths = {3: "<col width='120'><col><col width='120'>",
              4: "<col width='110'><col><col width='110'><col>"}
    for i, (name, reason) in enumerate(led["empty_sections"].items(), 2):
        tail = "(최근 3년)" if "훈련" in name or "조사" in name else ""
        p.append(sec(i, name + tail))
        h = heads[name]
        p.append(f"<table><colgroup>{widths[len(h)]}</colgroup>")
        p.append("<tr>" + "".join(f"<th>{x}</th>" for x in h) + "</tr>")
        p.append(f"<tr><td colspan='{len(h)}'>"
                 f"<span class='why'>{reason}</span></td></tr>")
        for _ in range(2):
            p.append("<tr>" + "<td></td>" * len(h) + "</tr>")
        p.append("</table>")

    p.append("<div class='foot'>210mm×297mm[백상지(80g/㎡) 또는 중질지(80g/㎡)]</div>")
    linkable = sorted({f.label for f in F.values()
                       if not f.value and f.route and "개인정보" not in f.blank_reason})
    private = sorted({f.label for f in F.values()
                      if not f.value and "개인정보" in f.blank_reason})
    p.append(
        f"<div class='legend'><b>불씨예보가 채운 칸 "
        f"{led['n_filled']}/{led['n_fields']}</b> ({led['fill_rate']:.0%}). "
        f"서식·괘선·칸 순서는 법제처 원본 그대로이며, 값만 데이터에서 채웠습니다."
        + (f"<br><b>연계하면 채워지는 칸</b> — {', '.join(linkable)}. "
           if linkable else "")
        + (f"<br><b>넣지 않는 칸</b> — {', '.join(private)} (개인정보)."
           if private else "")
        + (f"<br>기준 {city_label} {year}년." if city_label else "") + "</div>")
    p.append("</div>")
    return "\n".join(p)
