"""불씨예보(K-Firebird) — 화재예방 업무 지원 시스템.

화면을 기능별로 나열하지 않고 예방과 업무 순서대로 배열한다.

    1. 예방점검 배분   가용 인력 기준 점검 대상 격자 선정
    2. 위험요인·점검표  선정 사유(SHAP)와 업종별 점검 항목, 계획서 초안
    3. 예방순찰 계획   화재 다발 시간대와 순찰 동선
    4. 대응취약 구역   소방용수 사각지대, 화재 급증 구역
    5. 모델 검증       예측 성능과 검증 결과

실행: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

from firebird import dataset as D, evaluate as E, explain as X, grid as G, \
    hydrant as H, llm as L, model as M, operations as OP, patrol as P, rules as R  # noqa: E402
from firebird.config import load_config  # noqa: E402

st.set_page_config(page_title="불씨예보 K-Firebird", page_icon="🔥", layout="wide")

st.markdown("""
<style>
  .big-metric {font-size: 2.1rem; font-weight: 700; line-height: 1.1;}
  .metric-sub {color: #888; font-size: 0.82rem;}
  .callout {border-left: 4px solid #e34a33; background: rgba(227,74,51,.07);
            padding: .7rem 1rem; border-radius: 4px; margin: .5rem 0;}
  .good {border-left-color:#2c8; background: rgba(34,204,136,.07);}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ 적재

@st.cache_resource
def get_config():
    return load_config()


@st.cache_data(show_spinner="패널 불러오는 중…")
def get_panel(city: str) -> pd.DataFrame:
    return D.load_panel(get_config(), city)


@st.cache_resource
def get_model(city: str):
    path = get_config().paths.outputs / f"model_{city}.joblib"
    if not path.exists():
        return None
    b = joblib.load(path)
    return M.TrainedModel(estimator=b["estimator"], feature_cols=b["feature_cols"],
                          train_years=b["train_years"])


@st.cache_data
def get_evaluation() -> dict:
    p = get_config().paths.outputs / "evaluation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def get_fires(city: str) -> pd.DataFrame:
    p = get_config().paths.processed / f"fires_{city}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def get_manifest(city: str) -> dict:
    p = get_config().paths.processed / f"manifest_{city}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner="위험도 계산 중…")
def scored(city: str, year: int) -> pd.DataFrame:
    panel = get_panel(city)
    cur = panel[panel["year"] == year].copy()
    model = get_model(city)
    cur["pred"] = (model.predict(cur) if model is not None
                   else cur.get("fires_lag1", pd.Series(0.0, index=cur.index)))
    cur["위험점수"] = (cur["pred"].rank(pct=True) * 100).round(1)
    cur = cur.sort_values("pred", ascending=False).reset_index(drop=True)
    cur["순위"] = range(1, len(cur) + 1)
    cur["상위%"] = (cur["순위"] / len(cur) * 100).round(1)
    cur["점검대상수"] = OP.inspection_cost(cur)
    return cur


def risk_color(score: float, alpha_boost: int = 0) -> list[int]:
    t = max(0.0, min(1.0, float(score) / 100.0))
    return [255, int(215 * (1 - t)), int(50 * (1 - t)), 70 + int(130 * t) + alpha_boost]


def grid_layer(df: pd.DataFrame, cfg, color_col: str = "위험점수") -> pdk.Layer:
    rows = []
    for _, r in df.iterrows():
        try:
            poly = G.cell_bounds(str(r["grid_id"]), cfg)
        except (ValueError, IndexError):
            continue
        rows.append({"polygon": [list(p) for p in poly], "grid_id": r["grid_id"],
                     "위험점수": float(r.get("위험점수", 0)),
                     "순위": int(r.get("순위", 0)),
                     "시군구": r.get("sgg", ""),
                     "점검대상수": int(r.get("점검대상수", 0)),
                     "color": risk_color(r.get(color_col, 0))})
    return pdk.Layer("PolygonLayer", rows, get_polygon="polygon", get_fill_color="color",
                     get_line_color=[70, 70, 70], line_width_min_pixels=1,
                     pickable=True, auto_highlight=True)


def deck(layers, df, zoom=10.2):
    lat = float(pd.to_numeric(df["lat"], errors="coerce").mean()) if "lat" in df else 35.54
    lon = float(pd.to_numeric(df["lon"], errors="coerce").mean()) if "lon" in df else 129.31
    if not np.isfinite(lat):
        lat, lon = 35.54, 129.31
    return pdk.Deck(layers=layers, map_style=None,
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                    tooltip={"text": "격자 {grid_id} · {시군구}\n"
                                     "위험점수 {위험점수} (순위 {순위})\n점검대상 {점검대상수}개소"})


# ------------------------------------------------------------------ 사이드바

cfg = get_config()
st.sidebar.markdown("## 🔥 불씨예보")
st.sidebar.caption("K-Firebird · 화재예방 업무 지원 시스템")

city = st.sidebar.selectbox("도시", list(cfg["cities"]),
                            format_func=lambda c: cfg.city(c)["label"])
try:
    panel = get_panel(city)
except FileNotFoundError as exc:
    st.error(f"분석 자료가 없습니다. 파이프라인을 먼저 실행하십시오."
             f"\n\n```\n{exc}\n```")
    st.stop()

years = sorted(int(y) for y in panel["year"].unique())
year = st.sidebar.selectbox("기준 연도", years, index=len(years) - 1)
cur = scored(city, year)

st.sidebar.markdown("### 점검 가용 인력")
c1, c2 = st.sidebar.columns(2)
inspectors = c1.number_input("점검관(명)", 1, 100, 4)
per_day = c2.number_input("1일 건수", 1, 50, 8)
days = st.sidebar.slider("점검 기간(일)", 5, 120, 20, step=5)
capacity = OP.Capacity(int(inspectors), int(per_day), int(days))
st.sidebar.caption(f"점검 가능 물량 **{capacity.total_visits:,}건**")

sgg_opts = ["전체"] + sorted(x for x in cur.get("sgg", pd.Series(dtype=str)).unique() if x)
sgg = st.sidebar.selectbox("관할", sgg_opts)
view = cur if sgg == "전체" else cur[cur["sgg"] == sgg].reset_index(drop=True)

if get_model(city) is None:
    st.sidebar.warning("학습된 모델이 없어 전년도 화재 건수 순으로 표시합니다. "
                       "`scripts/04_train_eval.py` 를 먼저 실행하십시오.")
st.sidebar.divider()
man = get_manifest(city)
if man:
    cov = man.get("coverage", {}).get("fire", {})
    st.sidebar.caption(f"격자 {cur['grid_id'].nunique():,}개 · {cfg.grid_size_m}m\n\n"
                       f"화재 {man.get('panel',{}).get('total_fires',0):,.0f}건 "
                       f"· 좌표 확보 {cov.get('rate',0):.1%}")

tabs = st.tabs(["예방점검 배분", "위험요인·점검계획서", "예방순찰 계획",
                "대응취약 구역", "모델 검증"])


# ================================================================== ① 배분
with tabs[0]:
    st.subheader("가용 인력 기준 예방점검 배분")
    st.caption("격자마다 점검 대상물 수가 다릅니다. 위험도 순으로만 자르면 "
               "인력으로 소화할 수 없는 계획이 나오므로, 가용 물량 안에서 배분합니다.")

    cmp = OP.compare_to_topk(view, view["pred"], capacity, cfg.headline_k)
    alloc = OP.allocate(view, view["pred"], capacity)
    o, t = cmp["optimized"], cmp["top_k_percent"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("점검 가능 물량", f"{capacity.total_visits:,}건",
              help=capacity.describe())
    m2.metric("배분 결과", f"{o['n_grids']:,}개 격자",
              f"{o['cost_used']:,.0f}건 배정")
    m3.metric(f"위험도 상위 {cfg.headline_k}% 방식", f"{t['n_grids_affordable']:,}개 격자",
              f"대상 {t['n_grids_selected']:,}개 중 소화 가능분", delta_color="off")
    if "gain_pp" in cmp:
        m4.metric("실제 화재 포착률", f"{o['actual_capture_rate']:.1%}",
                  f"{cmp['gain_pp']:+.1f}%p")

    if t["n_grids_affordable"] < t["n_grids_selected"]:
        st.markdown(
            f"<div class='callout'><b>위험도 상위 {cfg.headline_k}% 안의 점검 대상은 "
            f"{t['cost_if_all']:,.0f}개소입니다.</b><br>"
            f"현재 가용 물량 {capacity.total_visits:,}건으로는 {t['n_grids_selected']:,}개 격자 중 "
            f"{t['n_grids_affordable']:,}개까지만 점검할 수 있습니다."
            f"</div>", unsafe_allow_html=True)
        if "gain_pp" in cmp:
            st.markdown(
                f"<div class='callout good'>동일 인력으로 <b>{o['n_grids']:,}개 격자</b>를 점검하여 "
                f"실제 화재 <b>{o['actual_capture_rate']:.1%}</b>를 포착합니다 "
                f"(상위 {cfg.headline_k}% 방식 {t['actual_capture_rate']:.1%}, "
                f"<b>{cmp['gain_pp']:+.1f}%p</b>).</div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.pydeck_chart(deck([grid_layer(alloc, cfg)], alloc))
    with right:
        st.markdown("**점검 순위표**")
        cols = [c for c in ["점검순서", "grid_id", "sgg", "위험점수", "점검대상수",
                            "expected_fires", "누적비용"] if c in alloc.columns]
        st.dataframe(alloc[cols].rename(columns={
            "grid_id": "격자", "sgg": "관할", "expected_fires": "기대화재",
            "누적비용": "누적건수"}), hide_index=True, height=430, width='stretch')
    st.download_button("점검 계획 내려받기 (CSV)",
                       alloc.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"점검배분_{city}_{year}.csv", mime="text/csv")


# ================================================================== ② 이유
with tabs[1]:
    st.subheader("선정 사유와 점검 항목")
    if alloc.empty:
        st.info("배분된 격자가 없습니다.")
    else:
        labels = {f"{r['점검순서']}순위 · {r['grid_id']} · {r.get('sgg','')} "
                  f"(위험 {r['위험점수']:.0f} · 대상 {int(r['점검대상수'])}개소)": r["grid_id"]
                  for _, r in alloc.head(60).iterrows()}
        pick = st.selectbox("점검 대상 격자", list(labels))
        gid = labels[pick]
        row = cur[cur["grid_id"] == gid].iloc[0]

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**위험도 상승 요인**")
            model = get_model(city)
            drivers = []
            if model is not None:
                try:
                    d = X.explain_grids(model, cur[cur["grid_id"] == gid])
                    drivers = d["drivers"].iloc[0] if len(d) else []
                    if drivers:
                        dd = pd.DataFrame(drivers)
                        st.dataframe(dd[["label", "value", "contribution"]].rename(
                            columns={"label": "요인", "value": "현재값", "contribution": "기여도"}),
                            hide_index=True, width='stretch')
                        st.bar_chart(dd.set_index("label")["contribution"])
                except Exception as exc:                     # noqa: BLE001
                    st.warning(f"SHAP 계산 실패: {exc}")
            else:
                st.info("학습된 모델이 없어 요인 분석을 표시할 수 없습니다.")

            st.markdown("**업종·소방시설별 점검 항목**")
            checklist = R.checklist_for_grid(row, cfg)
            st.caption(f"총 {checklist['n_items']}개 항목")
            for sec in checklist["sections"]:
                with st.expander(f"{sec['구분']} — {sec['근거']}", expanded=False):
                    for item in sec["항목"]:
                        st.checkbox(item, key=f"{gid}_{item}")

        with right:
            st.markdown("**점검계획서 초안**")
            backends = L.available_backends(cfg)
            real = [b for b in backends if b != "rule_based"]
            st.caption("생성 방식: " + (f"{real[0]}" if real else "규칙 기반"))
            if st.button("계획서 작성", type="primary"):
                risk = {"score_0_100": float(row["위험점수"]), "rank": int(row["순위"]),
                        "percentile": float(row["상위%"])}
                with st.spinner("작성 중입니다…"):
                    plan = L.inspection_plan(cfg, str(gid), risk, drivers, checklist)
                st.caption(f"작성 경로: `{plan['source']}`")
                st.markdown(plan["text"])
                st.download_button("계획서 내려받기", plan["text"].encode("utf-8"),
                                   file_name=f"점검계획서_{gid}.md")


# ================================================================== ③ 순찰
with tabs[2]:
    st.subheader("예방순찰 시간대 및 동선")
    fires = get_fires(city)
    if fires.empty:
        st.info("화재 이력 자료가 없어 시간대 분석을 표시할 수 없습니다.")
    else:
        with_time = int(fires["hour"].notna().sum()) if "hour" in fires else 0
        st.caption(f"발생 시각이 기록된 화재 {with_time:,}건 기준 "
                   f"(2021년 자료는 시각이 누락되어 제외)")
        hours, wdays = P.hour_profile(fires), P.weekday_profile(fires)
        c1, c2 = st.columns(2)
        c1.markdown("**시간대별 화재**"); c1.bar_chart(hours.set_index("hour")["n"])
        c2.markdown("**요일별 화재**"); c2.bar_chart(wdays.set_index("요일")["n"])
        peaks = P.peak_windows(fires, top_n=3)
        if peaks:
            st.success("순찰 집중 권장 시간대: " + " · ".join(
                f"{p['start_hour']:02d}~{p['end_hour']:02d}시 ({p['share']:.0%})" for p in peaks))

    st.markdown("**순찰 동선** — 위험 상위 격자를 최단 경로로 연결")
    k = st.slider("순찰 격자 수", 5, 40, 15)
    route = P.patrol_route(view.head(k))
    if route.empty:
        st.info("좌표가 확보된 격자가 없습니다.")
    else:
        st.caption(f"총 이동거리 {route['누적거리_m'].iloc[-1]:,.0f}m "
                   f"(직선거리 기준, 도로망 미반영)")
        path = [{"path": route[["lon", "lat"]].astype(float).values.tolist()}]
        st.pydeck_chart(deck([
            pdk.Layer("PathLayer", path, get_path="path", get_width=45,
                      get_color=[0, 122, 255], width_min_pixels=3),
            pdk.Layer("ScatterplotLayer", route, get_position=["lon", "lat"],
                      get_radius=200, get_fill_color=[230, 60, 50], pickable=True)], route))
        st.dataframe(route[[c for c in ["순번", "grid_id", "sgg", "위험점수",
                                        "이동거리_m", "누적거리_m"] if c in route.columns]],
                     hide_index=True, width='stretch')


# ================================================================== ④ 대응취약
with tabs[3]:
    st.subheader("소방용수 사각지대 및 급증 구역")
    cov = H.hydrant_coverage(cur)
    if not cov.get("available"):
        st.info("소방용수시설 자료가 없습니다.")
    else:
        blind = H.blind_spots(cur, cur["pred"], cfg)
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 격자", f"{cov['n_grids']:,}")
        c2.metric("소화전 미설치 격자", f"{cov['grids_without_hydrant']:,}",
                  f"{cov['share_without_hydrant']:.1%}", delta_color="off")
        c3.metric("고위험·용수 사각", f"{len(blind)}개",
                  help=f"위험 상위 {cfg['hydrant']['high_risk_percentile']}% 중 소화전이 없거나 "
                       f"{cfg['hydrant']['max_dist_m']}m 밖")
        if not blind.empty:
            st.markdown("**소화전 신설 우선순위**")
            st.dataframe(blind, hide_index=True, width='stretch')
            st.pydeck_chart(deck([pdk.Layer(
                "ScatterplotLayer", blind.assign(lon=pd.to_numeric(blind["lon"]),
                                                 lat=pd.to_numeric(blind["lat"])),
                get_position=["lon", "lat"], get_radius=350,
                get_fill_color=[220, 30, 30, 170], pickable=True)], blind))
            st.download_button("사각지대 내려받기 (CSV)", blind.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"소화전사각_{city}_{year}.csv")

    surge = H.surge_alert(panel, year)
    st.markdown(f"**화재 급증 구역 ({year}년, 전년 대비 2배 이상)**")
    if surge.empty:
        st.caption("해당 격자가 없습니다.")
    else:
        st.dataframe(surge, hide_index=True, width='stretch')


# ================================================================== ⑤ 검증
with tabs[4]:
    st.subheader("예측 성능 검증 결과")
    ev = get_evaluation()
    if not ev:
        st.info("`scripts/04_train_eval.py` 를 실행하면 검증 결과가 표시됩니다.")
    else:
        t = ev.get("temporal", {})
        h = t.get("headline", {})
        key = f"top{cfg.headline_k}"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"상위 {cfg.headline_k}% 포착률", f"{h.get('model_capture',0):.1%}",
                  f"{h.get('delta_pp',0):+.1f}%p vs 베이스라인")
        c2.metric("무작위 대비", f"{h.get('model_lift',0):.2f}배")
        pei = t.get("model", {}).get("pei", {}).get(key)
        if pei:
            c3.metric("PEI (달성 가능 최대 대비)", f"{pei:.1%}",
                      help="해당 자료에서 도달 가능한 최대 성능 대비 비율. "
                           "지역이 달라도 비교 가능한 지표입니다.")
        cal = t.get("calibration", {})
        if cal:
            c4.metric("예측 건수 정확도", f"{cal.get('total_ratio',0):.2f}",
                      f"예측 {cal.get('total_predicted',0):.0f} / 실제 {cal.get('total_actual',0):.0f}",
                      delta_color="off")

        st.caption(f"검증 방식: {t.get('train_years',[None])[0]}~{t.get('train_years',[None])[-1]}년 "
                   f"자료로 학습 후 {t.get('test_year')}년 예측 "
                   f"(검증 연도 자료는 학습에 미사용)")

        st.markdown("**포착률 곡선**")
        capm, capb = t.get("model", {}).get("capture", {}), t.get("baseline", {}).get("capture", {})
        if capm:
            curve = pd.DataFrame({"상위 %": [int(k[3:]) for k in capm],
                                  "모델": list(capm.values()),
                                  "베이스라인(작년화재순)": [capb.get(k) for k in capm]})
            st.line_chart(curve.set_index("상위 %"))

        cc = st.columns(2)
        with cc[0]:
            if "logo" in ev:
                st.markdown("**관할 제외 검증** — 해당 구·군을 학습에서 제외")
                lg = pd.DataFrame(ev["logo"]["per_group"])
                st.dataframe(lg[["group", "capture", "baseline_capture", "total_fires"]].rename(
                    columns={"group": "제외 지역", "capture": "모델", "baseline_capture": "베이스라인",
                             "total_fires": "화재"}), hide_index=True, width='stretch')
            if "ranking_comparison" in ev and ev["ranking_comparison"].get("capture_by_model"):
                st.markdown("**알고리즘 비교**")
                rc = ev["ranking_comparison"]["capture_by_model"]
                st.dataframe(pd.DataFrame({"모델": list(rc), "포착률": list(rc.values())}),
                             hide_index=True, width='stretch')
        with cc[1]:
            if "equity" in t:
                st.markdown("**관할별 배분 형평성**")
                eq = pd.DataFrame(t["equity"])
                eq = eq[eq["group"].astype(str).str.strip() != ""]
                st.dataframe(eq[["group", "share_of_fires", "share_of_inspections",
                                 "inspection_vs_risk"]].rename(
                    columns={"group": "관할", "share_of_fires": "화재비중",
                             "share_of_inspections": "점검비중", "inspection_vs_risk": "비율"}),
                    hide_index=True, width='stretch')
                es = t.get("equity_summary", {})
                if es.get("underserved"):
                    st.warning(f"{', '.join(es['underserved'])}: 화재 비중 대비 "
                               f"점검 배분이 낮습니다.")
            if "resolution_scenarios" in ev:
                st.markdown("**주소 해상도별 성능**")
                st.dataframe(pd.DataFrame(ev["resolution_scenarios"]),
                             hide_index=True, width='stretch')
                st.caption("화재의 약 63%는 도로명이 없어 읍면동 중심좌표로 배정됩니다. "
                           "도로명이 확보된 건만으로 별도 검증한 결과 성능 저하는 없었습니다.")

st.divider()
st.caption("본 시스템은 공개 데이터 기반 예측 결과이며, 법정 점검주기 및 관할 판단을 "
           "대체하지 않습니다. 격자 단위 집계로 개별 건물을 특정하지 않습니다.")
