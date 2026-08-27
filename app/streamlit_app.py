"""불씨예보(K-Firebird) 대시보드 — 소방관용 4화면.

  화면1 위험 지도·우선순위    화면2 도로별 AI 점검계획서
  화면3 순찰 시간대·요일·동선  화면4 대응취약·소화전

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

from firebird import dataset as D, explain as X, grid as G, hydrant as H, \
    llm as L, model as M, patrol as P, rules as R  # noqa: E402
from firebird.config import load_config  # noqa: E402

st.set_page_config(page_title="불씨예보 K-Firebird", page_icon="🔥", layout="wide")


# ------------------------------------------------------------------ 데이터 적재

@st.cache_resource
def get_config():
    return load_config()


@st.cache_data(show_spinner="패널 불러오는 중…")
def get_panel(city: str) -> pd.DataFrame:
    return D.load_panel(get_config(), city)


@st.cache_resource
def get_model(city: str):
    cfg = get_config()
    path = cfg.paths.outputs / f"model_{city}.joblib"
    if not path.exists():
        return None
    blob = joblib.load(path)
    return M.TrainedModel(estimator=blob["estimator"], feature_cols=blob["feature_cols"],
                          train_years=blob["train_years"])


@st.cache_data(show_spinner="위험도 계산 중…")
def scored_year(city: str, year: int) -> pd.DataFrame:
    panel = get_panel(city)
    cur = panel[panel["year"] == year].copy()
    model = get_model(city)
    if model is None or cur.empty:
        cur["pred"] = cur.get("fires_lag1", pd.Series(0.0, index=cur.index))
    else:
        cur["pred"] = model.predict(cur)
    cur["위험점수"] = (cur["pred"].rank(pct=True) * 100).round(1)
    cur = cur.sort_values("pred", ascending=False).reset_index(drop=True)
    cur["순위"] = range(1, len(cur) + 1)
    cur["상위%"] = (cur["순위"] / len(cur) * 100).round(1)
    return cur


@st.cache_data
def get_fires(city: str) -> pd.DataFrame:
    path = get_config().paths.processed / f"fires_{city}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def risk_color(score: float) -> list[int]:
    """0~100 -> 노랑에서 진한 빨강으로."""
    t = max(0.0, min(1.0, score / 100.0))
    return [int(255), int(220 * (1 - t)), int(40 * (1 - t)), int(90 + 120 * t)]


def polygon_layer(df: pd.DataFrame, cfg) -> pdk.Layer:
    rows = []
    for _, r in df.iterrows():
        try:
            poly = G.cell_bounds(str(r["grid_id"]), cfg)
        except (ValueError, IndexError):
            continue
        rows.append({"polygon": [list(p) for p in poly],
                     "grid_id": r["grid_id"],
                     "위험점수": float(r["위험점수"]),
                     "순위": int(r["순위"]),
                     "시군구": r.get("sgg", ""),
                     "color": risk_color(float(r["위험점수"]))})
    return pdk.Layer("PolygonLayer", rows, get_polygon="polygon",
                     get_fill_color="color", get_line_color=[80, 80, 80],
                     line_width_min_pixels=1, pickable=True, auto_highlight=True)


def deck(layers: list, df: pd.DataFrame, zoom: float = 10.5) -> pdk.Deck:
    lat = float(df["lat"].mean()) if "lat" in df and df["lat"].notna().any() else 35.54
    lon = float(df["lon"].mean()) if "lon" in df and df["lon"].notna().any() else 129.31
    return pdk.Deck(layers=layers,
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                    map_style=None,
                    tooltip={"text": "격자 {grid_id}\n{시군구}\n위험점수 {위험점수} (순위 {순위})"})


# ------------------------------------------------------------------ 사이드바

cfg = get_config()
st.sidebar.title("🔥 불씨예보")
st.sidebar.caption("K-Firebird — 한정된 인력을 가장 위험한 곳에")

cities = list(cfg["cities"])
city = st.sidebar.selectbox("도시", cities,
                            format_func=lambda c: cfg.city(c)["label"])

try:
    panel = get_panel(city)
except FileNotFoundError as exc:
    st.error(f"패널이 없다. 먼저 파이프라인을 돌려라.\n\n```\n{exc}\n```")
    st.stop()

years = sorted(int(y) for y in panel["year"].unique())
year = st.sidebar.selectbox("기준 연도", years, index=len(years) - 1)
cur = scored_year(city, year)

sgg_options = ["전체"] + sorted(x for x in cur.get("sgg", pd.Series(dtype=str)).unique() if x)
sgg = st.sidebar.selectbox("시군구", sgg_options)
top_pct = st.sidebar.slider("상위 위험 구간(%)", 5, 100, 20, step=5)

view = cur if sgg == "전체" else cur[cur["sgg"] == sgg]
n_top = max(1, int(round(len(view) * top_pct / 100)))
top = view.head(n_top)

if get_model(city) is None:
    st.sidebar.warning("학습된 모델이 없어 '작년 화재' 순으로 표시한다. "
                       "`scripts/04_train_eval.py` 를 먼저 돌려라.")

st.sidebar.divider()
st.sidebar.caption(f"격자 {len(cur):,}개 · {cfg.grid_size_m}m · {year}년 기준")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ 위험 지도·우선순위", "📋 AI 점검계획서", "🚓 순찰 시간·동선", "🚰 대응취약·소화전"])


# ------------------------------------------------------------------ 화면 1
with tab1:
    st.subheader(f"위험 지도 · 상위 {top_pct}% ({len(top):,}격자)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("대상 격자", f"{len(view):,}")
    c2.metric(f"상위 {top_pct}% 격자", f"{len(top):,}")
    if "fires" in view.columns and view["fires"].sum() > 0:
        cap = top["fires"].sum() / view["fires"].sum()
        c3.metric(f"{year}년 화재 포착률", f"{cap:.1%}",
                  help="상위 구간에 실제 화재의 몇 %가 들어왔는가")
        c4.metric("무작위 대비", f"{cap/(top_pct/100):.2f}배")

    st.pydeck_chart(deck([polygon_layer(top, cfg)], top))

    show = [c for c in ["순위", "grid_id", "sgg", "위험점수", "상위%", "fires_lag1",
                        "target_total", "biz_total", "n_hydrant", "fires"]
            if c in top.columns]
    st.dataframe(
        top[show].rename(columns={"grid_id": "격자", "sgg": "시군구", "fires_lag1": "작년화재",
                                  "target_total": "대상물수", "biz_total": "업소수",
                                  "n_hydrant": "소화전수", "fires": f"{year}실제화재"}),
        width='stretch', hide_index=True, height=360)
    st.download_button("우선순위 CSV 내려받기",
                       top[show].to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"priority_{city}_{year}.csv", mime="text/csv")


# ------------------------------------------------------------------ 화면 2
with tab2:
    st.subheader("도로별 AI 점검계획서")
    if top.empty:
        st.info("표시할 격자가 없다.")
    else:
        labels = {f"{r['순위']}위 · {r['grid_id']} · {r.get('sgg','')} "
                  f"(위험 {r['위험점수']:.0f})": r["grid_id"]
                  for _, r in top.head(50).iterrows()}
        pick = st.selectbox("격자 선택", list(labels))
        gid = labels[pick]
        row = cur[cur["grid_id"] == gid].iloc[0]

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**위험을 높인 요인 (SHAP)**")
            model = get_model(city)
            drivers = []
            if model is not None:
                try:
                    d = X.explain_grids(model, cur[cur["grid_id"] == gid])
                    drivers = d["drivers"].iloc[0] if len(d) else []
                    st.dataframe(pd.DataFrame(drivers)[["label", "value", "contribution"]]
                                 .rename(columns={"label": "요인", "value": "현재값",
                                                  "contribution": "기여도"}),
                                 hide_index=True, width='stretch')
                except Exception as exc:                      # noqa: BLE001
                    st.warning(f"SHAP 계산 실패: {exc}")
            else:
                st.info("모델이 없어 요인 분석을 건너뛴다.")

            st.markdown("**업종 맞춤 점검 체크리스트**")
            checklist = R.checklist_for_grid(row, cfg)
            for sec in checklist["sections"]:
                with st.expander(f"{sec['구분']} — {sec['근거']}", expanded=True):
                    for item in sec["항목"]:
                        st.checkbox(item, key=f"{gid}_{item}")

        with right:
            st.markdown("**점검계획서 초안**")
            available = L.is_available(cfg)
            st.caption("로컬 LLM(Ollama) 연결됨" if available else
                       "Ollama 미가동 — 규칙기반 초안으로 생성한다")
            if st.button("초안 생성", type="primary"):
                risk = {"score_0_100": float(row["위험점수"]), "rank": int(row["순위"]),
                        "percentile": float(row["상위%"])}
                with st.spinner("작성 중…"):
                    plan = L.inspection_plan(cfg, str(gid), risk, drivers, checklist,
                                             use_llm=available)
                st.caption(f"생성 경로: {plan['source']}")
                st.text_area("초안", plan["text"], height=460)
                st.download_button("초안 내려받기", plan["text"].encode("utf-8"),
                                   file_name=f"점검계획서_{gid}.txt")


# ------------------------------------------------------------------ 화면 3
with tab3:
    st.subheader("예방순찰: 언제 · 어디를 · 어떤 순서로")
    fires = get_fires(city)
    if fires.empty:
        st.info("화재 원본(parquet)이 없어 시간대 분석을 건너뛴다.")
    else:
        hours = P.hour_profile(fires)
        wdays = P.weekday_profile(fires)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**시간대별 화재**")
            st.bar_chart(hours.set_index("hour")["n"])
        with c2:
            st.markdown("**요일별 화재**")
            st.bar_chart(wdays.set_index("요일")["n"])
        peaks = P.peak_windows(fires, top_n=3)
        if peaks:
            st.success("집중 순찰 권장 시간대: " + ", ".join(
                f"{p['start_hour']:02d}~{p['end_hour']:02d}시 (전체의 {p['share']:.0%})"
                for p in peaks))

    st.markdown("**순찰 동선 (위험 상위 격자 최근접 연결)**")
    k = st.slider("동선에 포함할 격자 수", 5, 40, 15)
    route = P.patrol_route(view.head(k))
    if route.empty:
        st.info("좌표가 있는 격자가 없다.")
    else:
        st.caption(f"총 이동거리 {route['누적거리_m'].iloc[-1]:,.0f}m · 직선거리 기준 "
                   f"(도로망 최적화 아님)")
        path = [{"path": route[["lon", "lat"]].values.tolist(), "name": "순찰경로"}]
        layers = [pdk.Layer("PathLayer", path, get_path="path", get_width=40,
                            get_color=[0, 120, 255], width_min_pixels=3),
                  pdk.Layer("ScatterplotLayer", route, get_position=["lon", "lat"],
                            get_radius=180, get_fill_color=[255, 60, 60], pickable=True)]
        st.pydeck_chart(deck(layers, route))
        st.dataframe(route[[c for c in ["순번", "grid_id", "sgg", "위험점수",
                                        "이동거리_m", "누적거리_m"] if c in route.columns]],
                     hide_index=True, width='stretch')


# ------------------------------------------------------------------ 화면 4
with tab4:
    st.subheader("대응취약 구역 · 소화전 사각지대")
    cov = H.hydrant_coverage(cur)
    if not cov.get("available"):
        st.info("소방용수시설 데이터가 없어 이 화면은 비어 있다.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 격자", f"{cov['n_grids']:,}")
        c2.metric("소화전 없는 격자", f"{cov['grids_without_hydrant']:,}",
                  f"{cov['share_without_hydrant']:.1%}")
        blind = H.blind_spots(cur, cur["pred"], cfg)
        c3.metric(f"고위험 + 소화전 사각", f"{len(blind)}개",
                  help=f"위험 상위 {cfg['hydrant']['high_risk_percentile']}% 중 "
                       f"소화전이 없거나 {cfg['hydrant']['max_dist_m']}m 밖인 격자")

        if not blind.empty:
            st.markdown("**소화전 신설 우선순위**")
            st.dataframe(blind, hide_index=True, width='stretch')
            st.pydeck_chart(deck([
                pdk.Layer("ScatterplotLayer", blind, get_position=["lon", "lat"],
                          get_radius=300, get_fill_color=[255, 0, 0, 160], pickable=True)],
                blind))
            st.download_button("사각지대 CSV",
                               blind.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"hydrant_blindspots_{city}_{year}.csv")

    surge = H.surge_alert(panel, year)
    st.markdown(f"**화재 급증 경보 ({year}년, 직전연도 대비 2배 이상)**")
    if surge.empty:
        st.caption("해당 격자 없음")
    else:
        st.dataframe(surge, hide_index=True, width='stretch')

st.divider()
st.caption("공개 데이터 기반 예측 결과이며 법정 점검주기·관할 판단을 대체하지 않는다. "
           "격자 단위 산출물로 개별 건물을 특정하지 않는다.")
