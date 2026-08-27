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

from datetime import date  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

from firebird import dataset as D, evaluate as E, explain as X, grid as G, \
    hydrant as H, llm as L, model as M, operations as OP, patrol as P, \
    monthly as MO, patrol_modes as PM, plans as PLN, routing as RT, rules as R, \
    assistant as AS, lawdata as LW, stations as ST  # noqa: E402
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
def get_weather(city: str) -> pd.DataFrame:
    p = get_config().paths.processed / f"weather_monthly_{city}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner="월별 위험 계수 계산 중…")
def get_month_fit(city: str) -> dict:
    cfg = get_config()
    fires = get_fires(city)
    if fires.empty:
        return {}
    mf = MO.fires_by_month(fires, cfg.year_min, cfg.year_max)
    return MO.fit_month_risk(mf, get_weather(city))


@st.cache_data(show_spinner="소방 법령 불러오는 중…")
def get_law_articles() -> pd.DataFrame:
    try:
        return LW.collect(get_config())
    except Exception:                                   # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner="관서 위치 확인 중…")
def get_stations(city: str, year: int, level: str) -> pd.DataFrame:
    cfg = get_config()
    return ST.station_table(scored(city, year), cfg, level=level,
                            city_label=cfg.city(city)["label"])


@st.cache_resource(show_spinner="업무 자료 색인 중…")
def get_index(city: str, year: int):
    cfg = get_config()
    return AS.build_index(get_law_articles(), scored(city, year),
                          city_label=cfg.city(city)["label"], year=year)


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
equity_share = st.sidebar.slider(
    "관할별 최소 배분", 0.0, 1.0,
    float(cfg.get("operations", {}).get("equity_min_share", 1.0)), step=0.25,
    help="0 = 효율만 고려(특정 관할 쏠림 가능), "
         "1.0 = 각 관할이 화재 비중만큼 배분받도록 보장")

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
                "순찰·점검 계획서", "대응취약 구역", "모델 검증", "업무 도우미"])


# ================================================================== ① 배분
with tabs[0]:
    st.subheader("가용 인력 기준 예방점검 배분")
    st.caption("격자마다 점검 대상물 수가 다릅니다. 위험도 순으로만 자르면 "
               "인력으로 소화할 수 없는 계획이 나오므로, 가용 물량 안에서 배분합니다.")

    cmp = OP.compare_to_topk(view, view["pred"], capacity, cfg.headline_k)
    alloc, eq_info = OP.allocate_with_equity(view, view["pred"], capacity,
                                             min_share=equity_share)
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
    if eq_info.get("equity_constrained"):
        rows = [{"관할": g, "화재 비중": v["risk_share"],
                 "배분 비중": v["budget_share"],
                 "비율": (v["budget_share"] / v["risk_score"]) if False else
                        (v["budget_share"] / v["risk_share"] if v["risk_share"] else float("nan"))}
                for g, v in eq_info["by_group"].items() if v["risk_share"] > 0]
        if rows:
            with st.expander(f"관할별 배분 형평성 (최소 배분 {equity_share:.2f} 적용)"):
                st.dataframe(pd.DataFrame(rows).sort_values("화재 비중", ascending=False),
                             hide_index=True, width='stretch')
                st.caption("비율 1.0 = 화재 비중만큼 배분. 슬라이더를 0으로 내리면 "
                           "효율만 고려하여 특정 관할에 쏠릴 수 있습니다.")

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
            ai_on = L.is_available(cfg)
            st.caption("AI 문서 작성 " + ("사용 가능" if ai_on else "미연결 — 표준 서식으로 작성됩니다"))
            if st.button("계획서 작성", type="primary"):
                risk = {"score_0_100": float(row["위험점수"]), "rank": int(row["순위"]),
                        "percentile": float(row["상위%"])}
                with st.spinner("작성 중입니다…"):
                    plan = L.inspection_plan(cfg, str(gid), risk, drivers, checklist)
                st.caption("AI 작성" if str(plan["source"]).startswith(("cli", "api", "ollama"))
                           else "표준 서식 작성")
                st.markdown(plan["text"])
                st.download_button("계획서 내려받기", plan["text"].encode("utf-8"),
                                   file_name=f"점검계획서_{gid}.md")


# ================================================================== ③ 순찰
with tabs[2]:
    st.subheader("예방순찰 계획")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    mode_key = c1.selectbox(
        "순찰 목적", list(PM.MODES), format_func=lambda k: PM.MODES[k].label,
        help="목적이 다르면 가야 할 곳도, 시간도, 볼 것도 다릅니다.")
    mode = PM.MODES[mode_key]
    level = c2.selectbox("출동 단위", ["center", "station"],
                         format_func=lambda x: "119안전센터" if x == "center" else "소방서")
    n_grids = c3.number_input("순찰 격자 수", 4, 80, mode.default_k)
    budget = c4.number_input("1회 순찰 시간(분)", 0, 240, 60,
                             help="0이면 제한 없음. 초과하면 회차를 나눕니다.")
    st.caption(mode.purpose)

    r1, r2, r3 = st.columns([3, 1, 1])
    admin_level = r2.selectbox("지역 단위", ["emd", "sgg", "station", "center"],
                               format_func=lambda x: dict(D.ADMIN_LEVELS)[x])
    opts = sorted(x for x in cur.get(admin_level, pd.Series(dtype=str)).unique()
                  if str(x).strip())
    pick = r1.multiselect(f"{dict(D.ADMIN_LEVELS)[admin_level]} 선택 (비우면 전체)", opts)
    use_road = r3.toggle("도로 기준", value=True)
    st.session_state["patrol_mode_key"] = mode_key

    scope = cur if not pick else cur[cur[admin_level].astype(str).isin(pick)]
    targets = PM.select_targets(scope, mode, int(n_grids))

    if targets.empty:
        st.info("선택한 지역에 순찰 대상 격자가 없습니다.")
    else:
        stations = get_stations(city, year, level)
        targets = ST.assign_dispatch(targets, stations, level=level)
        hours = PM.recommended_hours(mode, get_fires(city))

        with st.spinner("관서 출발 동선 계산 중…"):
            plan = RT.plan_from_stations(targets, stations, level=level,
                                         use_road=use_road, budget_min=float(budget))
        st.session_state["patrol_plan"] = plan
        st.session_state["patrol_targets"] = targets

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("순찰 대상", f"{len(targets):,}격자")
        m2.metric("권장 시간대", f"{hours[0]:02d}–{hours[1]:02d}시")
        m3.metric("출동 관서", f"{targets['출동관서'].nunique()}개")
        m4.metric("총 이동", f"{plan['total_km']:.1f} km",
                  f"최장 {plan['max_team_km']:.1f} km", delta_color="off")

        st.caption("거리 기준: " + ("실제 도로 주행거리 (OSRM)"
                                 if plan["distance_source"] == "osrm"
                                 else "직선거리 × 우회계수 1.35 (도로망 서버 미응답)")
                   + " · 동선은 관서에서 출발해 관서로 복귀합니다.")

        if plan["summary"].empty:
            st.warning("관서 좌표를 확인하지 못해 동선을 만들지 못했습니다.")
        else:
            st.markdown("**관서별 순찰 구역**")
            st.dataframe(plan["summary"], hide_index=True, width='stretch')

            palette = [[227, 74, 51], [43, 108, 176], [47, 158, 110], [200, 120, 20],
                       [130, 70, 180], [20, 150, 160], [180, 60, 120], [90, 110, 40],
                       [230, 160, 40], [70, 70, 200], [160, 40, 40], [40, 160, 90]]
            layers, depots = [], []
            for i, r in enumerate(plan["routes"]):
                col = palette[i % len(palette)]
                dep = r.attrs.get("depot", {})
                path = ([[dep.get("lon"), dep.get("lat")]] if dep else []) \
                    + r[["lon", "lat"]].astype(float).values.tolist() \
                    + ([[dep.get("lon"), dep.get("lat")]] if dep else [])
                layers.append(pdk.Layer("PathLayer", [{"path": path}], get_path="path",
                                        get_width=45, get_color=col, width_min_pixels=3))
                layers.append(pdk.Layer("ScatterplotLayer", r, get_position=["lon", "lat"],
                                        get_radius=180, get_fill_color=col + [200],
                                        pickable=True))
                if dep:
                    depots.append({"lon": dep["lon"], "lat": dep["lat"],
                                   "name": dep.get("name", "")})
            if depots:
                layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame(depots),
                                        get_position=["lon", "lat"], get_radius=330,
                                        get_fill_color=[20, 20, 20, 230], pickable=True))
            st.pydeck_chart(deck(layers, targets))
            st.caption("검은 점 = 출동 관서 · 색깔 = 관서별 순찰 동선")

            names = [f"{r.attrs.get('depot', {}).get('name', '')} "
                     f"{int(r['회차'].iloc[0]) if '회차' in r else 1}회차"
                     for r in plan["routes"]]
            sel = st.selectbox("상세 동선", names)
            r = plan["routes"][names.index(sel)]
            cols = [c for c in ["순번", "grid_id", "emd", "sgg", "순찰점수",
                                "이동거리_m", "누적거리_m", "이동시간_분", "누적시간_분"]
                    if c in r.columns]
            st.dataframe(r[cols].rename(columns={"grid_id": "격자", "emd": "읍면동",
                                                 "sgg": "시군구"}),
                         hide_index=True, width='stretch')

        with st.expander(f"{mode.label} — 현장 중점 확인 항목", expanded=False):
            for chk in mode.checks:
                st.checkbox(chk, key=f"{mode.key}_{chk}")

    st.divider()
    st.markdown("### 월별 순찰 강도")
    fit = get_month_fit(city)
    wx = get_weather(city)
    if not fit:
        st.caption("화재 이력 자료가 없어 월별 분석을 표시할 수 없습니다.")
    else:
        plan_m = MO.monthly_plan(cur, "pred", fit, wx, year=year)
        st.session_state["month_plan"] = plan_m
        if not plan_m.empty:
            k1, k2, k3 = st.columns(3)
            hi = plan_m.loc[plan_m["위험계수"].idxmax()]
            lo = plan_m.loc[plan_m["위험계수"].idxmin()]
            k1.metric("가장 위험한 달", f"{hi['월']}", f"연평균 대비 {hi['위험계수']:.2f}배")
            k2.metric("가장 안전한 달", f"{lo['월']}", f"{lo['위험계수']:.2f}배",
                      delta_color="off")
            if fit.get("uses_weather"):
                k3.metric("기상 반영 설명력", f"{fit['weather_r2']:.2f}",
                          f"계절만 쓸 때 {fit['baseline_r2']:.2f}",
                          help="월별 화재 변동을 얼마나 설명하는가(R²).")
            st.bar_chart(plan_m.set_index("월")["위험계수"])
            show = [c for c in ["월", "위험계수", "등급", "예상화재_건", "humidity_mean",
                                "eh_mean", "dry_days", "wind_mean"] if c in plan_m.columns]
            st.dataframe(plan_m[show].rename(columns={
                "humidity_mean": "평균습도(%)", "eh_mean": "실효습도(%)",
                "dry_days": "건조일수", "wind_mean": "평균풍속(m/s)"}),
                hide_index=True, width='stretch')
            st.caption("위험계수 1.0 = 연평균 수준. 실효습도는 건조주의보 발표 기준값입니다.")

    st.divider()
    fires = get_fires(city)
    if not fires.empty:
        st.markdown("**화재 발생 시간대 분포**")
        cc1, cc2 = st.columns(2)
        cc1.bar_chart(P.hour_profile(fires).set_index("hour")["n"])
        cc2.bar_chart(P.weekday_profile(fires).set_index("요일")["n"])


# ================================================================== ④ 계획서
with tabs[3]:
    st.subheader("순찰·점검 계획서 생성")
    st.caption("동선·중점 확인사항·법령 근거가 들어간 결재용 문서를 만듭니다. "
               "숫자와 법령은 시스템이 확정하고, AI는 문장만 다듬습니다.")

    plan = st.session_state.get("patrol_plan")
    targets = st.session_state.get("patrol_targets")
    if plan is None or targets is None:
        st.info("먼저 **예방순찰 계획** 탭에서 순찰 조건을 설정하십시오.")
    else:
        g1, g2, g3 = st.columns([1, 1, 2])
        kind = g1.radio("계획 종류", ["일별", "월별", "연간"], horizontal=False)
        plan_date = g2.date_input("기준일", value=date.today())
        polish = g3.toggle("AI로 문체 다듬기", value=L.is_available(cfg),
                           disabled=not L.is_available(cfg),
                           help="숫자·동선·법령 조문은 바뀌지 않습니다.")

        st.markdown("**문서 정보** — 공문 머리·결재란에 들어갑니다")
        d1, d2, d3, d4 = st.columns(4)
        org = d1.text_input("기관명", value=f"{cfg.city(city)['label']}소방본부")
        dept = d2.text_input("부서", value="예방과")
        writer = d3.text_input("기안자", value="")
        tel = d4.text_input("연락처", value="")
        doc_meta = PLN.DocMeta(기관명=org, 부서=dept, 기안자=writer,
                               연락처=tel, 시행일=plan_date)

        mode = PM.MODES[st.session_state.get("patrol_mode_key", "general")]
        ctx = PLN.PlanContext(
            city_label=cfg.city(city)["label"], year=int(year), mode=mode,
            targets=targets, routes=plan["routes"], summary=plan["summary"],
            distance_source=plan["distance_source"],
            month_plan=st.session_state.get("month_plan", pd.DataFrame()),
            law_index=get_index(city, year), fires=get_fires(city))

        if st.button("계획서 생성", type="primary"):
            with st.spinner("계획서 작성 중…"):
                if kind == "일별":
                    doc = PLN.daily_plan(ctx, plan_date)
                elif kind == "월별":
                    doc = PLN.monthly_plan(ctx, plan_date.month)
                else:
                    doc = PLN.annual_plan(ctx)
                md = PLN.render(doc, ctx, doc_meta)
                if polish:
                    md = PLN.polish(cfg, md)["text"]
            st.session_state["last_doc"] = md

        md = st.session_state.get("last_doc")
        if md:
            st.download_button("계획서 내려받기 (Markdown)", md.encode("utf-8"),
                               file_name=f"순찰계획서_{kind}_{city}_{plan_date}.md")
            st.markdown("---")
            st.markdown(md)


# ================================================================== ⑤ 대응취약
with tabs[4]:
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


# ================================================================== ⑥ 검증
with tabs[5]:
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
        c2.metric("무작위 배정 대비", f"{h.get('model_lift',0):.2f}배",
                  help="같은 면적을 아무 데나 골랐을 때보다 몇 배 더 잡는가. "
                       "학계에서는 PAI(Predictive Accuracy Index)라 부릅니다.")
        pei = t.get("model", {}).get("pei", {}).get(key)
        if pei:
            c3.metric("달성 가능 최대치 대비", f"{pei:.1%}",
                      help="실제 화재를 다 알고 줄 세웠을 때의 성능을 100으로 봤을 때 "
                           "우리 모델이 어디쯤인지. 학계에서는 PEI"
                           "(Predictive Efficiency Index)라 부릅니다. "
                           "지역이 달라도 비교할 수 있는 지표입니다.")
        cal = t.get("calibration", {})
        if cal:
            c4.metric("예측 건수 정확도", f"{cal.get('total_ratio',0):.2f}",
                      f"예측 {cal.get('total_predicted',0):.0f} / 실제 {cal.get('total_actual',0):.0f}",
                      delta_color="off",
                      help="예측한 화재 총건수 ÷ 실제 총건수. 1.00 이면 건수까지 맞다는 뜻으로, "
                           "'이 격자는 연 3건 예상' 같은 말을 쓸 수 있습니다.")

        st.caption(f"검증 방식: {int(t.get('train_years',[0])[0])}–"
                   f"{int(t.get('train_years',[0])[-1])}년 "
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
                st.caption("점검 배분 비중 ÷ 화재 비중. 1.0 이면 위험한 만큼 점검이 갔다는 뜻입니다.")
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
                st.caption("PAI = 무작위 대비 배수 · PEI = 달성 가능 최대치 대비 비율")
                st.dataframe(pd.DataFrame(ev["resolution_scenarios"]),
                             hide_index=True, width='stretch')
                st.caption("화재의 약 63%는 도로명이 없어 읍면동 중심좌표로 배정됩니다. "
                           "도로명이 확보된 건만으로 별도 검증한 결과 성능 저하는 없었습니다.")

    with st.expander("용어 설명"):
        st.markdown("""
| 화면 표기 | 뜻 | 학술 용어 |
|---|---|---|
| **상위 20% 화재 포착률** | 위험 상위 20% 격자에 점검을 집중했을 때, 그해 실제 화재의 몇 %가 그 안에서 났는가 | capture rate |
| **무작위 배정 대비** | 같은 면적을 아무 데나 골랐을 때보다 몇 배 더 잡는가 | PAI (Predictive Accuracy Index) |
| **달성 가능 최대치 대비** | 실제 화재를 다 알고 줄 세운 '정답 순위'를 100으로 봤을 때 어디쯤인가. 화재가 원래 몇 군데에 몰린 지역은 어떤 모델이든 포착률이 높게 나오므로, 이 지표라야 지역 간 비교가 됩니다 | PEI (Predictive Efficiency Index) |
| **예측 건수 정확도** | 예측 총건수 ÷ 실제 총건수. 순위뿐 아니라 값도 맞는가 | calibration |
| **95% CI** | 같은 조사를 100번 다시 하면 95번은 이 범위 안에 들어온다는 뜻. 범위가 넓으면 표본이 작다는 신호입니다 | 신뢰구간 |
| **단순 기준** | 전년도에 화재가 많았던 격자 순으로 줄 세운 것. 학습 없이 누구나 할 수 있는 방법이라 비교 기준으로 씁니다 | baseline |
| **관할 제외 검증** | 구·군을 하나씩 통째로 빼고 학습해 그 지역을 맞히기. 한 지역만 외운 모델인지 가립니다 | LOGO (Leave-One-Group-Out) |
| **위험도 상승 요인** | 이 격자의 점수를 무엇이 얼마나 끌어올렸는가 | SHAP |
| **위험 등급** | 위험도 순으로 10등분한 것. 1등급이 가장 낮고 10등급이 가장 높습니다 | decile |
""")

# ================================================================== ⑦ 업무 도우미
with tabs[6]:
    st.subheader("화재예방 업무 도우미")
    st.caption("소방 법령, 업종별 점검 항목, 관할 위험 현황을 근거와 함께 찾아 드립니다. "
               "답변에는 반드시 근거 조문 또는 자료 출처가 함께 표시됩니다.")

    arts = get_law_articles()
    if arts.empty:
        st.warning("법령 자료를 불러오지 못했습니다. 네트워크를 확인하십시오. "
                   "업무규칙과 분석 결과만으로 답변합니다.")
    else:
        st.caption(f"수록 법령 {arts['law'].nunique()}종 · 조문 {len(arts):,}개 "
                   f"(국가법령정보센터) · 업무규칙 · {cfg.city(city)['label']} 분석 결과")

    examples = [
        "화재예방강화지구는 어떤 지역을 지정하고, 소방관서장은 무엇을 해야 하나요?",
        "노래연습장 점검할 때 중점적으로 봐야 할 항목은?",
        "다중이용업소 안전시설등 설치 기준이 어떻게 되나요?",
        "소방안전관리자를 선임해야 하는 대상물은?",
        f"{cur['center'].mode().iloc[0] if 'center' in cur and len(cur['center'].mode()) else '삼산119안전센터'} 관할에서 위험이 높은 곳은?",
    ]
    pick_ex = st.selectbox("예시 질문", ["(직접 입력)"] + examples)
    default_q = "" if pick_ex == "(직접 입력)" else pick_ex
    question = st.text_area("질문", value=default_q, height=80,
                            placeholder="예) 3급 대상물 자체점검 주기가 어떻게 되나요?")

    col_a, col_b = st.columns([1, 3])
    ask = col_a.button("질문하기", type="primary")
    top_k = col_b.slider("참고 자료 수", 3, 10, 5)

    if ask and question.strip():
        index = get_index(city, year)
        with st.spinner("자료를 찾고 답변을 작성 중…"):
            res = AS.answer(cfg, question.strip(), index, top_k=int(top_k))
        st.session_state["qa_result"] = res

    res = st.session_state.get("qa_result")
    if res:
        if res["source_backend"] == "search_only":
            st.info("AI 답변 생성이 연결되지 않아 관련 자료를 그대로 보여 드립니다.")
        elif res["source_backend"] == "no_match":
            st.warning("관련 자료를 찾지 못했습니다.")
        st.markdown("### 답변")
        st.markdown(res["answer"])

        if res["sources"]:
            st.markdown("### 근거 자료")
            for i, src in enumerate(res["sources"], 1):
                icon = {"법령": "📘", "업무규칙": "📋", "데이터": "📊"}.get(src["source"], "📄")
                with st.expander(f"{icon} ({i}) {src['title']}", expanded=(i <= 2)):
                    if src["ref"]:
                        st.caption(f"근거: {src['ref']}")
                    st.write(src["excerpt"])
        st.caption("법령 원문은 국가법령정보센터(law.go.kr)에서 확인하십시오. "
                   "본 답변은 업무 참고용이며 법적 효력을 갖지 않습니다.")


st.divider()
st.caption("본 시스템은 공개 데이터 기반 예측 결과이며, 법정 점검주기 및 관할 판단을 "
           "대체하지 않습니다. 격자 단위 집계로 개별 건물을 특정하지 않습니다.")
