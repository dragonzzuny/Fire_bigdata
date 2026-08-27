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
    hydrant as H, llm as L, model as M, operations as OP, patrol as P, \
    monthly as MO, patrol_modes as PM, routing as RT, rules as R  # noqa: E402
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
                "대응취약 구역", "모델 검증"])


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
    st.subheader("예방순찰 계획")

    c1, c2, c3 = st.columns([2, 1, 1])
    mode_key = c1.selectbox(
        "순찰 목적", list(PM.MODES),
        format_func=lambda k: PM.MODES[k].label,
        help="목적이 다르면 가야 할 곳도, 시간도, 볼 것도 다릅니다.")
    mode = PM.MODES[mode_key]
    n_teams = c2.number_input("동시 순찰 팀 수", 1, 12, 2,
                              help="팀 수만큼 구역을 나눠 서로 겹치지 않게 배분합니다.")
    n_grids = c3.number_input("순찰 격자 수", 4, 60, mode.default_k)

    st.caption(mode.purpose)

    r1, r2 = st.columns([3, 1])
    sgg_all = sorted(x for x in cur.get("sgg", pd.Series(dtype=str)).unique() if str(x).strip())
    pick_sgg = r1.multiselect("순찰 관할 (여러 개 선택 가능 · 비우면 전체)", sgg_all,
                              default=[sgg] if sgg != "전체" and sgg in sgg_all else [])
    use_road = r2.toggle("도로 기준 거리", value=True,
                         help="끄면 직선거리 × 우회계수로 계산합니다(빠르지만 부정확).")
    respect = r2.toggle("관할 경계 존중", value=False,
                        help="켜면 한 팀이 두 관할에 걸치지 않게 배정합니다.")

    targets = PM.select_targets(cur, mode, int(n_grids), sgg=pick_sgg or None)
    if targets.empty:
        st.info("선택한 관할에 순찰 대상 격자가 없습니다.")
    else:
        hours = PM.recommended_hours(mode, get_fires(city))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("순찰 대상", f"{len(targets):,}격자")
        m2.metric("권장 시간대", f"{hours[0]:02d}–{hours[1]:02d}시")
        m3.metric("순찰 팀", f"{int(n_teams)}개")

        with st.spinner("도로 경로 계산 중…" if use_road else "경로 계산 중…"):
            plan = RT.plan_patrol(targets, int(n_teams), use_road=use_road,
                                  weight_col="순찰점수", respect_groups=respect)
        m4.metric("최장 팀 이동", f"{plan['max_team_km']:.1f} km",
                  f"총 {plan['total_km']:.1f} km", delta_color="off")

        src = ("실제 도로 주행거리 (OSRM)" if plan["distance_source"] == "osrm"
               else "직선거리 × 우회계수 1.35 (도로망 서버 응답 없음)")
        st.caption(f"거리 기준: {src}")

        if respect and len(sgg_all) > int(n_teams):
            st.info(f"관할이 {len(sgg_all)}개인데 팀이 {int(n_teams)}개라 "
                    f"일부 팀은 여러 관할을 맡습니다. 팀을 늘리면 해소됩니다.")

        st.markdown("**팀별 순찰 구역**")
        st.dataframe(plan["summary"], hide_index=True, width='stretch')

        # 팀별 색으로 구분한 지도
        palette = [[227, 74, 51], [43, 108, 176], [47, 158, 110], [200, 120, 20],
                   [130, 70, 180], [20, 150, 160], [180, 60, 120], [90, 110, 40],
                   [230, 160, 40], [70, 70, 200], [160, 40, 40], [40, 160, 90]]
        layers = []
        for i, r in enumerate(plan["routes"]):
            col = palette[i % len(palette)]
            layers.append(pdk.Layer(
                "PathLayer", [{"path": r[["lon", "lat"]].astype(float).values.tolist()}],
                get_path="path", get_width=50, get_color=col, width_min_pixels=3))
            layers.append(pdk.Layer(
                "ScatterplotLayer", r.assign(팀=i + 1),
                get_position=["lon", "lat"], get_radius=190,
                get_fill_color=col + [200], pickable=True))
        st.pydeck_chart(deck(layers, targets))

        team_pick = st.selectbox("상세 경로", [f"{i+1}팀" for i in range(len(plan["routes"]))])
        r = plan["routes"][int(team_pick[0]) - 1] if plan["routes"] else pd.DataFrame()
        if not r.empty:
            cols = [c for c in ["순번", "grid_id", "sgg", "순찰점수", "이동거리_m",
                                "누적거리_m", "이동시간_분", "누적시간_분"] if c in r.columns]
            st.dataframe(r[cols].rename(columns={"grid_id": "격자", "sgg": "관할"}),
                         hide_index=True, width='stretch')
            st.download_button(
                f"{team_pick} 순찰 계획 (CSV)",
                r[cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"순찰_{mode.key}_{team_pick}_{city}_{year}.csv")

        with st.expander(f"{mode.label} — 현장 중점 확인 항목", expanded=True):
            for chk in mode.checks:
                st.checkbox(chk, key=f"{mode.key}_{chk}")

    st.divider()
    st.markdown("### 월별 순찰 강도 — 언제 더 돌아야 하는가")
    fit = get_month_fit(city)
    wx = get_weather(city)
    if not fit:
        st.caption("화재 이력 자료가 없어 월별 분석을 표시할 수 없습니다.")
    else:
        plan_m = MO.monthly_plan(cur, "pred", fit, wx, year=year)
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
                          help="월별 화재 변동을 얼마나 설명하는가(R²). "
                               "습도·건조일수를 넣으면 설명력이 올라갑니다.")
            else:
                k3.metric("기상 반영", "미채택",
                          help="기상을 넣어도 설명력이 늘지 않아 계절 패턴만 사용합니다.")

            st.bar_chart(plan_m.set_index("월")["위험계수"])
            show = [c for c in ["월", "위험계수", "등급", "예상화재_건",
                                "humidity_mean", "eh_mean", "dry_days", "wind_mean"]
                    if c in plan_m.columns]
            st.dataframe(plan_m[show].rename(columns={
                "humidity_mean": "평균습도(%)", "eh_mean": "실효습도(%)",
                "dry_days": "건조일수", "wind_mean": "평균풍속(m/s)"}),
                hide_index=True, width='stretch')
            st.caption("위험계수 1.0 = 연평균 수준. 실효습도는 건조주의보 발표 기준값으로, "
                       "여러 날의 습도를 누적해 계산합니다(기상청 공식, 감쇠계수 0.7). "
                       "격자 순위 × 월 위험계수 = 그 달 그 격자의 위험도.")

    st.divider()
    fires = get_fires(city)
    if fires.empty:
        st.caption("화재 이력 자료가 없어 시간대 분석을 표시할 수 없습니다.")
    else:
        with_time = int(fires["hour"].notna().sum()) if "hour" in fires else 0
        st.markdown("**화재 발생 시간대 분포**")
        st.caption(f"발생 시각이 기록된 화재 {with_time:,}건 기준 "
                   f"(2021년 자료는 시각이 누락되어 제외)")
        cc1, cc2 = st.columns(2)
        cc1.bar_chart(P.hour_profile(fires).set_index("hour")["n"])
        cc2.bar_chart(P.weekday_profile(fires).set_index("요일")["n"])


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

st.divider()
st.caption("본 시스템은 공개 데이터 기반 예측 결과이며, 법정 점검주기 및 관할 판단을 "
           "대체하지 않습니다. 격자 단위 집계로 개별 건물을 특정하지 않습니다.")
