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
import math
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

from firebird import dataset as D, evaluate as E, explain as X, forms as FM, \
    grid as G, \
    hydrant as H, llm as L, model as M, operations as OP, patrol as P, \
    monthly as MO, mapviz as MV, patrol_modes as PM, plans as PLN, routing as RT, rules as R, \
    assistant as AS, lawdata as LW, stations as ST  # noqa: E402
from firebird.config import load_config  # noqa: E402

# 브라우저 탭 아이콘도 로고로. 없으면 이모지로 물러난다.
_ICON = Path(__file__).resolve().parents[1] / "reports" / "brand" / "logo_mark.png"
st.set_page_config(page_title="불씨예보 K-Firebird",
                   page_icon=str(_ICON) if _ICON.exists() else "🔥",
                   layout="wide")

# 실행 중인 앱은 임포트한 모듈을 메모리에 물고 있다. 코드를 고쳐도 재기동하지 않으면
# 옛 모듈이 그대로 쓰여 'has no attribute' 같은 오류가 난다. 필요한 기능이
# 실제로 있는지 시작할 때 확인하고, 없으면 무엇을 해야 하는지 알려준다.
_REQUIRED = [
    (PLN, "DocMeta", "계획서 공문 서식"),
    (RT, "plan_from_stations", "관서 출발 순찰 동선"),
    (AS, "build_index", "업무 도우미 색인"),
    (ST, "station_table", "관서 위치"),
    (MO, "fit_month_risk", "월별 위험계수"),
]
_missing = [f"{name}({desc})" for mod, name, desc in _REQUIRED if not hasattr(mod, name)]
if _missing:
    st.error(
        "코드가 갱신되었으나 실행 중인 앱이 이전 버전을 사용하고 있습니다.\n\n"
        f"누락된 기능: {', '.join(_missing)}\n\n"
        "터미널에서 앱을 종료(Ctrl+C)한 뒤 다시 실행하십시오:\n"
        "`.venv/bin/streamlit run app/streamlit_app.py`")
    st.stop()

st.markdown("""
<style>
  :root { --ink:#22262d; --muted:#78808c; --line:#e6e9ec; --brand:#c0492f; }

  /* 본문 폭을 넓혀 표가 잘리지 않게 */
  .block-container { padding-top: 2.2rem; max-width: 1500px; }

  /* 탭을 눌러야 할 것처럼 보이게 */
  button[data-baseweb="tab"] { font-size: 0.97rem; font-weight: 600; }
  button[data-baseweb="tab"][aria-selected="true"] { color: var(--brand); }

  h1,h2,h3 { letter-spacing:-0.01em; }
  div[data-testid="stMetricValue"] { font-size: 1.55rem; }
  div[data-testid="stMetricLabel"] { color: var(--muted); }

  .callout { border-left:4px solid var(--brand); background:rgba(192,73,47,.06);
             padding:.75rem 1rem; border-radius:6px; margin:.5rem 0; }
  .good    { border-left-color:#2f8f6b; background:rgba(47,143,107,.07); }
  .info    { border-left-color:#41607f; background:rgba(65,96,127,.07); }

  /* 계획서 미리보기 — 문서처럼 보이게 */
  .docview { background:#fff; border:1px solid var(--line); border-radius:8px;
             padding:2.2rem 2.6rem; box-shadow:0 1px 3px rgba(0,0,0,.05);
             font-size:.94rem; line-height:1.75; }
  .docview table { width:100%; border-collapse:collapse; margin:.6rem 0 1rem; }
  .docview th,.docview td { border:1px solid var(--line); padding:.4rem .6rem;
                            font-size:.88rem; }
  .docview th { background:#f7f8f9; font-weight:600; }
  .docview h1 { font-size:1.5rem; text-align:center; margin:.2rem 0 1.2rem;
                padding-bottom:.6rem; border-bottom:2px solid var(--ink); }
  /* 공문은 줄이 붙어 있다. 문단 간격이 벌어지면 보고서처럼 읽힌다. */
  .docview p { margin:0 0 .18rem; }
  .docview h2 { font-size:1.02rem; margin:1rem 0 .35rem; }
  .docview img { max-width:100%; border:1px solid var(--line); border-radius:4px; }

  /* 업무 도우미 답변 */
  .answer { background:#fff; border:1px solid var(--line); border-left:4px solid var(--brand);
            border-radius:8px; padding:1.4rem 1.7rem; line-height:1.8; }
  .srcbox { background:#f7f8f9; border-radius:6px; padding:.7rem .9rem;
            font-size:.86rem; color:var(--muted); margin:.3rem 0; }
  .steps { display:flex; gap:.5rem; margin:.3rem 0 1rem; flex-wrap:wrap; }
  .step  { background:#f2f4f6; color:var(--muted); border-radius:999px;
           padding:.25rem .8rem; font-size:.82rem; }
  .step.on { background:var(--brand); color:#fff; font-weight:600; }
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


@st.cache_data(show_spinner="법령 별표 불러오는 중…")
def get_law_annexes() -> pd.DataFrame:
    try:
        return LW.collect_forms(get_config())
    except Exception:                                   # noqa: BLE001
        return pd.DataFrame()


@st.cache_resource(show_spinner="업무 자료 색인 중…")
def get_index(city: str, year: int):
    cfg = get_config()
    return AS.build_index(get_law_articles(), scored(city, year),
                          get_law_annexes(),
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


#: 지도 말풍선 — 층마다 들어 있는 열이 달라 따로 지정한다.
TIP_GRID = {"text": "격자 {grid_id} · {시군구}\n"
                    "위험점수 {위험점수} (순위 {순위})\n점검대상 {점검대상수}개소"}
TIP_PLAIN = {"text": "격자 {grid_id}"}


def deck(layers, df, zoom=None, tooltip=TIP_GRID):
    """데이터가 놓인 범위에 맞춰 화면을 잡는다.

    한 개 읍면동만 골랐는데 시 전체가 보이면 정작 봐야 할 동선이 점으로
    찍힌다. 고정 배율 대신 실제 좌표 범위에서 배율을 역산한다.
    """
    lat_s = pd.to_numeric(df["lat"], errors="coerce") if "lat" in df else pd.Series(dtype=float)
    lon_s = pd.to_numeric(df["lon"], errors="coerce") if "lon" in df else pd.Series(dtype=float)
    lat_s, lon_s = lat_s.dropna(), lon_s.dropna()
    if lat_s.empty or lon_s.empty:
        return pdk.Deck(layers=layers, map_style=None, tooltip=tooltip,
                        initial_view_state=pdk.ViewState(latitude=35.54, longitude=129.31,
                                                         zoom=10.2))
    lat, lon = float(lat_s.mean()), float(lon_s.mean())
    if zoom is None:
        # 위도 1도 ≈ 111km. 경도는 위도만큼 좁아진다.
        span_km = max(float(lat_s.max() - lat_s.min()) * 111.0,
                      float(lon_s.max() - lon_s.min()) * 111.0
                      * math.cos(math.radians(lat)),
                      0.8)
        span_km *= 1.12                       # 가장자리가 잘리지 않을 만큼만
        zoom = float(np.clip(math.log2(360.0 * 111.0 / span_km) - 1.2, 8.0, 14.0))
    return pdk.Deck(layers=layers, map_style=None, tooltip=tooltip,
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon,
                                                     zoom=zoom))


PALETTE = [[227, 74, 51], [43, 108, 176], [47, 158, 110], [200, 120, 20],
           [130, 70, 180], [20, 150, 160], [180, 60, 120], [90, 110, 40],
           [230, 160, 40], [70, 70, 200], [160, 40, 40], [40, 160, 90]]


def route_layers(routes, *, width=45, radius=180):
    """관서별 동선을 지도 층으로 만든다. 현재 계획과 직전 계획에 같이 쓴다."""
    layers, depots = [], []
    for i, r in enumerate(routes):
        col = PALETTE[i % len(PALETTE)]
        dep = r.attrs.get("depot", {})
        path = ([[dep.get("lon"), dep.get("lat")]] if dep else []) \
            + r[["lon", "lat"]].astype(float).values.tolist() \
            + ([[dep.get("lon"), dep.get("lat")]] if dep else [])
        layers.append(pdk.Layer("PathLayer", [{"path": path}], get_path="path",
                                get_width=width, get_color=col, width_min_pixels=3))
        layers.append(pdk.Layer("ScatterplotLayer", r, get_position=["lon", "lat"],
                                get_radius=radius, get_fill_color=col + [200],
                                pickable=True))
        if dep:
            depots.append({"lon": dep["lon"], "lat": dep["lat"],
                           "name": dep.get("name", "")})
    if depots:
        layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame(depots),
                                get_position=["lon", "lat"], get_radius=330,
                                get_fill_color=[20, 20, 20, 230], pickable=True))
    return layers, depots


def plan_facts(plan, targets) -> dict:
    """계획 하나를 몇 개의 숫자로 요약한다. 비교는 이 숫자들로 한다."""
    summ = plan.get("summary")
    return {
        "격자": int(len(targets)),
        "관서": int(len(summ)) if summ is not None and not summ.empty else 0,
        "회차": (int(summ["회차"].max()) if summ is not None and not summ.empty
                and "회차" in summ else 1),
        "총이동": float(plan.get("total_km", 0.0)),
        "최장": float(plan.get("max_team_km", 0.0)),
        "grids": set(targets["grid_id"].astype(str)) if "grid_id" in targets else set(),
    }


def _md_to_html(md: str) -> str:
    """계획서 미리보기용 최소 마크다운 변환.

    Streamlit 의 기본 렌더는 표 테두리가 없어 공문처럼 보이지 않는다.
    외부 라이브러리를 더하지 않고 필요한 만큼만 직접 변환한다.
    """
    import html as _html
    import re as _re

    out, in_table = [], False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(_re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue                       # 구분선
            if not in_table:
                out.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            row = "".join(f"<{tag}>{_fmt_inline(c)}</{tag}>" for c in cells)
            out.append(f"<tr>{row}</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if not line.strip():
            continue
        m = _re.match(r"!\[[^\]]*\]\((.+?)\)", line.strip())
        if m:
            out.append(f'<img src="{_html.escape(m.group(1))}">')
            continue
        if line.startswith("### "):
            out.append(f"<h3>{_fmt_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_fmt_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_fmt_inline(line[2:])}</h1>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip().startswith("> "):
            out.append(f"<blockquote>{_fmt_inline(line.strip()[2:])}</blockquote>")
        else:
            out.append(f"<p>{_fmt_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


def _fmt_inline(text: str) -> str:
    import html as _html
    import re as _re
    t = _html.escape(str(text))
    t = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t


def _doc_html(md: str, map_path: str = "") -> str:
    """인쇄용 HTML. 브라우저에서 열어 그대로 A4 로 인쇄한다."""
    import base64
    body = _md_to_html(md)
    if map_path:
        try:
            with open(map_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            body = body.replace(f'<img src="{map_path}">',
                                f'<img src="data:image/png;base64,{b64}">')
        except OSError:
            pass
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>순찰계획서</title><style>
@page {{ size: A4; margin: 18mm 16mm; }}
body {{ font-family:'Malgun Gothic','Noto Sans KR',sans-serif; color:#1a1d23;
        line-height:1.6; font-size:10.5pt; }}
/* 공문은 줄이 붙어 있다. 문단마다 간격이 벌어지면 보고서처럼 보인다. */
p {{ margin:0 0 3pt; }}
p + p {{ margin-top:0; }}
h1 {{ font-size:16pt; text-align:center; margin:0 0 16pt;
      padding-bottom:8pt; border-bottom:2px solid #1a1d23; }}
h2 {{ font-size:12pt; margin:14pt 0 5pt; }}
table {{ width:100%; border-collapse:collapse; margin:6pt 0 12pt; }}
th,td {{ border:1px solid #c8ccd2; padding:4pt 6pt; font-size:9.5pt; }}
th {{ background:#f2f4f6; }}
img {{ max-width:100%; margin:8pt 0; }}
blockquote {{ margin:4pt 0 8pt 12pt; color:#555; font-size:9.5pt; }}
hr {{ border:0; border-top:1px solid #c8ccd2; margin:14pt 0; }}
</style></head><body>{body}</body></html>"""


# ------------------------------------------------------------------ 사이드바

cfg = get_config()

BRAND = Path(__file__).resolve().parents[1] / "reports" / "brand"
_lockup = BRAND / "logo_lockup.png"
if _lockup.exists():
    st.sidebar.image(str(_lockup), width='stretch')
else:
    st.sidebar.markdown("## 불씨예보")
    st.sidebar.caption("K-Firebird · 화재예방 점검·순찰 의사결정 시스템")
st.sidebar.divider()

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
    st.caption("구역마다 점검 대상물 수가 다릅니다. 위험도 순으로만 자르면 "
               "인력으로 감당할 수 없는 계획이 나오므로, 가용 물량 안에서 배분합니다.")

    cmp = OP.compare_to_topk(view, view["pred"], capacity, cfg.headline_k)
    alloc, eq_info = OP.allocate_with_equity(view, view["pred"], capacity,
                                             min_share=equity_share)
    o, t = cmp["optimized"], cmp["top_k_percent"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("점검 가능 물량", f"{capacity.total_visits:,}건",
              help=capacity.describe())
    m2.metric("배분 결과", f"{o['n_grids']:,}개 격자",
              f"{o['cost_used']:,.0f}건 배정")
    m3.metric(f"상위 {cfg.headline_k}% 그대로 갈 때",
              f"{t['n_grids_affordable']:,} / {t['n_grids_selected']:,}개 격자",
              f"필요 {t['cost_if_all']:,.0f}건 · 가용 {capacity.total_visits:,}건",
              delta_color="off",
              help="위험도 높은 순서대로 격자를 통째로 점검해 나갈 때, "
                   "가용 물량으로 끝까지 마칠 수 있는 격자 수")
    if "gain_pp" in cmp:
        m4.metric("실제 화재 포착률", f"{o['actual_capture_rate']:.1%}",
                  f"{cmp['gain_pp']:+.1f}%p")

    if t["n_grids_affordable"] < t["n_grids_selected"]:
        st.markdown(
            f"<div class='callout'><b>위험도 상위 {cfg.headline_k}% 안의 점검 대상은 "
            f"{t['cost_if_all']:,.0f}개소입니다.</b><br>"
            + (f"가용 물량 {capacity.total_visits:,}건으로는 첫 번째 격자 하나도 "
               f"끝내지 못합니다. 위험한 순서대로 줄을 세우는 것만으로는 "
               f"계획이 되지 않습니다."
               if t["n_grids_affordable"] == 0 else
               f"현재 가용 물량 {capacity.total_visits:,}건으로는 "
               f"{t['n_grids_selected']:,}개 격자 중 "
               f"{t['n_grids_affordable']:,}개까지만 점검할 수 있습니다.")
            + "</div>", unsafe_allow_html=True)
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
        cols = [c for c in ["점검순서", "grid_id", "emd", "sgg", "위험점수",
                            "점검대상수", "expected_fires", "누적비용"]
                if c in alloc.columns]
        rank_tb = alloc[cols].rename(columns={
            "grid_id": "격자", "emd": "읍면동", "sgg": "관할",
            "expected_fires": "기대화재", "누적비용": "누적건수"})
        # 소수점 네 자리는 현장에서 아무 의미가 없다.
        if "위험점수" in rank_tb:
            rank_tb["위험점수"] = rank_tb["위험점수"].astype(float).round(1)
        if "기대화재" in rank_tb:
            rank_tb["기대화재"] = rank_tb["기대화재"].astype(float).round(2)
        for c in ("점검대상수", "누적건수"):
            if c in rank_tb:
                rank_tb[c] = pd.to_numeric(rank_tb[c], errors="coerce").fillna(0).astype(int)
        st.dataframe(rank_tb, hide_index=True, height=430, width='stretch')
    if eq_info.get("equity_constrained"):
        rows = [{"관할": g,
                 "화재 비중": round(v["risk_share"], 3),
                 "배분 비중": round(v["budget_share"], 3),
                 "비율": round(v["budget_share"] / v["risk_share"], 2)}
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
    st.subheader("구역별 위험요인 및 점검계획서")
    st.caption("이 구역의 위험을 끌어올린 요인과, 현장에서 확인할 항목을 "
               "함께 보여 드립니다.")

    if alloc.empty:
        st.markdown("<div class='callout info'>배분된 구역이 없습니다. "
                    "‘예방점검 배분’ 탭에서 인력을 설정하십시오.</div>",
                    unsafe_allow_html=True)
    else:
        labels = {f"{r['점검순서']}순위 · {r['grid_id']} · "
                  f"{r.get('emd') or r.get('sgg', '')} "
                  f"(위험 {r['위험점수']:.0f} · 대상 {int(r['점검대상수'])}개소)": r["grid_id"]
                  for _, r in alloc.head(60).iterrows()}
        pick = st.selectbox("점검 대상 구역", list(labels))
        gid = labels[pick]
        row = cur[cur["grid_id"] == gid].iloc[0]

        # --- 구역 한눈에 ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("위험점수", f"{row['위험점수']:.0f}",
                  f"관내 {int(row['순위'])}위 (상위 {row['상위%']:.1f}%)")
        k2.metric("점검 대상물", f"{int(row.get('점검대상수', 0)):,}개소")
        if "biz_total" in row:
            k3.metric("다중이용업소", f"{int(row['biz_total']):,}개소")
        k4.metric("전년 화재", f"{int(row.get('fires_lag1', 0))}건",
                  f"누적 {int(row.get('fires_cum', 0))}건", delta_color="off")
        hyd = int(row.get("n_hydrant", 0) or 0)
        dist = row.get("dist_hydrant_m", float("nan"))
        k5.metric("소화전", f"{hyd}개",
                  ("없음, 인접 수리 확인 필요" if hyd == 0
                   else (f"최근접 {dist:,.0f} m" if dist == dist and dist < 1e6 else "")),
                  delta_color="off")

        loc = " · ".join(str(row.get(c, "")) for c in ("station", "center", "emd")
                         if str(row.get(c, "")).strip())
        if loc:
            st.caption(f"관할: {loc}")

        st.divider()
        left, right = st.columns([1, 1])

        with left:
            st.markdown("##### 위험도를 끌어올린 요인")
            model = get_model(city)
            drivers = []
            if model is None:
                st.markdown("<div class='callout info'>학습된 모델이 없어 "
                            "요인 분석을 표시할 수 없습니다.</div>",
                            unsafe_allow_html=True)
            else:
                try:
                    d = X.explain_grids(model, cur[cur["grid_id"] == gid])
                    drivers = d["drivers"].iloc[0] if len(d) else []
                except Exception as exc:                     # noqa: BLE001
                    st.warning(f"요인 계산 실패: {exc}")
                if drivers:
                    dd = pd.DataFrame(drivers)
                    st.bar_chart(dd.set_index("label")["contribution"],
                                 color="#c0492f", height=210)
                    st.dataframe(
                        dd[["label", "value", "contribution"]].rename(columns={
                            "label": "요인", "value": "현재값", "contribution": "기여도"}),
                        hide_index=True, width='stretch')
                    st.caption("기여도가 클수록 이 구역의 위험을 많이 끌어올린 요인입니다.")
                else:
                    st.caption("표시할 요인이 없습니다.")

        with right:
            st.markdown("##### 현장 점검 항목")
            checklist = R.checklist_for_grid(row, cfg)
            done_key = f"chk_{gid}"
            st.caption(f"이 구역의 업종·소방시설 구성에 맞춰 {checklist['n_items']}개 항목이 "
                       "자동으로 구성됩니다.")
            for i, sec in enumerate(checklist["sections"]):
                with st.expander(f"{sec['구분']} — {sec['근거']}", expanded=(i == 0)):
                    for item in sec["항목"]:
                        st.checkbox(item, key=f"{done_key}_{item}")

        st.divider()
        st.markdown("##### 점검계획서 초안")
        c1, c2 = st.columns([1, 3])
        make = c1.button("계획서 작성", type="primary", width='stretch')
        c2.caption("AI 문서 작성 " + ("사용 가능 · 20~40초 소요"
                                   if L.is_available(cfg)
                                   else "미연결(표준 서식으로 작성됩니다)"))
        if make:
            risk = {"score_0_100": float(row["위험점수"]), "rank": int(row["순위"]),
                    "percentile": float(row["상위%"])}
            with st.spinner("작성 중입니다…"):
                doc = L.inspection_plan(cfg, str(gid), risk, drivers, checklist)
            st.session_state["insp_doc"] = doc

        doc = st.session_state.get("insp_doc")
        if doc and doc.get("grid_id") == str(gid):
            st.download_button("계획서 내려받기", doc["text"].encode("utf-8"),
                               file_name=f"점검계획서_{gid}.md")
            st.markdown(f"<div class='docview'>{_md_to_html(doc['text'])}</div>",
                        unsafe_allow_html=True)


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
        # --- 조건이 바뀌면 무엇이 달라지는지 남겨 둔다 -------------------
        cond = {"목적": mode.label,
                "출동 단위": "119안전센터" if level == "center" else "소방서",
                "순찰 격자 수": int(n_grids),
                "1회 순찰 시간": f"{int(budget)}분" if budget else "제한 없음",
                "지역": ", ".join(pick) if pick else "관내 전체",
                "거리 기준": "도로" if use_road else "직선"}
        facts = plan_facts(plan, targets)
        prev = st.session_state.get("patrol_prev")
        changed = bool(prev) and prev["cond"] != cond
        st.session_state["patrol_prev"] = {"cond": cond, "facts": facts,
                                           "routes": plan["routes"],
                                           "targets": targets} \
            if changed or not prev else prev
        if changed:
            st.session_state["patrol_before"] = prev

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
            summ = plan["summary"].rename(columns={
                "관서→첫격자_km": "관서→첫 구역(km)", "순찰거리_km": "순찰 거리(km)",
                "복귀거리_km": "복귀 거리(km)", "총_km": "총 거리(km)",
                "총_분": "소요 시간(분)", "격자수": "구역 수"})
            if "소요 시간(분)" in summ:
                summ["소요 시간(분)"] = pd.to_numeric(
                    summ["소요 시간(분)"], errors="coerce").fillna(0).astype(int)

            layers, depots = route_layers(plan["routes"])
            # 관서가 대상 구역 밖에 있을 수 있으므로 함께 넣어 범위를 잡는다.
            extent = pd.concat(
                [targets[["lon", "lat"]]] +
                ([pd.DataFrame(depots)[["lon", "lat"]]] if depots else []),
                ignore_index=True)
            # 동선 그림이 이 화면의 결과물이다. 표보다 위에 둔다.
            mp, tb = st.columns([3, 2])
            with mp:
                st.pydeck_chart(deck(layers, extent, tooltip=TIP_PLAIN))
                st.caption("검은 점 = 출동 관서 · 색깔 = 관서별 순찰 동선 "
                           "· 관서에서 출발해 관서로 돌아옵니다.")
            with tb:
                st.markdown("**관서별 순찰 구역**")
                # 좁은 칸에 아홉 열을 넣으면 다 잘린다. 요약만 두고 나머지는 접는다.
                brief = [c for c in ["출동관서", "구역 수", "총 거리(km)", "소요 시간(분)"]
                         if c in summ.columns]
                st.dataframe(summ[brief], hide_index=True, width='stretch', height=380)
                st.caption(f"{len(summ)}개 조가 동시에 나갑니다. "
                           "구역이 겹치지 않도록 나눈 결과입니다.")
                with st.expander("구간별 거리 자세히"):
                    st.dataframe(summ, hide_index=True, width='stretch')

            # ---------------- 조건을 바꾸면 계획이 어떻게 달라지는가 ----------
            before = st.session_state.get("patrol_before")
            if before:
                st.divider()
                bf, bc = before["facts"], before["cond"]
                diff_keys = [k for k in cond if bc.get(k) != cond[k]]
                st.markdown("#### 조건을 바꾸기 전과 후")
                st.markdown(
                    "<div class='callout'>바꾼 조건: <b>"
                    + "</b>, <b>".join(
                        f"{k} {bc.get(k)} → {cond[k]}" for k in diff_keys)
                    + "</b></div>", unsafe_allow_html=True)

                same = len(bf["grids"] & facts["grids"])
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("순찰 격자", f"{facts['격자']}개",
                          f"{facts['격자'] - bf['격자']:+d}개", delta_color="off")
                d2.metric("총 이동", f"{facts['총이동']:.1f} km",
                          f"{facts['총이동'] - bf['총이동']:+.1f} km",
                          delta_color="inverse")
                d3.metric("가장 먼 순찰조", f"{facts['최장']:.1f} km",
                          f"{facts['최장'] - bf['최장']:+.1f} km",
                          delta_color="inverse")
                d4.metric("겹치는 구역", f"{same}개",
                          f"이전 {bf['격자']}개 중", delta_color="off")

                bl, bd = route_layers(before["routes"], width=38, radius=150)
                bext = pd.concat(
                    [before["targets"][["lon", "lat"]]] +
                    ([pd.DataFrame(bd)[["lon", "lat"]]] if bd else []),
                    ignore_index=True)
                # 두 지도의 화면 범위가 다르면 눈으로 비교가 안 된다. 같은 틀에 놓는다.
                both = pd.concat([bext, extent], ignore_index=True)

                b1, b2 = st.columns(2)
                with b1:
                    st.markdown(f"**바꾸기 전: {bc['목적']}**")
                    st.pydeck_chart(deck(bl, both, tooltip=TIP_PLAIN))
                    st.caption(f"{bf['격자']}격자 · {bf['관서']}개 관서 · "
                               f"총 {bf['총이동']:.1f} km")
                with b2:
                    st.markdown(f"**바꾼 뒤: {cond['목적']}**")
                    st.pydeck_chart(deck(layers, both, tooltip=TIP_PLAIN))
                    st.caption(f"{facts['격자']}격자 · {facts['관서']}개 관서 · "
                               f"총 {facts['총이동']:.1f} km")
                st.caption("두 지도는 같은 범위·같은 배율입니다.")

                gone = sorted(bf["grids"] - facts["grids"])
                new_g = sorted(facts["grids"] - bf["grids"])
                cc1, cc2 = st.columns(2)
                cc1.markdown(f"**빠진 구역 {len(gone)}개**  \n"
                             + (", ".join(gone[:12]) + (" …" if len(gone) > 12 else "")
                                if gone else "없음"))
                cc2.markdown(f"**새로 들어온 구역 {len(new_g)}개**  \n"
                             + (", ".join(new_g[:12]) + (" …" if len(new_g) > 12 else "")
                                if new_g else "없음"))
                if st.button("비교 지우기"):
                    st.session_state.pop("patrol_before", None)
                    st.rerun()
                st.divider()

            with st.expander("계획서용 지도 (인쇄·첨부용)", expanded=False):
                st.caption("화재위험 분포 위에 순찰 동선을 얹은 그림입니다. "
                           "계획서에 그대로 붙습니다.")
                if st.button("지도 만들기"):
                    with st.spinner("지도 생성 중…"):
                        fig = MV.route_map(
                            cur, plan["routes"], cfg,
                            title=f"{cfg.city(city)['label']} {mode.label} — "
                                  f"화재위험 및 순찰 동선",
                            top_n_outline=40)
                        path = cfg.paths.figures / f"map_{city}_{year}_{mode.key}.png"
                        MV.save(fig, path)
                        st.session_state["map_path"] = str(path)
                if st.session_state.get("map_path"):
                    st.image(st.session_state["map_path"], width='stretch')
                    with open(st.session_state["map_path"], "rb") as fh:
                        st.download_button("지도 내려받기 (PNG)", fh.read(),
                                           file_name=f"순찰지도_{city}_{year}.png",
                                           mime="image/png")

            names = [f"{r.attrs.get('depot', {}).get('name', '')} "
                     f"{int(r['회차'].iloc[0]) if '회차' in r else 1}회차"
                     for r in plan["routes"]]
            sel = st.selectbox("상세 동선", names)
            r = plan["routes"][names.index(sel)]
            cols = [c for c in ["순번", "grid_id", "emd", "sgg", "순찰점수",
                                "이동거리_m", "누적거리_m", "이동시간_분", "누적시간_분"]
                    if c in r.columns]
            det = r[cols].rename(columns={
                "grid_id": "격자", "emd": "읍면동", "sgg": "시군구",
                "이동거리_m": "직전 지점에서(m)", "누적거리_m": "누적 거리(m)",
                "이동시간_분": "이동(분)", "누적시간_분": "누적 시간(분)"})
            if "순찰점수" in det:
                det["순찰점수"] = det["순찰점수"].astype(float).round(1)
            for c in ("직전 지점에서(m)", "누적 거리(m)", "이동(분)", "누적 시간(분)"):
                if c in det:
                    det[c] = pd.to_numeric(det[c], errors="coerce").fillna(0).astype(int)
            st.dataframe(det, hide_index=True, width='stretch')
            st.caption("순번대로 이동합니다. 관서에서 출발해 마지막 구역을 돌고 "
                       "관서로 복귀하는 시간까지 포함한 계획입니다.")

        with st.expander(f"{mode.label} 현장 중점 확인 항목", expanded=False):
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
            mtb = plan_m[show].rename(columns={
                "humidity_mean": "평균습도(%)", "eh_mean": "실효습도(%)",
                "dry_days": "건조일수", "wind_mean": "평균풍속(m/s)",
                "예상화재_건": "예상 화재(건)"})
            for c, nd in (("위험계수", 2), ("평균습도(%)", 1), ("실효습도(%)", 1),
                          ("평균풍속(m/s)", 1), ("예상 화재(건)", 1)):
                if c in mtb:
                    mtb[c] = pd.to_numeric(mtb[c], errors="coerce").round(nd)
            if "건조일수" in mtb:
                mtb["건조일수"] = pd.to_numeric(
                    mtb["건조일수"], errors="coerce").fillna(0).astype(int)
            st.dataframe(mtb, hide_index=True, width='stretch')
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
    st.subheader("순찰·점검 계획서")
    st.caption("동선·중점 확인사항·법령 근거가 들어간 결재용 공문을 만듭니다. "
               "숫자와 법령은 시스템이 확정하고, AI는 문장만 다듬습니다.")

    plan = st.session_state.get("patrol_plan")
    targets = st.session_state.get("patrol_targets")
    has_plan = plan is not None and targets is not None
    has_map = bool(st.session_state.get("map_path"))
    has_doc = bool(st.session_state.get("last_doc"))

    st.markdown(
        f"<div class='steps'>"
        f"<span class='step {'on' if has_plan else ''}'>1 순찰 조건</span>"
        f"<span class='step {'on' if has_map else ''}'>2 동선도</span>"
        f"<span class='step {'on' if has_doc else ''}'>3 계획서</span>"
        f"</div>", unsafe_allow_html=True)

    if not has_plan:
        st.markdown("<div class='callout info'><b>먼저 순찰 조건을 정하십시오.</b><br>"
                    "‘예방순찰 계획’ 탭에서 순찰 목적·지역·팀을 설정하면 "
                    "이 화면에서 계획서를 만들 수 있습니다.</div>",
                    unsafe_allow_html=True)
    else:
        mode = PM.MODES[st.session_state.get("patrol_mode_key", "general")]
        n_grid = sum(len(r) for r in plan["routes"])
        n_st = len({r.attrs.get("depot", {}).get("name", "") for r in plan["routes"]})
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("순찰 종류", mode.label)
        i2.metric("출동 관서", f"{n_st}개")
        i3.metric("순찰 구역", f"{n_grid}개")
        i4.metric("총 이동", f"{plan['total_km']:.1f} km")

        st.divider()
        left, right = st.columns([1, 1])

        with left:
            st.markdown("##### 1. 문서 정보")
            org = st.text_input("기관명", value=f"{cfg.city(city)['label']}소방본부")
            c1, c2 = st.columns(2)
            dept = c1.text_input("부서", value="예방과")
            writer = c2.text_input("기안자", value="", placeholder="예) 소방교 홍길동")
            c3, c4 = st.columns(2)
            tel = c3.text_input("연락처", value="", placeholder="052-000-0000")
            docno = c4.text_input("문서번호", value="", placeholder="예방과-1234")

            st.markdown("##### 2. 계획 종류")
            kind = st.radio("계획 종류", ["일별", "월별", "연간"], horizontal=True,
                            label_visibility="collapsed")
            plan_date = st.date_input("기준일", value=date.today())
            doc_meta = PLN.DocMeta(기관명=org, 부서=dept, 기안자=writer, 연락처=tel,
                                   문서번호=docno, 시행일=plan_date)

        with right:
            st.markdown("##### 3. 동선도 첨부")
            if has_map:
                st.image(st.session_state["map_path"], width='stretch')
                st.caption("‘예방순찰 계획’ 탭에서 다시 만들 수 있습니다.")
            else:
                st.markdown("<div class='callout info'>동선도가 없습니다. "
                            "지금 만들면 계획서에 함께 들어갑니다.</div>",
                            unsafe_allow_html=True)
                if st.button("동선도 만들기"):
                    with st.spinner("지도 생성 중…"):
                        fig = MV.route_map(
                            cur, plan["routes"], cfg,
                            title=f"{cfg.city(city)['label']} {mode.label} — "
                                  f"화재위험 및 순찰 동선", top_n_outline=40)
                        path = cfg.paths.figures / f"map_{city}_{year}_{mode.key}.png"
                        MV.save(fig, path)
                        st.session_state["map_path"] = str(path)
                    st.rerun()

            st.markdown("##### 4. 생성")
            polish = st.toggle("AI로 문체 다듬기", value=L.is_available(cfg),
                               disabled=not L.is_available(cfg),
                               help="숫자·동선·법령 조문은 바뀌지 않습니다. "
                                    "30~60초 걸립니다.")
            go = st.button("계획서 생성", type="primary", width='stretch')

        if go:
            ctx = PLN.PlanContext(
                city_label=cfg.city(city)["label"], year=int(year), mode=mode,
                targets=targets, routes=plan["routes"], summary=plan["summary"],
                distance_source=plan["distance_source"],
                month_plan=st.session_state.get("month_plan", pd.DataFrame()),
                law_index=get_index(city, year), fires=get_fires(city),
                map_path=st.session_state.get("map_path", ""))
            with st.spinner("계획서 작성 중…"):
                if kind == "일별":
                    doc = PLN.daily_plan(ctx, plan_date)
                elif kind == "월별":
                    doc = PLN.monthly_plan(ctx, plan_date.month)
                else:
                    doc = PLN.annual_plan(ctx)
                md = PLN.render(doc, ctx, doc_meta)
                if polish:
                    res = PLN.polish(cfg, md)
                    md = res["text"]
                    if not res.get("polished"):
                        st.info("AI 다듬기를 적용하지 않았습니다. 표준 서식 그대로 출력합니다.")
            st.session_state["last_doc"] = md
            st.session_state["last_doc_kind"] = kind
            st.session_state["last_doc_legal"] = doc.get("legal", [])

        md = st.session_state.get("last_doc")
        if md:
            st.divider()
            d1, d2, d3 = st.columns([1, 1, 3])
            d1.download_button("Markdown 내려받기", md.encode("utf-8"),
                               file_name=f"순찰계획서_{st.session_state.get('last_doc_kind','')}"
                                         f"_{city}_{plan_date}.md", width='stretch')
            html_doc = _doc_html(md, st.session_state.get("map_path", ""))
            d2.download_button("인쇄용 HTML", html_doc.encode("utf-8"),
                               file_name=f"순찰계획서_{city}_{plan_date}.html",
                               mime="text/html", width='stretch')
            d3.caption("HTML을 열어 브라우저에서 인쇄하면 A4 문서로 출력됩니다.")

            legal = st.session_state.get("last_doc_legal") or []
            if legal:
                with st.expander(f"인용된 법령 조문 {len(legal)}건", expanded=False):
                    for l in legal:
                        st.markdown(f"**{l['ref']}**")
                        st.markdown(f"<div class='srcbox'>{l['excerpt']}</div>",
                                    unsafe_allow_html=True)

            st.markdown("##### 미리보기")
            st.markdown(f"<div class='docview'>{_md_to_html(md)}</div>",
                        unsafe_allow_html=True)

    # ---------------- 법정 서식 채우기 --------------------------------
    st.divider()
    st.markdown("### 법정 서식으로 내보내기")
    st.caption("자체 계획서와 별개로, 법에 서식이 정해진 문서는 그 서식을 써야 "
               "결재가 됩니다. 왼쪽이 법제처 원본 서식, 오른쪽이 같은 서식을 "
               "우리 데이터로 채운 것입니다.")

    blank_form = cfg.paths.processed / "forms" / "서식11_빈양식.png"
    if alloc.empty:
        st.markdown("<div class='callout info'>‘예방점검 배분’ 탭에서 인력을 "
                    "설정하면 대상 구역이 정해집니다.</div>", unsafe_allow_html=True)
    else:
        fl, fr = st.columns([2, 1])
        gid_opts = {f"{r['점검순서']}순위 · {r['grid_id']} · "
                    f"{r.get('emd') or r.get('sgg', '')}": r["grid_id"]
                    for _, r in alloc.head(40).iterrows()}
        pick_g = fl.selectbox("대장을 작성할 구역", list(gid_opts), key="form_grid")
        make = fr.button("서식 채우기", type="primary", width='stretch')

        if make or st.session_state.get("ledger_html"):
            if make:
                grow = cur[cur["grid_id"] == gid_opts[pick_g]].iloc[0]
                st_all = pd.concat(
                    [get_stations(city, year, "station"),
                     get_stations(city, year, "center")], ignore_index=True)
                drv = []
                mdl = get_model(city)
                if mdl is not None:
                    try:
                        d = X.explain_grids(mdl, cur[cur["grid_id"] == gid_opts[pick_g]])
                        drv = list(d["drivers"].iloc[0]) if len(d) else []
                    except Exception:                       # noqa: BLE001
                        drv = []
                led = FM.zone_ledger(grow, city_label=cfg.city(city)["label"],
                                     year=int(year), stations=st_all,
                                     drivers=drv, grid_m=int(cfg.grid_size_m))
                st.session_state["ledger_html"] = FM.render_ledger_html(
                    led, city_label=cfg.city(city)["label"], year=int(year))
                st.session_state["ledger_stat"] = (led["n_filled"], led["n_fields"],
                                                   led["grid_id"])

            n_f, n_t, g_id = st.session_state.get("ledger_stat", (0, 0, ""))
            st.markdown(
                f"<div class='callout good'><b>{FM.FORM_NO} {FM.FORM_TITLE}</b> "
                f"— {n_t}개 칸 중 <b>{n_f}개</b>를 공개 데이터로 채웠습니다. "
                f"나머지는 연계 자료가 없어 비워 두고, 칸마다 사유를 적었습니다. "
                f"빈칸을 그럴듯한 값으로 메우면 결재 문서가 아니라 추정치가 "
                f"됩니다.</div>", unsafe_allow_html=True)

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**법제처 원본 서식 (빈 양식)**")
                if blank_form.exists():
                    st.image(str(blank_form), width='stretch')
                    st.caption(f"「{FM.FORM_LAW}」 {FM.FORM_NO} · "
                               "`scripts/09_forms.py` 로 법제처에서 직접 내려받습니다.")
                else:
                    st.markdown("<div class='callout info'>`python scripts/09_forms.py` "
                                "를 실행하면 원본 서식이 표시됩니다.</div>",
                                unsafe_allow_html=True)
            with g2:
                st.markdown(f"**불씨예보가 채운 대장: {g_id}**")
                st.markdown(
                    f"<div style='border:1px solid #e5e5e5;padding:14px;"
                    f"background:#fff;max-height:1180px;overflow:auto'>"
                    f"{st.session_state['ledger_html']}</div>",
                    unsafe_allow_html=True)
                st.download_button(
                    "채운 대장 내려받기 (HTML)",
                    ("<html><head><meta charset='utf-8'><title>"
                     f"{FM.FORM_TITLE}</title></head><body style='width:900px;"
                     "margin:20px auto;background:#fff'>"
                     + st.session_state["ledger_html"] + "</body></html>").encode("utf-8"),
                    file_name=f"화재예방강화지구_관리대장_{g_id}.html",
                    mime="text/html")


# ================================================================== ⑤ 대응취약
with tabs[4]:
    st.subheader("소방용수 사각지대 및 화재 급증 구역")
    st.caption("소화전이 없는 구역은 전체의 절반에 가깝습니다. 산지에도 소화전은 "
               "없기 때문입니다. **위험 상위 구간과 교차한 구역**만 추려야 "
               "신설 우선순위가 됩니다.")

    cov = H.hydrant_coverage(cur)
    if not cov.get("available"):
        st.markdown("<div class='callout info'>소방용수시설 자료가 없습니다.</div>",
                    unsafe_allow_html=True)
    else:
        blind = H.blind_spots(cur, cur["pred"], cfg)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 구역", f"{cov['n_grids']:,}개")
        c2.metric("소화전 미설치", f"{cov['grids_without_hydrant']:,}개",
                  f"{cov['share_without_hydrant']:.1%}", delta_color="off")
        c3.metric("고위험 · 용수 사각", f"{len(blind)}개",
                  help=f"위험 상위 {cfg['hydrant']['high_risk_percentile']}% 중 "
                       f"소화전이 없거나 {cfg['hydrant']['max_dist_m']}m 밖인 구역")
        if len(blind):
            c4.metric("사각 구역 실제 화재",
                      f"{int(blind['fires'].sum()) if 'fires' in blind else 0}건",
                      delta_color="off")

        if not blind.empty:
            st.markdown(
                f"<div class='callout'><b>소화전 신설 우선순위 {len(blind)}개 구역</b><br>"
                f"위험은 상위 {cfg['hydrant']['high_risk_percentile']}%인데 "
                f"소화전이 없거나 {cfg['hydrant']['max_dist_m']}m 밖입니다. "
                f"예산 요구 시 객관적 근거로 씁니다.</div>", unsafe_allow_html=True)

            show = [c for c in ["우선순위", "grid_id", "station", "center", "emd",
                                "risk", "fires", "n_hydrant", "dist_hydrant_m",
                                "target_total", "biz_total"] if c in blind.columns]
            tb = blind[show].rename(columns={
                "grid_id": "구역", "station": "소방서", "center": "119안전센터",
                "emd": "읍면동", "risk": "위험도", "fires": "실제화재",
                "n_hydrant": "소화전", "dist_hydrant_m": "최근접 소화전",
                "target_total": "대상물", "biz_total": "업소"})
            # 소수점 여섯 자리는 읽는 사람에게 아무 정보도 주지 않는다.
            if "위험도" in tb:
                tb["위험도"] = tb["위험도"].astype(float).round(1)
            if "최근접 소화전" in tb:
                tb["최근접 소화전"] = tb["최근접 소화전"].apply(
                    lambda v: "—" if pd.isna(v) else f"{float(v):,.0f}m")
            for c in ("실제화재", "소화전", "대상물", "업소"):
                if c in tb:
                    tb[c] = pd.to_numeric(tb[c], errors="coerce").fillna(0).astype(int)
            st.dataframe(tb, hide_index=True, width='stretch', height=330,
                         column_config={"위험도": st.column_config.ProgressColumn(
                             "위험도", format="%.1f", min_value=0.0,
                             max_value=float(max(tb["위험도"].max(), 1)))}
                         if "위험도" in tb else None)

            m1, m2 = st.columns([2, 1])
            with m1:
                st.pydeck_chart(deck([pdk.Layer(
                    "ScatterplotLayer",
                    blind.assign(lon=pd.to_numeric(blind["lon"]),
                                 lat=pd.to_numeric(blind["lat"])),
                    get_position=["lon", "lat"], get_radius=380,
                    get_fill_color=[192, 73, 47, 180], pickable=True)], blind,
                    tooltip=TIP_PLAIN))
            with m2:
                if "station" in blind.columns:
                    by = (blind.groupby("station").size()
                          .rename("사각 구역").reset_index()
                          .sort_values("사각 구역", ascending=False))
                    st.markdown("**소방서별 사각 구역**")
                    st.dataframe(by.rename(columns={"station": "소방서"}),
                                 hide_index=True, width='stretch')
            st.download_button("사각지대 내려받기 (CSV)",
                               blind.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"소방용수사각_{city}_{year}.csv")

    st.divider()
    surge = H.surge_alert(panel, year)
    st.markdown(f"##### 화재 급증 구역 ({year}년, 전년 대비 2배 이상)")
    if surge.empty:
        st.caption("해당 구역이 없습니다.")
    else:
        st.caption(f"{len(surge)}개 구역. 같은 위험요인이 반복되는지 확인이 필요합니다.")
        sg = surge.rename(columns={
            "grid_id": "구역", "station": "소방서", "center": "119안전센터",
            "sgg": "시군구", "emd": "읍면동",
            "fires": f"{year}년", "fires_lag1": f"{year - 1}년"})
        if "증가배수" in sg:
            sg["증가배수"] = sg["증가배수"].astype(float).round(1)
        for c in (f"{year}년", f"{year - 1}년"):
            if c in sg:
                sg[c] = pd.to_numeric(sg[c], errors="coerce").fillna(0).astype(int)
        st.dataframe(sg, hide_index=True, width='stretch', height=260)


# ================================================================== ⑥ 검증
with tabs[5]:
    st.subheader("예측 성능 검증 결과")
    ev = get_evaluation()
    if not ev:
        st.markdown("<div class='callout info'>`scripts/04_train_eval.py` 를 실행하면 "
                    "검증 결과가 표시됩니다.</div>", unsafe_allow_html=True)
    else:
        t = ev.get("temporal", {})
        h = t.get("headline", {})
        key = f"top{cfg.headline_k}"
        ci = t.get("ci", {}).get(key, {})
        m_ci, d_ci = ci.get("model", {}), ci.get("delta", {})
        tr = [int(y) for y in t.get("train_years", [0])]

        st.markdown(
            f"<div class='callout good'><b>{tr[0]}~{tr[-1]}년 자료로 학습해 "
            f"{int(t.get('test_year', 0))}년 화재를 예측했습니다.</b> "
            f"{int(t.get('test_year', 0))}년 자료는 학습에 한 건도 쓰지 않았습니다.</div>",
            unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"위험 상위 {cfg.headline_k}% 포착률", f"{h.get('model_capture',0):.1%}",
                  (f"95% CI {m_ci['lo']:.1%}–{m_ci['hi']:.1%}"
                   if m_ci and m_ci.get("lo") == m_ci.get("lo") else ""),
                  delta_color="off",
                  help="위험 상위 구간에 점검을 집중했을 때 그해 실제 화재의 몇 %가 "
                       "그 안에서 발생했는가")
        c2.metric("무작위 배정 대비", f"{h.get('model_lift',0):.2f}배",
                  help="같은 면적을 아무 데나 골랐을 때보다 몇 배 더 잡는가. "
                       "학계에서는 PAI(Predictive Accuracy Index)라 부릅니다.")
        pei = t.get("model", {}).get("pei", {}).get(key)
        if pei:
            c3.metric("달성 가능 최대치 대비", f"{pei:.1%}",
                      help="실제 화재를 다 알고 줄 세웠을 때를 100으로 봤을 때 어디쯤인지. "
                           "학계에서는 PEI 라 부르며, 지역이 달라도 비교할 수 있습니다.")
        cal = t.get("calibration", {})
        if cal:
            c4.metric("예측 건수 정확도", f"{cal.get('total_ratio',0):.2f}",
                      f"예측 {cal.get('total_predicted',0):.0f} / "
                      f"실제 {cal.get('total_actual',0):.0f}", delta_color="off",
                      help="1.00 이면 건수까지 맞다는 뜻으로, "
                           "'이 구역은 연 3건 예상' 같은 말을 쓸 수 있습니다.")

        if d_ci and d_ci.get("lo_pp") == d_ci.get("lo_pp"):
            verdict = ("통계적으로 유의합니다" if d_ci["excludes_zero"]
                       else "신뢰구간이 0을 포함해, 개선으로 단정하기 어렵습니다")
            st.caption(f"전년 화재 순으로 갈 때({h.get('baseline_capture',0):.1%}) 대비 "
                       f"{d_ci['point_pp']:+.1f}%p (95% CI {d_ci['lo_pp']:+.1f}~"
                       f"{d_ci['hi_pp']:+.1f}%p) — {verdict}.")

        st.divider()
        g1, g2 = st.columns([3, 2])
        with g1:
            st.markdown("##### 포착률 곡선")
            capm = t.get("model", {}).get("capture", {})
            capb = t.get("baseline", {}).get("capture", {})
            if capm:
                curve = pd.DataFrame({"상위 %": [int(k[3:]) for k in capm],
                                      "불씨예보": list(capm.values()),
                                      "단순 기준(전년 화재 순)": [capb.get(k) for k in capm]})
                curve["무작위 배정"] = curve["상위 %"] / 100
                st.line_chart(curve.set_index("상위 %"), height=280)
        with g2:
            st.markdown("##### 위험 등급별 실제 화재")
            dec = t.get("model", {}).get("decile", [])
            if dec:
                dd = pd.DataFrame(dec)
                st.bar_chart(dd.set_index("grade")["mean_fires"],
                             color="#c0492f", height=280)
                st.caption(f"1등급 {dd['mean_fires'].iloc[0]:.2f}건 → "
                           f"{len(dd)}등급 {dd['mean_fires'].iloc[-1]:.2f}건")

        st.divider()
        st.markdown("##### 타 지역·타 관할 적용 검증")
        rows = []
        rows.append({"확인한 것": "미래 예측", "질문": f"{int(t.get('test_year',0))}년을 맞히는가",
                     "결과": f"{h.get('model_capture',0):.1%}",
                     "비고": (f"단순 기준 대비 {d_ci['point_pp']:+.1f}%p"
                            if d_ci else "")})
        if "logo" in ev:
            lg = ev["logo"]
            rows.append({"확인한 것": "관할 제외", "질문": "특정 구·군만 잘 맞는 것은 아닌가",
                         "결과": f"{lg['capture_min']:.1%} ~ {lg['capture_max']:.1%}",
                         "비고": f"평균 {lg['capture_mean']:.1%}"})
        if "transfer" in ev:
            tf = ev["transfer"]
            tci = tf.get("ci", {}).get(key, {}).get("model", {})
            rows.append({"확인한 것": "타 지역 적용",
                         "질문": "울산에서 만든 것이 세종에서도 되는가",
                         "결과": f"{tf['headline']['model_capture']:.1%}",
                         "비고": (f"표본이 작아 {tci['lo']:.0%}–{tci['hi']:.0%} 범위"
                                if tci and tci.get("lo") == tci.get("lo") else "")})
        if "resolution_scenarios" in ev:
            road = next((x for x in ev["resolution_scenarios"]
                         if "도로명" in str(x.get("시나리오", ""))), None)
            if road:
                rows.append({"확인한 것": "주소 정밀도",
                             "질문": "주소가 거칠어 성능이 부풀려진 것은 아닌가",
                             "결과": f"{road.get('모델포착@20%', 0):.1%}",
                             "비고": "부풀림 없음"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

        cc = st.columns(2)
        with cc[0]:
            if "equity" in t:
                st.markdown("##### 관할별 배분 형평성")
                eq = pd.DataFrame(t["equity"])
                eq = eq[eq["group"].astype(str).str.strip() != ""]
                st.dataframe(eq[["group", "share_of_fires", "share_of_inspections",
                                 "inspection_vs_risk"]].rename(columns={
                    "group": "관할", "share_of_fires": "화재비중",
                    "share_of_inspections": "점검비중", "inspection_vs_risk": "비율"}),
                    hide_index=True, width='stretch')
                st.caption("비율 1.0 = 위험한 만큼 점검이 갔다는 뜻입니다.")
        with cc[1]:
            if "ranking_comparison" in ev and ev["ranking_comparison"].get("capture_by_model"):
                st.markdown("##### 알고리즘 비교")
                rc = ev["ranking_comparison"]["capture_by_model"]
                names = {"lambdarank": "순위학습", "poisson": "회귀(기준선)",
                         "baseline_last_year": "단순 기준(전년 화재 순)"}
                st.dataframe(pd.DataFrame({
                    "방식": [names.get(k, k) for k in rc],
                    f"상위 {cfg.headline_k}% 포착률": list(rc.values())}),
                    hide_index=True, width='stretch')
            if "resolution_scenarios" in ev:
                st.markdown("##### 주소 해상도별 성능")
                st.dataframe(pd.DataFrame(ev["resolution_scenarios"])
                             [["시나리오", "모델포착@20%", "PAI", "PEI"]],
                             hide_index=True, width='stretch')

        with st.expander("용어 설명"):
            st.markdown("""
| 화면 표기 | 뜻 | 학술 용어 |
|---|---|---|
| **상위 20% 포착률** | 위험 상위 20% 구역에 점검을 집중했을 때, 그해 실제 화재의 몇 %가 그 안에서 났는가 | capture rate |
| **무작위 배정 대비** | 같은 면적을 아무 데나 골랐을 때보다 몇 배 더 잡는가 | PAI |
| **달성 가능 최대치 대비** | 실제 화재를 다 알고 줄 세운 ‘정답 순위’를 100으로 봤을 때 어디쯤인가. 화재가 몇 군데에 몰린 지역은 어떤 방법이든 포착률이 높게 나오므로, 이 지표라야 지역 간 비교가 됩니다 | PEI |
| **예측 건수 정확도** | 예측 총건수 ÷ 실제 총건수. 순위뿐 아니라 값도 맞는가 | calibration |
| **95% CI** | 같은 조사를 100번 다시 하면 95번은 이 범위에 들어온다는 뜻. 넓으면 표본이 작다는 신호 | 신뢰구간 |
| **단순 기준** | 전년도에 화재가 많았던 구역 순으로 줄 세운 것. 학습 없이 누구나 할 수 있어 비교 기준으로 씁니다 | baseline |
| **관할 제외 검증** | 구·군을 하나씩 통째로 빼고 학습해 그 지역을 맞히기 | LOGO |
| **위험도 상승 요인** | 이 구역의 점수를 무엇이 얼마나 끌어올렸는가 | SHAP |
""")


# ================================================================== ⑦ 업무 도우미
with tabs[6]:
    st.subheader("화재예방 업무 도우미")

    arts = get_law_articles()
    annex = get_law_annexes()
    n_annex = int(len(annex)) if annex is not None and not annex.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("수록 법령", f"{arts['law'].nunique() if not arts.empty else 0}종")
    c2.metric("조문", f"{len(arts):,}개" if not arts.empty else "0개")
    c3.metric("별표·서식", f"{n_annex}건")
    st.caption("국가법령정보센터 법령 + 업종별 점검 규칙 + "
               f"{cfg.city(city)['label']} 분석 결과를 함께 찾습니다. "
               "답변에는 반드시 근거 조문 또는 출처가 표시됩니다.")

    if arts.empty:
        st.markdown("<div class='callout'>법령 자료를 불러오지 못했습니다. "
                    "네트워크를 확인하십시오. 업무규칙과 분석 결과만으로 답변합니다.</div>",
                    unsafe_allow_html=True)

    QUICK = [
        ("법령", "화재예방강화지구는 어떤 지역을 지정하고, 소방관서장은 무엇을 해야 하나요?"),
        ("법령", "소방안전관리자를 선임해야 하는 특급 대상물 범위는?"),
        ("법령", "다중이용업소 안전시설등 정기점검은 어떻게 하나요?"),
        ("법령", "화재안전조사를 연기하려면 어떻게 해야 하나요?"),
        ("법령", "특수가연물에는 어떤 것들이 있나요?"),
        ("업무", "노래연습장 순찰할 때 무엇을 중점적으로 봐야 하나요?"),
        ("업무", "건조기 특별경계순찰은 어떤 곳을 도나요?"),
        ("현황", f"{cur['center'].mode().iloc[0] if 'center' in cur and len(cur['center'].mode()) else '삼산119안전센터'} 관할에서 위험이 높은 구역은?"),
    ]

    st.markdown("##### 자주 찾는 질문")
    cols = st.columns(4)
    for i, (tag, q) in enumerate(QUICK):
        label = q if len(q) <= 26 else q[:25] + "…"
        if cols[i % 4].button(f"{label}", key=f"q_{i}", width='stretch',
                              help=q):
            st.session_state["qa_question"] = q
            st.session_state["qa_run"] = True

    question = st.text_area(
        "질문", value=st.session_state.get("qa_question", ""), height=90,
        placeholder="예) 3급 대상물 자체점검 주기가 어떻게 되나요?",
        key="qa_input")

    a1, a2, a3 = st.columns([1, 1, 3])
    ask = a1.button("질문하기", type="primary", width='stretch')
    top_k = a2.number_input("참고 자료", 3, 10, 5, label_visibility="collapsed",
                            help="답변에 사용할 자료 수")
    if L.is_available(cfg):
        a3.caption("AI 답변 사용 가능 · 15~40초 소요")
    else:
        a3.caption("AI 미연결. 관련 자료를 찾아 그대로 보여 드립니다")

    if (ask or st.session_state.pop("qa_run", False)) and question.strip():
        index = get_index(city, year)
        with st.spinner("법령과 자료를 찾고 답변을 작성 중…"):
            res = AS.answer(cfg, question.strip(), index, top_k=int(top_k))
        st.session_state["qa_result"] = res
        st.session_state["qa_asked"] = question.strip()

    res = st.session_state.get("qa_result")
    if res:
        st.divider()
        if res["source_backend"] == "no_match":
            st.markdown("<div class='callout'>관련 자료를 찾지 못했습니다. "
                        "질문을 다르게 표현해 보십시오.</div>", unsafe_allow_html=True)
        else:
            if res["source_backend"] == "search_only":
                st.markdown("<div class='callout info'>AI 답변 생성이 연결되지 않아 "
                            "관련 자료를 그대로 보여 드립니다.</div>",
                            unsafe_allow_html=True)
            st.caption(f"질문: {st.session_state.get('qa_asked','')}")
            st.markdown(f"<div class='answer'>{_md_to_html(res['answer'])}</div>",
                        unsafe_allow_html=True)

            # 답변이 인용한 조문을 검색된 원문과 하나하나 대조한 결과.
            # 지어낸 조문 번호는 실무에서 답변이 없는 것보다 나쁘다.
            g = res.get("grounding") or {}
            warn = []
            if g.get("unsupported"):
                warn.append("자료에서 찾지 못한 인용: <b>"
                            + ", ".join(x.split(":", 1)[1] for x in g["unsupported"])
                            + "</b>")
            if g.get("unpaired"):
                warn.append("법령과 조문이 같은 자료 안에서 확인되지 않음: <b>"
                            + ", ".join(g["unpaired"]) + "</b>")
            if warn:
                st.markdown(
                    "<div class='callout'><b>확인 필요</b><br>"
                    + "<br>".join(warn)
                    + "<br>국가법령정보센터(law.go.kr)에서 직접 확인하십시오.</div>",
                    unsafe_allow_html=True)
            elif g.get("cited"):
                msg = (f"인용한 법령·조문 {len(g['cited'])}건이 모두 아래 "
                       f"근거 자료 안에서 확인되었습니다.")
                if g.get("unverified"):
                    msg += (" 다만 " + ", ".join(
                        x.split(":", 1)[1] for x in g["unverified"])
                        + " 은 원문이 항 번호를 '①' 형태로 적어 항까지는 대조하지 "
                          "못했습니다.")
                st.caption(msg)

            if res["sources"]:
                st.markdown("##### 근거 자료")
                icons = {"법령": "📘", "업무규칙": "📋", "데이터": "📊"}
                for i, src in enumerate(res["sources"], 1):
                    icon = icons.get(src["source"], "📄")
                    with st.expander(f"{icon} ({i}) {src['title']}", expanded=(i == 1)):
                        if src["ref"]:
                            st.caption(f"근거: {src['ref']}")
                        st.markdown(f"<div class='srcbox'>{src['excerpt']}</div>",
                                    unsafe_allow_html=True)
            st.download_button(
                "답변 내려받기", (f"# {st.session_state.get('qa_asked','')}\n\n"
                             + res["answer"] + "\n\n## 근거\n"
                             + "\n".join(f"- {x['ref']}" for x in res["sources"])
                             ).encode("utf-8"),
                file_name="업무도우미_답변.md")
        st.caption("법령 원문은 국가법령정보센터(law.go.kr)에서 확인하십시오. "
                   "본 답변은 업무 참고용이며 법적 효력을 갖지 않습니다.")


st.divider()
st.caption("본 시스템은 공개 데이터 기반 예측 결과이며, 법정 점검주기 및 관할 판단을 "
           "대체하지 않습니다. 격자 단위 집계로 개별 건물을 특정하지 않습니다.")
