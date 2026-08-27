"""화재위험 지도와 순찰 동선을 한 장의 그림으로.

이 서비스의 핵심은 두 가지다. **어디가 위험한가**, 그리고 **어떤 순서로 도는가**.
표와 숫자로는 그 둘이 한눈에 들어오지 않는다. 계획서를 받는 담당자도,
발표를 듣는 심사위원도 결국 지도를 본다.

웹 지도(pydeck)는 화면에서만 살아 있다. 계획서에 붙이고 발표자료에 넣으려면
파일로 남는 그림이 필요하다. 여기서는 격자 자체로 도시 윤곽을 그린다 —
외부 지도 타일에 기대지 않으므로 인터넷 없이도 같은 그림이 나온다.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap

from . import grid as G

log = logging.getLogger(__name__)

#: 위험도 색. 흰색에서 벽돌색으로. 무지개는 등급의 크기를 왜곡한다.
RISK_CMAP = LinearSegmentedColormap.from_list(
    "firebird", ["#f7f7f6", "#f6ddd2", "#e9a184", "#d05f3c", "#9e2b12"])

ROUTE_COLORS = ["#1f4e79", "#0f7b6c", "#8b3d8b", "#b06000", "#2f6f2f",
                "#7a1f3d", "#31708e", "#6b4f1d", "#4a3f8f", "#8a5a2b"]

INK = "#22262d"
MUTED = "#78808c"


def _pick_font() -> str | None:
    """한글이 실제로 그려지는 폰트. 이름만 보고 고르면 두부가 나온다."""
    import warnings
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic",
                 "Malgun Gothic", "AppleGothic"):
        if cand not in available:
            continue
        fig, ax = plt.subplots(figsize=(1, 1))
        plt.rcParams["font.family"] = cand
        ax.text(0.1, 0.5, "화재 순찰", fontsize=10)
        ax.axis("off")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                fig.canvas.draw()
            plt.close(fig)
            return cand
        except Exception:                                # noqa: BLE001
            plt.close(fig)
    return None


_FONT = _pick_font()
if _FONT:
    plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False


def _cells(grid_ids, cfg) -> tuple[np.ndarray, np.ndarray]:
    """격자 id -> 좌하단 미터 좌표 (gx*size, gy*size)."""
    ids = pd.Series(grid_ids).dropna().astype(str)
    gx = ids.str.split("_").str[0].astype(int).to_numpy()
    gy = ids.str.split("_").str[1].astype(int).to_numpy()
    s = cfg.grid_size_m
    return gx * s, gy * s


def _scale_bar(ax, x0, y0, length_m=2000, label="2 km"):
    ax.plot([x0, x0 + length_m], [y0, y0], color=INK, lw=2.4, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 120, y0 + 120], color=INK, lw=2.4)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 120, y0 + 120], color=INK, lw=2.4)
    ax.text(x0 + length_m / 2, y0 + 260, label, ha="center", va="bottom",
            fontsize=9, color=INK)


def _north(ax, x, y, size=900):
    ax.annotate("", xy=(x, y + size), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=INK))
    ax.text(x, y + size + 220, "N", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=INK)


def risk_map(panel_year: pd.DataFrame, cfg, *, risk_col: str = "pred",
             title: str = "", highlight: pd.DataFrame | None = None,
             figsize=(9.0, 8.4)):
    """격자별 화재위험 지도.

    격자 자체가 도시 윤곽을 그린다. 색이 짙을수록 위험이 높다.
    """
    df = panel_year.dropna(subset=["grid_id"]).copy()
    if df.empty:
        return None
    x, y = _cells(df["grid_id"], cfg)
    s = cfg.grid_size_m
    risk = pd.to_numeric(df[risk_col], errors="coerce").fillna(0.0).to_numpy()
    # 백분위로 색을 매긴다. 원값은 소수 격자에 몰려 있어 대부분이 같은 색이 된다.
    pct = pd.Series(risk).rank(pct=True).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    patches = [mpatches.Rectangle((xi, yi), s, s) for xi, yi in zip(x, y)]
    pc = PatchCollection(patches, cmap=RISK_CMAP, edgecolor="#ffffff",
                         linewidths=0.18)
    pc.set_array(pct)
    pc.set_clim(0, 1)
    ax.add_collection(pc)

    if highlight is not None and not highlight.empty:
        hx, hy = _cells(highlight["grid_id"], cfg)
        for xi, yi in zip(hx, hy):
            ax.add_patch(mpatches.Rectangle((xi, yi), s, s, fill=False,
                                            edgecolor="#111111", lw=1.5, zorder=5))

    _finish(ax, x, y, s, title)
    cb = fig.colorbar(pc, ax=ax, fraction=0.032, pad=0.02)
    cb.set_ticks([0.02, 0.5, 0.98])
    cb.set_ticklabels(["낮음", "보통", "높음"])
    cb.ax.tick_params(labelsize=9, colors=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("화재위험", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def route_map(panel_year: pd.DataFrame, routes: list[pd.DataFrame], cfg, *,
              risk_col: str = "pred", title: str = "",
              show_numbers: bool = True, figsize=(10.6, 7.6),
              zoom: bool = True, pad_m: float = 2500.0,
              top_n_outline: int = 0):
    """위험 지도 위에 관서별 순찰 동선을 얹는다.

    관서에서 출발해 순번대로 돌고 관서로 돌아오는 선을 그린다.
    '위험한 곳'과 '실제로 가는 경로'가 겹쳐 보여야 계획이 설명된다.
    """
    df = panel_year.dropna(subset=["grid_id"]).copy()
    if df.empty:
        return None
    x, y = _cells(df["grid_id"], cfg)
    s = cfg.grid_size_m
    risk = pd.to_numeric(df[risk_col], errors="coerce").fillna(0.0)
    pct = risk.rank(pct=True).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    patches = [mpatches.Rectangle((xi, yi), s, s) for xi, yi in zip(x, y)]
    pc = PatchCollection(patches, cmap=RISK_CMAP, edgecolor="#ffffff",
                         linewidths=0.15, alpha=0.85)
    pc.set_array(pct)
    pc.set_clim(0, 1)
    ax.add_collection(pc)

    tf = G._to_metric(cfg.crs_geographic, cfg.crs_metric)
    legend = []
    for i, r in enumerate(routes):
        if r.empty:
            continue
        col = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        depot = r.attrs.get("depot", {})
        rx, ry = tf.transform(r["lon"].astype(float).to_numpy(),
                              r["lat"].astype(float).to_numpy())
        if depot:
            dx, dy = tf.transform(float(depot["lon"]), float(depot["lat"]))
            px = np.r_[dx, rx, dx]        # 관서 → 순찰 → 관서
            py = np.r_[dy, ry, dy]
            ax.plot(dx, dy, marker="s", ms=11, color="#111111", zorder=9,
                    markeredgecolor="white", markeredgewidth=1.4)
            ax.annotate(depot.get("name", "").replace("119안전센터", ""),
                        (dx, dy), textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=8.5, fontweight="bold",
                        color="#111111", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.82))
        else:
            px, py = rx, ry
        ax.plot(px, py, color=col, lw=2.2, alpha=0.9, zorder=7,
                solid_capstyle="round")
        ax.scatter(rx, ry, s=64, color=col, zorder=8, edgecolor="white",
                   linewidths=1.1)
        if show_numbers:
            for n, (xi, yi) in enumerate(zip(rx, ry), 1):
                ax.annotate(str(n), (xi, yi), ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold", zorder=9)
        name = depot.get("name", f"{i+1}팀")
        legend.append(mpatches.Patch(color=col, label=f"{name} ({len(r)}개소)"))

    # 위험 상위 격자에 테두리를 둘러 '어디가 위험한가'가 먼저 눈에 들게 한다.
    if top_n_outline:
        top = df.assign(_r=risk).nlargest(top_n_outline, "_r")
        tx, ty = _cells(top["grid_id"], cfg)
        for xi, yi in zip(tx, ty):
            ax.add_patch(mpatches.Rectangle((xi, yi), s, s, fill=False,
                                            edgecolor="#7a1a05", lw=1.1, zorder=4))

    if zoom and any(not r.empty for r in routes):
        # 도시 전체를 그리면 도심 동선이 점처럼 작아진다. 순찰 범위로 맞춘다.
        xs, ys = [], []
        for r in routes:
            if r.empty:
                continue
            rx, ry = tf.transform(r["lon"].astype(float).to_numpy(),
                                  r["lat"].astype(float).to_numpy())
            xs += list(rx); ys += list(ry)
            dep = r.attrs.get("depot", {})
            if dep:
                dx, dy = tf.transform(float(dep["lon"]), float(dep["lat"]))
                xs.append(dx); ys.append(dy)
        if xs:
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            half_x = max(max(xs) - min(xs), 1.0) / 2 + pad_m
            half_y = max(max(ys) - min(ys), 1.0) / 2 + pad_m
            # 그림틀이 가로형이면 지도도 가로형이어야 여백이 안 생긴다.
            # 좁은 축을 늘려 화면 비율에 맞춘다.
            want = figsize[0] / figsize[1]
            if half_x / half_y < want:
                half_x = half_y * want
            else:
                half_y = half_x / want
            bx = np.array([cx - half_x, cx + half_x])
            by = np.array([cy - half_y, cy + half_y])
            _finish(ax, bx, by, s, title, fit=False)
        else:
            _finish(ax, x, y, s, title)
    else:
        _finish(ax, x, y, s, title)

    if legend:
        ax.legend(handles=legend[:8], loc="upper left", frameon=True,
                  framealpha=0.94, fontsize=9, edgecolor="#dcdfe3")
    fig.tight_layout()
    return fig


def _finish(ax, x, y, s, title: str, *, fit: bool = True):
    if fit:
        ax.set_xlim(x.min() - s, x.max() + 2 * s)
        ax.set_ylim(y.min() - s, y.max() + 2 * s)
    else:
        ax.set_xlim(float(x[0]), float(x[1]))
        ax.set_ylim(float(y[0]), float(y[1]))
        x = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y = np.array([ax.get_ylim()[0], ax.get_ylim()[1]])
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13.5, color=INK, pad=12)
    span = (x.max() - x.min())
    bar = 2000 if span > 12000 else 1000
    _scale_bar(ax, x.min(), y.min() - s * 0.4, bar,
               f"{bar // 1000} km")
    _north(ax, x.max() + s, y.min() + s)


def save(fig, path) -> str:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)
