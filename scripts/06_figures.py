#!/usr/bin/env python
"""발표용 그림 생성 — 전부 outputs/evaluation.json 의 실측값에서만 그린다.

손으로 그린 숫자를 슬라이드에 올리면 질문 한 번에 무너진다.
여기서 만드는 그림은 파이프라인이 실제로 뱉은 값만 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from firebird.config import load_config  # noqa: E402

def _pick_korean_font() -> str | None:
    """한글이 실제로 그려지는 폰트를 고른다.

    이름으로 고르면 안 된다. Noto Sans CJK 는 하나의 .ttc 에 KR/JP/SC/TC 가
    같이 들어 있어서, matplotlib 에는 'Noto Sans CJK JP' 하나만 등록되는 일이
    흔하다. 그 폰트도 한글을 갖고 있다. 그래서 후보를 하나씩 **그려 보고**
    경고가 나지 않는 것을 쓴다 — 라벨이 두부(□)로 나온 슬라이드는 쓸 수 없다.
    """
    import warnings
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = [c for c in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic",
                              "NanumBarunGothic", "Malgun Gothic", "AppleGothic",
                              "Noto Serif CJK JP")
                  if c in available]
    for cand in candidates:
        fig, ax = plt.subplots(figsize=(1, 1))
        plt.rcParams["font.family"] = cand
        ax.text(0.1, 0.5, "화재 위험 점검", fontsize=10)
        ax.axis("off")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                fig.canvas.draw()
            plt.close(fig)
            return cand
        except (UserWarning, Exception):          # noqa: BLE001
            plt.close(fig)
    return None


_FONT = _pick_korean_font()
if _FONT:
    plt.rcParams["font.family"] = _FONT
else:
    print("경고: 한글 폰트를 찾지 못했다. 그림의 한글이 깨진다. "
          "`sudo apt install fonts-noto-cjk` 로 설치하라.")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

RED, BLUE, GRAY, GREEN = "#e34a33", "#2b6cb0", "#9aa5b1", "#2f9e6e"


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.name}")
    return path


def fig_capture_curve(ev: dict, out: Path, k: int) -> None:
    t = ev["temporal"]
    ks = [int(x[3:]) for x in t["model"]["capture"]]
    m = [t["model"]["capture"][f"top{x}"] * 100 for x in ks]
    b = [t["baseline"]["capture"][f"top{x}"] * 100 for x in ks]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, m, "o-", color=RED, lw=2.6, ms=8, label="불씨예보", zorder=3)
    ax.plot(ks, b, "s--", color=BLUE, lw=2, ms=6, label="단순 기준 (전년 화재 순)")
    ax.plot(ks, ks, ":", color=GRAY, lw=1.6, label="무작위 배정")
    i = ks.index(k)
    ax.annotate(f"{m[i]:.1f}%", (ks[i], m[i]), textcoords="offset points",
                xytext=(6, 10), color=RED, fontweight="bold", fontsize=13)
    ax.annotate(f"{b[i]:.1f}%", (ks[i], b[i]), textcoords="offset points",
                xytext=(6, -18), color=BLUE, fontsize=11)
    ax.axvline(k, color=GRAY, lw=1, alpha=.5)
    ax.set_xlabel("점검·순찰을 집중하는 상위 위험 격자 비율 (%)")
    ax.set_ylabel("해당 격자에서 발생한 실제 화재 비율 (%)")
    ax.set_title(f"{int(t['test_year'])}년 화재 포착률 (학습에 미사용한 자료)")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    save(fig, out / "fig_capture_curve.png")


def fig_pei(ev: dict, out: Path, k: int) -> None:
    t = ev["temporal"]["model"]
    key = f"top{k}"
    pai, pmax, pei = t["pai"][key], t["pai_max"][key], t["pei"][key]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh([0], [pmax], color=GRAY, alpha=.45, height=.5, label=f"달성 가능 최대 {pmax:.2f}")
    ax.barh([0], [pai], color=RED, height=.5, label=f"불씨예보 {pai:.2f}")
    ax.set_yticks([]); ax.set_xlabel("PAI (무작위 배정 대비 배수)")
    ax.set_title(f"달성 가능 최대치의 {pei:.0%} 수준  ·  PEI = PAI ÷ PAI 최대")
    ax.text(pai / 2, 0, f"PEI {pei:.0%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=15)
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", alpha=.25)
    save(fig, out / "fig_pei.png")


def fig_decile(ev: dict, out: Path) -> None:
    d = pd.DataFrame(ev["temporal"]["model"]["decile"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = [plt.cm.YlOrRd(0.25 + 0.7 * i / len(d)) for i in range(len(d))]
    ax.bar(d["grade"], d["mean_fires"], color=colors, edgecolor="white")
    for _, r in d.iterrows():
        ax.text(r["grade"], r["mean_fires"], f"{r['mean_fires']:.2f}",
                ha="center", va="bottom", fontsize=9)
    lo, hi = d["mean_fires"].iloc[0], d["mean_fires"].iloc[-1]
    ax.set_xlabel("위험 등급 (1=최저 … 10=최고)")
    ax.set_ylabel("격자당 실제 화재 (건)")
    ax.set_title(f"위험 등급별 실제 화재 발생량  ({lo:.2f} → {hi:.2f}건)")
    ax.set_xticks(d["grade"]); ax.grid(axis="y", alpha=.25)
    save(fig, out / "fig_decile.png")


def fig_allocation(summary: dict, out: Path, k: int) -> None:
    a = summary.get("allocation")
    if not a:
        return
    o, t = a["optimized"], a["top_k_percent"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    ax = axes[0]
    names = ["인력 제약 배분", f"위험도 상위 {k}%"]
    reach = [o["n_grids"], t["n_grids_affordable"]]
    ax.bar(names, reach, color=[GREEN, BLUE], width=.55, label="실제로 갈 수 있음")
    gap = t["n_grids_selected"] - t["n_grids_affordable"]
    ax.bar([names[1]], [gap], bottom=[t["n_grids_affordable"]],
           color=GRAY, alpha=.45, width=.55, label="예산 초과로 못 감")
    # 값을 안 적으면 '상위 k%' 의 파란 칸(1개)이 눈에 보이지 않아
    # 그림이 정반대로 읽힌다.
    ax.text(0, reach[0], f"{reach[0]:,}개", ha="center", va="bottom", fontweight="bold")
    ax.text(1, t["n_grids_selected"], f"{t['n_grids_selected']:,}개 지목",
            ha="center", va="bottom", color=GRAY, fontsize=9)
    ax.annotate(f"실제 {reach[1]:,}개", xy=(1, reach[1]), xytext=(1.32, reach[1] + gap * .35),
                fontweight="bold", color=BLUE, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.set_ylabel("점검 격자 수"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(0, t["n_grids_selected"] * 1.2)
    ax.set_title(f"가용 물량 {a['budget_visits']:,}건으로 점검 가능한 격자")
    ax.grid(axis="y", alpha=.25)

    ax = axes[1]
    if "actual_capture_rate" in o:
        vals = [o["actual_capture_rate"] * 100, t["actual_capture_rate"] * 100]
        ax.bar(["인력 제약 배분", f"위험도 상위 {k}%"], vals, color=[GREEN, BLUE], width=.55)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("실제 화재 포착률 (%)")
        ax.set_title(f"동일 인력 기준 실제 화재 포착률  ({a['gain_pp']:+.1f}%p)")
        ax.grid(axis="y", alpha=.25)
    save(fig, out / "fig_allocation.png")


def fig_equity(ev: dict, out: Path, k: int) -> None:
    t = ev["temporal"]
    if "equity" not in t:
        return
    eq = pd.DataFrame(t["equity"])
    eq = eq[eq["group"].astype(str).str.strip() != ""].sort_values("share_of_fires",
                                                                  ascending=False)
    x = np.arange(len(eq)); w = .38
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - w / 2, eq["share_of_fires"] * 100, w, label="화재 비중", color=RED)
    ax.bar(x + w / 2, eq["share_of_inspections"] * 100, w, label="점검 배분 비중", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(eq["group"])
    ax.set_ylabel("비중 (%)")
    ax.set_title(f"관할별 화재 비중 대비 점검 배분 비중 (상위 {k}%)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    for i, r in enumerate(eq.itertuples()):
        ratio = r.inspection_vs_risk
        if ratio == ratio:
            ax.text(i, max(r.share_of_fires, r.share_of_inspections) * 100 + 1,
                    f"{ratio:.2f}", ha="center", fontsize=9,
                    color=RED if ratio < 0.75 else "black",
                    fontweight="bold" if ratio < 0.75 else "normal")
    save(fig, out / "fig_equity.png")


def fig_model_compare(ev: dict, out: Path, k: int) -> None:
    rc = ev.get("ranking_comparison", {}).get("capture_by_model")
    if not rc:
        return
    names = {"lambdarank": "순위학습\n(LambdaRank)", "poisson": "회귀\n(LightGBM Poisson)",
             "baseline_last_year": "단순 기준\n(전년 화재 순)"}
    items = sorted(rc.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [v * 100 for _, v in items]
    cols = [GREEN if i == 0 else (RED if items[i][0] == "poisson" else GRAY)
            for i in range(len(items))]
    ax.bar([names.get(n, n) for n, _ in items], vals, color=cols, width=.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel(f"상위 {k}% 포착률 (%)"); ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title("동일 검증 조건에서 비교")
    ax.grid(axis="y", alpha=.25)
    save(fig, out / "fig_model_compare.png")


def fig_data_funnel(manifest: dict, out: Path) -> None:
    f = manifest.get("fires", {})
    cov = manifest.get("coverage", {}).get("fire", {})
    dedup = manifest.get("fire_deduplication", {})
    stages = [("원본 자료", f.get("rows", 0) + dedup.get("removed", 0)),
              ("중복 출동 제거", f.get("rows", 0)),
              ("좌표 확보", int(cov.get("with_coords", 0))),
              ("학습 사용", f.get("usable", 0))]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [s[0] for s in stages]; vals = [s[1] for s in stages]
    ax.barh(range(len(vals))[::-1], vals, color=[GRAY, BLUE, BLUE, RED], height=.6)
    for i, (lab, v) in enumerate(zip(labels, vals)):
        ax.text(v, len(vals) - 1 - i, f"  {v:,}", va="center", fontsize=10)
    ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labels)
    ax.set_title("전처리 단계별 자료 건수 (전 단계 기록·공개)")
    ax.grid(axis="x", alpha=.25)
    save(fig, out / "fig_data_funnel.png")


def fig_hour_profile(cfg, city: str, out: Path) -> None:
    p = cfg.paths.processed / f"fires_{city}.parquet"
    if not p.exists():
        return
    fires = pd.read_parquet(p)
    if "hour" not in fires.columns or fires["hour"].notna().sum() == 0:
        return
    from firebird import patrol as P
    prof = P.hour_profile(fires)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cols = [RED if v >= prof["n"].quantile(.75) else BLUE for v in prof["n"]]
    ax.bar(prof["hour"], prof["n"], color=cols, width=.75)
    ax.set_xlabel("시각"); ax.set_ylabel("화재 건수"); ax.set_xticks(range(0, 24, 2))
    n = int(fires["hour"].notna().sum())
    ax.set_title(f"시간대별 화재 발생 분포 (발생 시각 기록분 {n:,}건)")
    ax.grid(axis="y", alpha=.25)
    save(fig, out / "fig_hour_profile.png")


def main() -> int:
    cfg = load_config()
    out = cfg.paths.figures
    ev_path = cfg.paths.outputs / "evaluation.json"
    if not ev_path.exists():
        print("evaluation.json 이 없다. scripts/04_train_eval.py 를 먼저 돌려라.")
        return 1
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    city = ev.get("city", "ulsan")
    k = cfg.headline_k

    year = ev["temporal"]["test_year"]
    sum_path = cfg.paths.outputs / f"artifacts_summary_{city}_{year}.json"
    summary = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path.exists() else {}
    man_path = cfg.paths.processed / f"manifest_{city}.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}

    print(f"그림 생성 -> {out}")
    fig_capture_curve(ev, out, k)
    fig_pei(ev, out, k)
    fig_decile(ev, out)
    fig_allocation(summary, out, k)
    fig_equity(ev, out, k)
    fig_model_compare(ev, out, k)
    fig_data_funnel(manifest, out)
    fig_hour_profile(cfg, city, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
