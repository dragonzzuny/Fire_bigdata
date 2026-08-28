#!/usr/bin/env python
"""노후도 피처가 실제로 도움이 되는지 측정한다.

넣고 좋아졌다고 말하려면 넣기 전과 비교해야 하고, 비교는 같은 조건에서
해야 한다. 같은 분할·같은 파라미터·같은 시드로 두 번 돌려 차이만 본다.

**도움이 되지 않으면 넣지 않는다.** 피처를 늘리는 것 자체는 성과가 아니다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402

from firebird import buildings as BD             # noqa: E402
from firebird import dataset as D                # noqa: E402
from firebird import evaluate as E               # noqa: E402
from firebird import features as F               # noqa: E402
from firebird import model as M                  # noqa: E402
from firebird.config import load_config          # noqa: E402


def run(panel: pd.DataFrame, cfg, feats: list[str]) -> dict:
    return M.temporal_validation(panel, feats, cfg)


def main() -> int:
    cfg = load_config()
    city = sys.argv[1] if len(sys.argv) > 1 else "ulsan"
    panel = D.load_panel(cfg, city)
    k = float(cfg.headline_k)
    key = f"top{int(k)}"

    base_feats = F.feature_columns(panel)
    base = run(panel, cfg, base_feats)

    years = sorted(int(y) for y in panel["year"].unique())
    bf = BD.emd_year_features(cfg, years)
    if bf.empty:
        print("건축물대장 자료가 없습니다. scripts/15_fetch_buildings.py 를 먼저 실행하십시오.")
        return 1
    wide = BD.attach_features(panel, bf)
    F.assert_no_leakage(wide)                    # 붙이고 나서도 누수가 없어야 한다
    new_feats = F.feature_columns(wide)
    added = [c for c in new_feats if c not in base_feats]
    withb = run(wide, cfg, new_feats)

    cov = wide[[c for c in added]].notna().any(axis=1).mean() if added else 0.0

    # 같은 홀드아웃에서 두 점수를 짝지어 비교한다. 재표본이 다르면 차이가
    # 부풀거나 사라진다.
    y = withb["predictions"]["fires"].to_numpy()
    delta = E.bootstrap_delta_ci(
        y, withb["predictions"]["pred"].to_numpy(),
        base["predictions"]["pred"].to_numpy(), k,
        n_boot=int(cfg["evaluation"].get("n_bootstrap", 1000)))

    # 더 중요한 검정: 스냅샷 피처를 **빼고** 노후도로 대신할 수 있는가.
    #
    # 대상물·업소·소화전은 '언제부터 존재했는가'가 없어 과거 행에 현재 값이
    # 들어간다. 그 피처들을 빼면 성능이 떨어지는데, 시점이 확실한 노후도가
    # 그 자리를 메운다면 '스냅샷 없이도 같은 성능'이라고 말할 수 있다.
    # 그게 이 자료를 가져온 진짜 이유다.
    hist = F.history_only_columns(panel)
    hist_base = run(panel, cfg, hist)
    hist_with = run(wide, cfg, hist + added)
    hb = hist_base["result"]["model"]["capture"][key]
    hw = hist_with["result"]["model"]["capture"][key]
    hist_delta = E.bootstrap_delta_ci(
        hist_with["predictions"]["fires"].to_numpy(),
        hist_with["predictions"]["pred"].to_numpy(),
        hist_base["predictions"]["pred"].to_numpy(), k,
        n_boot=int(cfg["evaluation"].get("n_bootstrap", 1000)))

    b_cap = base["result"]["model"]["capture"][key]
    w_cap = withb["result"]["model"]["capture"][key]
    out = {
        "city": city, "added_features": added,
        "coverage_share": float(cov),
        "n_emd_with_data": int(bf["emd"].nunique()),
        "baseline_capture": float(b_cap),
        "with_buildings_capture": float(w_cap),
        "delta_pp": float((w_cap - b_cap) * 100),
        "delta_ci": delta,
        "adopted": bool(delta.get("excludes_zero") and w_cap > b_cap),
        "history_only": {
            "baseline_capture": float(hb),
            "with_buildings_capture": float(hw),
            "delta_pp": float((hw - hb) * 100),
            "delta_ci": hist_delta,
        },
        "full_model_capture": float(b_cap),
    }
    print(f"노후도 피처 {len(added)}개: {', '.join(added)}")
    print(f"  자료가 있는 읍면동 {out['n_emd_with_data']}개 · "
          f"격자행 {cov:.1%} 에 값이 있음")
    print(f"  상위 {k:.0f}% 포착률  기존 {b_cap:.1%} -> 넣은 뒤 {w_cap:.1%} "
          f"({out['delta_pp']:+.2f}%p)")
    if delta.get("lo_pp") == delta.get("lo_pp"):
        print(f"  95% 신뢰구간 {delta['lo_pp']:+.2f} ~ {delta['hi_pp']:+.2f}%p "
              f"— {'0을 포함하지 않음(유의)' if delta['excludes_zero'] else '0을 포함(개선 단정 불가)'}")
    print()
    print("스냅샷 피처를 빼고 노후도로 대신했을 때 (누수 없는 조건):")
    print(f"  이력만          {hb:.1%}")
    print(f"  이력 + 노후도   {hw:.1%}  ({(hw - hb) * 100:+.2f}%p)")
    if hist_delta.get("lo_pp") == hist_delta.get("lo_pp"):
        print(f"  95% 신뢰구간 {hist_delta['lo_pp']:+.2f} ~ {hist_delta['hi_pp']:+.2f}%p")
    print(f"  (참고: 스냅샷까지 쓴 전체 모델 {b_cap:.1%})")

    print(f"\n판정: {'채택' if out['adopted'] else '채택하지 않음'}")
    if not out["adopted"]:
        print("  피처를 늘리는 것 자체는 성과가 아니다. 도움이 확인되지 않으면 넣지 않는다.")

    path = cfg.paths.outputs / f"building_feature_eval_{city}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n기록 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
