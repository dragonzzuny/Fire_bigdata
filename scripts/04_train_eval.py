#!/usr/bin/env python
"""위험모델 학습과 세 가지 검증 -> outputs/evaluation.json + 사람이 읽는 요약.

기획서의 핵심 수치가 나오는 자리다:
  · 상위 20% 포착률 (모델 vs '작년 화재 순' 베이스라인)
  · 시간분할 홀드아웃(2021) 포착률·리프트·10등급 단조성
  · LOGO(구·군 제외) 범위
  · 울산 -> 세종 이식
숫자를 주장하기 전에 이 스크립트가 실제로 그 숫자를 뱉어야 한다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from firebird import dataset as D, evaluate as E, explain as X, features as F, model as M  # noqa: E402
from firebird.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="ulsan", help="학습·검증 도시")
    ap.add_argument("--transfer-to", default="sejong", help="이식 검증 도시(없으면 생략)")
    ap.add_argument("--skip-logo", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    panel = D.load_panel(cfg, args.city)
    feats = F.feature_columns(panel)
    F.assert_no_leakage(panel)
    log.info("패널 %s행 · 피처 %d개 · 격자 %d개",
             f"{len(panel):,}", len(feats), panel["grid_id"].nunique())

    report: dict = {"city": args.city, "n_features": len(feats), "features": feats,
                    "grid_size_m": cfg.grid_size_m}

    # --- 1) 시간분할 홀드아웃 ---
    print(f"\n{'='*78}\n[1] 시간분할: ~{cfg.split_year} 학습 -> {cfg.holdout_year} 예측")
    temporal = M.temporal_validation(panel, feats, cfg)
    print(E.format_report(temporal["result"]))
    report["temporal"] = temporal["result"]

    model = temporal["model"]
    cfg.paths.outputs.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": model.estimator, "feature_cols": model.feature_cols,
                 "train_years": model.train_years},
                cfg.paths.outputs / f"model_{args.city}.joblib")
    temporal["predictions"].to_csv(
        cfg.paths.outputs / f"predictions_{args.city}_{cfg.holdout_year}.csv",
        index=False, encoding="utf-8-sig")

    # --- 1b) 스냅샷 피처를 뺀 시간분할 ---
    # 대상물·업소·소화전 데이터에는 '언제부터 존재했는가'가 없어서, 과거 연도 행에도
    # 현재 값이 들어간다. 그 상태의 시간분할 점수를 '미래 성능'이라고 부르면
    # 심사에서 바로 무너진다. 그래서 이력 피처만으로도 한 번 더 잰다.
    hist_only = F.history_only_columns(panel)
    print(f"\n{'='*78}\n[1b] 스냅샷 피처 제외(이력만 {len(hist_only)}개)로 다시 시간분할")
    print("     대상물·업소는 현재 시점 스냅샷이라 과거 행에 미래 정보가 섞인다.")
    if hist_only:
        temporal_hist = M.temporal_validation(panel, hist_only, cfg)
        print(E.format_report(temporal_hist["result"]))
        report["temporal_history_only"] = temporal_hist["result"]
        gap = (temporal["result"]["headline"]["model_capture"]
               - temporal_hist["result"]["headline"]["model_capture"]) * 100
        print(f"\n스냅샷 피처가 더한 몫: {gap:+.1f}%p "
              f"(이 값이 크면 '미래 성능' 주장은 그만큼 조심해야 한다)")
        report["snapshot_contribution_pp"] = gap
    else:
        print("     이력 피처가 없어 생략")

    # --- 2) 단일 피처 대조 ---
    print(f"\n{'='*78}\n[2] 단일 피처만으로 줄 세우면? (모델의 진짜 몫을 가늠한다)")
    probe = M.single_feature_probe(panel, cfg)
    if not probe.empty:
        print(probe.to_string(index=False, float_format=lambda v: f"{v:.1%}"))
        report["single_feature_probe"] = probe.to_dict(orient="records")

    # --- 3) LOGO ---
    if not args.skip_logo:
        print(f"\n{'='*78}\n[3] LOGO: 구·군을 통째로 빼고 학습")
        try:
            logo = M.logo_validation(panel, feats, cfg)
            for r in logo["per_group"]:
                print(f"  {r['group']:<10} 포착 {r['capture']:.1%} "
                      f"(베이스라인 {r['baseline_capture']:.1%}, 화재 {r['total_fires']:.0f}건)")
            print(f"  범위 {logo['capture_min']:.1%} ~ {logo['capture_max']:.1%} "
                  f"(평균 {logo['capture_mean']:.1%})")
            report["logo"] = logo
        except (KeyError, ValueError) as exc:
            log.warning("LOGO 생략: %s", exc)

    # --- 4) 도시 이식 ---
    if args.transfer_to:
        print(f"\n{'='*78}\n[4] 도시 이식: {args.city} 학습 -> {args.transfer_to} 적용")
        try:
            target = D.load_panel(cfg, args.transfer_to)
            tr = M.transfer_validation(panel, target, feats, cfg,
                                       target_years=[cfg.holdout_year])
            print(E.format_report(tr))
            report["transfer"] = tr
        except FileNotFoundError as exc:
            log.warning("이식 검증 생략: %s", exc)

    # --- 5) SHAP 전역 중요도 ---
    if not args.skip_shap:
        print(f"\n{'='*78}\n[5] 모델이 무엇을 보는가 (|SHAP| 평균)")
        test = panel[panel["year"] == cfg.holdout_year]
        imp = X.global_importance(model, test.sample(min(3000, len(test)), random_state=0))
        print(imp.to_string(index=False))
        imp.to_csv(cfg.paths.outputs / "shap_importance.csv", index=False, encoding="utf-8-sig")
        report["shap_importance"] = imp.to_dict(orient="records")

    out = cfg.paths.outputs / "evaluation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"\n{'='*78}\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
