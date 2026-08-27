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

from firebird import dataset as D, evaluate as E, explain as X, features as F, \
    model as M, resolution as RES  # noqa: E402
from firebird.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width)


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
    ci_line = E.format_ci(temporal["result"].get("ci", {}), cfg.headline_k)
    if ci_line:
        print(f"신뢰구간: {ci_line}")
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

    # --- 3b) 해상도 정직성: 읍면동 뭉침이 포착률을 얼마나 부풀리는가 ---
    print(f"\n{'='*78}\n[3b] 해상도 검사 — 동 중심점 뭉침의 영향")
    fires_path = cfg.paths.processed / f"fires_{args.city}.parquet"
    if fires_path.exists():
        fires = pd.read_parquet(fires_path)
        conc = RES.concentration(fires[fires["grid_id"].notna()])
        print(f"  화재 {conc['n_fires']:,}건이 격자 {conc['n_grids']:,}개에 분포 "
              f"(지니 {conc['gini']:.2f}, 최다 격자 1개가 {conc['top1_share']:.1%})")
        if "share_from_emd_centroid" in conc:
            print(f"  이 중 {conc['share_from_emd_centroid']:.1%} 는 읍면동 중심점으로 배정됨 "
                  f"— 격자 {conc.get('emd_centroid_grids',0):,}개에 최대 "
                  f"{conc.get('max_fires_in_one_centroid_grid',0):,}건이 쌓여 있다")
        report["resolution_concentration"] = conc

        scenarios = {"원본(동 중심점 포함)": temporal["result"]}

        # (a) 도로명으로 좌표를 얻은 화재만 라벨로 — 해상도가 실제로 도로 단위인 구간
        road_panel = RES.road_only_panel(panel, fires, cfg)
        if road_panel["fires"].sum() > 0:
            try:
                scenarios["도로명 좌표만"] = M.temporal_validation(road_panel, feats, cfg)["result"]
            except (ValueError, KeyError) as exc:
                log.warning("도로명 한정 평가 생략: %s", exc)

        # (b) 동 중심점에 뭉친 화재를 그 동의 격자에 대상물 밀도 비례로 흩뿌린 뒤 재평가.
        #     화재가 실제로 어디서 났는지는 모른다. 다만 한 점에 전부 몰아두는 것보다
        #     건물이 있는 곳에 비례해 나누는 편이 덜 틀린다. 추정이므로 보정 전/후를
        #     항상 함께 보고한다.
        try:
            targets = pd.read_parquet(cfg.paths.processed / f"targets_{args.city}.parquet") \
                if (cfg.paths.processed / f"targets_{args.city}.parquet").exists() else pd.DataFrame()
            biz = pd.read_parquet(cfg.paths.processed / f"businesses_{args.city}.parquet") \
                if (cfg.paths.processed / f"businesses_{args.city}.parquet").exists() else pd.DataFrame()
            weights = RES.build_emd_weights(targets, biz)
            if not weights.empty:
                spread = RES.spread_centroid_fires(fires, weights)
                spread_panel = RES.relabel_panel(panel, spread, cfg)
                moved = int((spread["geo_level"] == "emd_spread").sum())
                print(f"  동 중심점 화재 {moved:,}건을 동 내 {weights['grid_id'].nunique():,}개 "
                      f"격자에 대상물 밀도 비례로 분산 배정")
                scenarios["동 내 분산 배정(진단용)"] = M.temporal_validation(
                    spread_panel, feats, cfg)["result"]
            else:
                log.warning("분산 배정 생략: 대상물/업소 격자 정보가 없다")
        except (ValueError, KeyError, FileNotFoundError) as exc:
            log.warning("분산 배정 평가 생략: %s", exc)

        table = RES.compare_resolutions(scenarios, cfg.headline_k)
        if not table.empty:
            print()
            print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            report["resolution_scenarios"] = table.to_dict(orient="records")
            if "동 내 분산 배정(진단용)" in scenarios:
                print()
                for line in _wrap(RES.SPREAD_CIRCULARITY_WARNING, 92):
                    print(f"  ! {line}")
                report["spread_warning"] = RES.SPREAD_CIRCULARITY_WARNING
    else:
        log.warning("fires parquet 없음 — 해상도 검사 생략")

    # --- 3c) 순위학습 비교 ---
    print(f"\n{'='*78}\n[3c] 모델 비교: 회귀(Poisson) vs 순위학습(LambdaRank) vs 베이스라인")
    print("     이 도구의 출력은 '건수'가 아니라 '점검 순서'다. 순위를 직접 배우면 나은가?")
    try:
        rank_cmp = M.ranking_comparison(panel, feats, cfg)
        if rank_cmp.get("capture_by_model"):
            key = f"top{cfg.headline_k}"
            for name, r in rank_cmp.items():
                if not isinstance(r, dict) or "capture" not in r:
                    continue
                print(f"  {name:<20} 포착 {r['capture'][key]:.1%}  "
                      f"PAI {r['pai'][key]:.2f}  PEI {r['pei'][key]:.1%}")
            print(f"  -> 우승: {rank_cmp['winner']}")
            report["ranking_comparison"] = rank_cmp
    except Exception as exc:                                   # noqa: BLE001
        log.warning("순위학습 비교 생략: %s", exc)

    # --- 4) 도시 이식 ---
    if args.transfer_to:
        print(f"\n{'='*78}\n[4] 도시 이식: {args.city} 학습 -> {args.transfer_to} 적용")
        try:
            target = D.load_panel(cfg, args.transfer_to)
            tr = M.transfer_validation(panel, target, feats, cfg,
                                       target_years=[cfg.holdout_year])
            print(E.format_report(tr))
            ci_line = E.format_ci(tr.get("ci", {}), cfg.headline_k)
            if ci_line:
                print(f"신뢰구간: {ci_line}")
                n = tr["model"]["total_fires"]
                if n < 300:
                    print(f"  주의: 화재 {n:.0f}건짜리 표본이다. 점추정보다 구간을 보라.")
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

    # --- 6) 기획서에 없던 지표 요약 ---
    t = report["temporal"]
    print(f"\n{'='*78}\n[6] 예측치안 표준지표 · 캘리브레이션 · 형평성")
    key = f"top{cfg.headline_k}"
    print(f"  PEI(이론상 최선 대비) 상위{cfg.headline_k}%: "
          f"{t['model']['pei'][key]:.1%}  (PAI {t['model']['pai'][key]:.2f} / "
          f"도달가능 최대 {t['model']['pai_max'][key]:.2f})")
    c = t["calibration"]
    print(f"  캘리브레이션: 실제 {c['total_actual']:.0f}건 vs 예측 {c['total_predicted']:.0f}건 "
          f"(비율 {c['total_ratio']:.2f}) — 값 자체를 인력 배분에 쓸 수 있는가의 척도")
    if "equity_summary" in t:
        es = t["equity_summary"]
        print(f"  형평성: 점검/위험 비율 {es['inspection_vs_risk_min']:.2f}~"
              f"{es['inspection_vs_risk_max']:.2f}"
              + (f", 상대적 소외 {es['underserved']}" if es.get("underserved") else ", 편중 없음"))
    if "recapture" in t:
        rc = t["recapture"]
        print(f"  재발: 작년 화재 격자({rc['share_of_grids']:.0%})가 올해 화재의 "
              f"{rc['share_of_curr_fires_in_prev_grids']:.0%}를 담는다 "
              f"— 베이스라인이 센 이유")

    out = cfg.paths.outputs / "evaluation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"\n{'='*78}\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
