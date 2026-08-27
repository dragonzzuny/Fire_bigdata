# 재현 상태

기획서에 적은 수치를 이 저장소에서 다시 만들어내는 작업의 진행표.
**원본 데이터가 없으면 어떤 수치도 주장하지 않는다.** 아래 표의 '재현값'은
`scripts/04_train_eval.py` 가 실제로 출력한 값만 채운다.

## 코드 상태

| 항목 | 상태 |
|---|---|
| 파이프라인 코드 (적재→격자→학습→산출물) | 완료 |
| 단위·통합 테스트 | 완료 (합성 데이터로 전 구간 통과) |
| 카카오 지오코딩 (실키 검증) | 완료 |
| 원본 데이터 확보 | **대기** — docs/DATA_SOURCES.md 참조 |
| 실데이터 재현 | **대기** |

## 재현해야 할 수치 (기획서 기준)

| # | 지표 | 기획서 값 | 재현값 | 산출 위치 |
|---|---|---|---|---|
| 1 | 지오코딩 좌표 확보율 | 95.3% | — | `manifest_ulsan.json` → `coverage` |
| 2 | 상위 20% 포착률 (모델) | 73% | — | `evaluation.json` → `temporal.headline.model_capture` |
| 3 | 상위 10% 포착률 | 54% | — | `temporal.model.capture.top10` |
| 4 | 베이스라인(작년화재순) 상위 20% | 65.5% | — | `temporal.baseline.capture.top20` |
| 5 | 모델 − 베이스라인 | +7.4%p | — | `temporal.headline.delta_pp` |
| 6 | 2021 홀드아웃 상위 20% | 67% | — | 동일(홀드아웃 = 2021) |
| 7 | 무작위 대비 리프트 | 3.35배 | — | `temporal.headline.model_lift` |
| 8 | 10등급 평균 화재 | 0.07 → 4.09 | — | `temporal.model.decile` |
| 9 | LOGO(구·군) 상위 20% | 46~59% | — | `logo.capture_min` / `capture_max` |
| 10 | 울산→세종 이식 상위 20% | 47.6% | — | `transfer.headline.model_capture` |
| 11 | 소화전 없는 격자 비율 | 48% | — | `artifacts_summary_*.json` → `hydrant_coverage` |
| 12 | 고위험 + 소화전 사각 격자 | 19개 | — | `artifacts_summary_*.json` → `n_blind_spots` |
| 13 | 스냅샷 피처가 더한 몫 | (기획서에 없음) | — | `evaluation.json` → `snapshot_contribution_pp` |

## 재현 절차

```bash
# 1. data/raw/{ulsan,sejong}/ 에 CSV 배치 (docs/DATA_SOURCES.md)
.venv/bin/python scripts/01_inspect_raw.py      # 컬럼 매칭 확인, 필요시 configs/schema.yaml 보완
./scripts/run_all.sh
```

## 값이 안 맞을 때 먼저 볼 것

기획서 수치와 재현값이 다르면 대개 아래 중 하나다. 모델을 만지기 전에 여기를 본다.

1. **격자 크기** — `configs/config.yaml` 의 `grid.size_m`. 격자를 키우면 포착률이 올라간다
   (같은 화재가 더 적은 셀에 모이므로). 500m 가 아닌 값으로 낸 수치는 비교 대상이 아니다.
2. **격자 모집단** — 화재 0인 격자를 포함했는가. 빼면 분모가 줄어 포착률이 부풀려진다.
3. **연도 범위** — `years.min/max`. 라벨 연도가 다르면 누적 피처가 통째로 달라진다.
4. **좌표 확보율** — 지오코딩이 덜 됐으면 격자 수 자체가 달라진다. `manifest` 를 먼저 본다.
5. **동점 처리** — 베이스라인 수치는 동점 처리 방식에 민감하다. 이 저장소는 고정 시드
   난수로 깬다(입력 순서로 깨면 베이스라인이 부풀려진다).

## 기록

- 2026-08-27 — 원본 작업물 유실 확인. 기획서(PDF)만 남음. 기획서 기준 재구축 시작.
- 2026-08-27 — 파이프라인 전 구간 구현 + 테스트 통과(합성 데이터). 카카오 지오코딩 실키 검증 완료.
- 2026-08-27 — 외부 검토(codex)에서 4건 지적 → 전부 수정:
  스냅샷 피처 시간누수(측정으로 대응), `--no-api` 우회, 좌표 확보율 과대계상, LOGO 무성 실패.
- 2026-08-27 — LLM 백엔드를 CLI 우선(codex, 약 20초/건)으로 전환. API/Ollama/규칙기반 폴백.
- 2026-08-27 — 원본 데이터 대기 중.
