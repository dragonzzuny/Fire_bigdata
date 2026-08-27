#!/usr/bin/env bash
# 불씨예보 전 과정 재현. 원본 CSV -> 패널 -> 학습·검증 -> 현장 산출물.
#
# 사용법:
#   ./scripts/run_all.sh                 # 전체
#   ./scripts/run_all.sh --no-api        # 지오코딩 캐시만 사용(네트워크 없이)
#   FIREBIRD_DATA_ROOT=/mnt/x ./scripts/run_all.sh   # 데이터를 다른 디스크에 둘 때
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

NO_API=""
for arg in "$@"; do
  [ "$arg" = "--no-api" ] && NO_API="--no-api"
done

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "0/7 환경 확인"
"$PY" -c "import pandas, lightgbm, shap, pyproj, streamlit; print('의존성 OK')"

step "1/7 원본 진단"
# 필수 컬럼이 안 맞으면 여기서 멈춘다. 반쯤 맞는 데이터로 학습하느니
# 스키마를 먼저 고치는 편이 싸다.
"$PY" scripts/01_inspect_raw.py

step "2/7 지오코딩 캐시 채우기"
if [ -n "$NO_API" ]; then
  echo "  --no-api: 건너뜀 (기존 캐시 사용)"
else
  "$PY" scripts/02_geocode.py
fi

step "3/7 격자 패널 구축"
"$PY" scripts/03_build_dataset.py $NO_API

step "4/7 학습 · 검증 (시간분할 / 관할제외 / 타지역 적용)"
"$PY" scripts/04_train_eval.py

step "5/7 현장 산출물 (배분·점검계획서·순찰·소화전)"
"$PY" scripts/05_artifacts.py

step "6/7 발표용 그림"
"$PY" scripts/06_figures.py

step "7/7 발표자료(PPTX)"
"$PY" scripts/07_deck.py

step "완료"
echo "결과:   outputs/  (evaluation.json, allocation_*.csv, inspection_plans/, *.pptx)"
echo "그림:   reports/figures/"
echo "대시보드: $PY -m streamlit run app/streamlit_app.py"
