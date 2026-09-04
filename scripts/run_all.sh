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

step "0/10 환경 확인"
"$PY" -c "import pandas, lightgbm, shap, pyproj, streamlit; print('의존성 OK')"

step "1/10 원본 진단"
# 필수 컬럼이 안 맞으면 여기서 멈춘다. 반쯤 맞는 데이터로 학습하느니
# 스키마를 먼저 고치는 편이 싸다.
"$PY" scripts/01_inspect_raw.py

step "2/10 지오코딩 캐시 채우기"
if [ -n "$NO_API" ]; then
  echo "  --no-api: 건너뜀 (기존 캐시 사용)"
else
  "$PY" scripts/02_geocode.py
fi

step "3/10 격자 패널 구축"
"$PY" scripts/03_build_dataset.py $NO_API

step "4/10 학습 · 검증 (시간분할 / 관할제외 / 타지역 적용)"
"$PY" scripts/04_train_eval.py

step "5/10 현장 산출물 (배분·점검계획서·순찰·소화전)"
"$PY" scripts/05_artifacts.py

step "5b/10 회고 검증 (그해 이전 자료만으로 순찰을 짰다면)"
# '화재가 몇 건 줄어드느냐'에 답하는 자리다. 발표에서 쓰는 수이므로
# 발표자료를 만들기 전에 반드시 돌아야 한다.
"$PY" scripts/17_backtest_patrol.py

step "6/10 브랜드 자산(로고)"
"$PY" scripts/13_brand.py || echo "  건너뜀"

step "6b/10 건축물대장·노후도 (선택, 키가 있을 때)"
# 노후도는 측정 결과 도움이 되지 않아 모델에 넣지 않는다. 다만 법정 서식의
# 연면적·건축연도를 채우는 데 쓰고, 측정 자체를 재현할 수 있게 남겨 둔다.
if [ -n "$NO_API" ]; then
  echo "  --no-api: 건너뜀"
else
  "$PY" scripts/15_fetch_buildings.py || echo "  수집 건너뜀"
  "$PY" scripts/16_eval_buildings.py  || echo "  측정 건너뜀"
fi

step "7/10 발표용 그림"
"$PY" scripts/06_figures.py
"$PY" scripts/14_compare_map.py || echo "  전후 비교 지도 건너뜀"

step "8/10 법정 서식 원본·대조 이미지"
# 네트워크가 막히면 서식만 건너뛴다. 발표자료는 그 없이도 만들어진다.
"$PY" scripts/09_forms.py || echo "  건너뜀 (법제처 접속 실패)"

step "9/10 발표자료(PPTX)"
"$PY" scripts/07_deck.py

step "10/11 발표자료 검증 (수치 출처 · 장표 배치 · 문서 수치)"
# 장표에 근거 없는 숫자가 들어가면 발표장에서 고칠 수 없다. 만들 때 잡는다.
"$PY" scripts/10_audit_numbers.py
"$PY" scripts/11_audit_layout.py
"$PY" scripts/12_audit_docs.py
# 발표자료·대본·영상은 따로 고쳐진다. 서로 어긋나면 발표장에서야 드러난다.
"$PY" scripts/19_audit_talk.py

step "11/11 제출·발표 묶음"
# 손으로 모으면 옛 파일이 섞인다. 스크립트가 그때그때 있는 것을 모은다.
"$PY" scripts/20_bundle.py

step "완료"
echo "결과:   outputs/  (evaluation.json, allocation_*.csv, inspection_plans/, *.pptx)"
echo "화면 캡처: $PY scripts/08_screenshots.py  (대시보드를 먼저 띄워 둘 것)"
echo "시연 녹화: $PY scripts/18_demo_video.py      (리허설·백업용 mp4)"
echo "그림:   reports/figures/"
echo "대시보드: $PY -m streamlit run app/streamlit_app.py"
