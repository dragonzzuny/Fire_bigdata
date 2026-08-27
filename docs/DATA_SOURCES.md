# 활용 데이터 원본 (소방안전 빅데이터 플랫폼)

플랫폼: https://www.bigdata-119.kr
전부 **무료** 상품이지만 다운로드에는 **로그인 + 주문(무료 결제) 절차**가 필요하다.
비로그인 상태에서 받을 수 있는 것은 컬럼 스키마 확인용 **샘플(.xls)** 뿐이다.

## 다운로드 방법
1. https://www.bigdata-119.kr/login/login 로그인
2. 아래 상품 페이지 접속 → `데이터 선택` 체크 → `구매하기`(무료) → 주문 완료
3. `마이페이지 > 주문내역`에서 CSV 다운로드
4. 받은 CSV를 `data/raw/ulsan/`, `data/raw/sejong/` 아래에 그대로 둔다 (파일명 변경 불필요)

## 1차 데이터 — 울산광역시소방본부 (모델 학습·검증)

| 용도 | 상품명 | 용량 | 링크 |
|---|---|---|---|
| 라벨(화재 건수) | 울산광역시소방본부_화재발생현황 데이터셋 | 6.02 MB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=14 |
| 피처(대상물 용도구성) | 울산광역시소방본부_특정소방대상물 현황 데이터셋 | 9.56 MB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=12 |
| 피처(업소 업종구성) | 울산광역시소방본부_다중이용업소 현황 데이터셋 | 1.21 MB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=13 |
| 대응취약(소화전) | 울산광역시소방본부_소방용수시설운영현황 데이터셋 | 910 KB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=16 |
| 보조(점검 이력) | 울산광역시소방본부_화재안전특별조사 데이터셋 | 46.33 MB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=23 |

## 2차 데이터 — 세종특별자치시소방본부 (도시 이식 검증)

| 용도 | 상품명 | 용량 | 링크 |
|---|---|---|---|
| 라벨 | 세종특별자치시소방본부_화재발생현황 데이터셋 | 1.02 MB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=27 |
| 피처 | 세종특별자치시소방본부_특정소방대상물 현황 데이터셋 | 3.20 MB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=25 |
| 피처 | 세종특별자치시소방본부_다중이용업소 현황 데이터셋 | 333 KB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=26 |
| 대응취약(선택) | 세종특별자치시소방본부_소방용수시설운영현황 데이터셋 | 383 KB | https://www.bigdata-119.kr/goods/goodsInfo?goods_mng_sn=29 |

## 외부 API
| 용도 | 발급처 | 비고 |
|---|---|---|
| 도로명 → 좌표 지오코딩 | 카카오 개발자 https://developers.kakao.com | REST API 키. 일 30만건 무료 |
| 계절·기상 보조 | 기상청 API 허브 https://apihub.kma.go.kr | 선택 |

키는 저장소에 넣지 않는다. `.env` 에 `KAKAO_REST_API_KEY=...` 로 둔다 (.gitignore 처리됨).


## 소방 법령 및 법정 서식 (국가법령정보센터)

`scripts/` 실행 시 자동으로 수집·캐시된다. 별도 신청 없이 `OC=test` 로 조회된다.

| 항목 | 수량 | 저장 위치 |
|---|---|---|
| 법령 조문 | 7개 법령 457개 조문 | `data/cache/law_articles.parquet` |
| 별표·서식 목록 | 68건 | `data/cache/law_forms.parquet` |
| 법정 서식 파일(HWP) | 39건 | `data/processed/forms/` |

수집 명령:

```python
from firebird.config import load_config
from firebird import lawdata as LW
cfg = load_config()
LW.collect(cfg)                                   # 조문
forms = LW.collect_forms(cfg)                     # 별표·서식 목록
LW.download_forms(cfg, forms)                     # HWP 내려받기
```

주의: 본문 조회는 `MST`(법령일련번호) 파라미터로만 된다. 검색 결과의 `ID` 를 그대로
넣으면 "일치하는 법령이 없습니다"가 돌아온다. 또 검색어에 공백이 있으면 다른 법령이
1순위로 오므로 공백을 제거해 질의한다.
