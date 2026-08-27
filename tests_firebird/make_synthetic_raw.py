"""합성 원본 CSV 생성기 — 파이프라인 스모크 테스트용.

**플랫폼 실제 배포본의 형태를 그대로 흉내낸다**: 영문 컬럼 코드, 조각난 주소,
`_0000`/`_2021` 두 벌, `_2021` 의 뭉개진 시각. 이 특성들이 파이프라인을
깨뜨리는 실제 원인이었으므로, 테스트 데이터가 그걸 재현하지 않으면
테스트가 통과해도 아무것도 보장하지 못한다.

실제 도시가 아니다. 성능 수치의 근거로 쓰면 안 되고, 오직
'스크립트가 처음부터 끝까지 도는가'를 확인하는 데만 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CITY = {
    "ulsan": dict(label="울산광역시", short="울산", lon0=129.30, lat0=35.50,
                  sggs=["남구", "중구", "동구", "북구", "울주군"]),
    "sejong": dict(label="세종특별자치시", short="세종", lon0=127.10, lat0=36.40,
                   sggs=[""]),          # 단층제: 시군구 없음
}
EMDS = ["삼산동", "달동", "무거동", "옥동", "성안동", "연암동", "방어동", "온산읍"]
ROADS = ["삼산로", "번영로", "태화로", "문수로", "산업로", "처용로"]
BIZ = ["일반음식점", "노래연습장업", "유흥주점", "단란주점",
       "인터넷컴퓨터게임시설제공업(PC방)", "고시원업", "스크린 골프연습장"]
USAGE = ["근린생활", "공동주택(아파트/기숙사)", "공장", "창고시설", "복합건축물",
         "노유자시설", "숙박시설", "업무시설"]
FACIL = ["일반대상물", "자동화재탐지설치대상", "옥내소화전설치대상",
         "스프링클러,물분무등설치대상", "11층이상"]
HYD = ["소화전(지상식)", "소화전(지하식)", "저수조", "급수탑"]


def _hour_weights():
    """오후·심야에 화재가 몰리게 한다 (실제 데이터의 시간대 패턴)."""
    w = np.ones(24)
    w[13:19] = 2.4
    w[0:4] = 1.6
    return w / w.sum()


def generate(city: str, raw_dir: Path, seed: int) -> dict:
    c = CITY[city]
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)

    combos = [(s, e, r) for s in c["sggs"] for e in EMDS for r in ROADS]
    coords, cache = {}, {}
    for i, (sgg, emd, road) in enumerate(combos):
        lon = c["lon0"] + (i % 25) * 0.009
        lat = c["lat0"] + (i // 25) * 0.006
        coords[(sgg, emd, road)] = (lon, lat)
        key_road = " ".join(x for x in [c["label"], sgg, road] if x)
        key_emd = " ".join(x for x in [c["label"], sgg, emd] if x)
        cache[key_road] = {"lon": lon, "lat": lat, "matched": True}
        cache[key_emd] = {"lon": lon + 0.002, "lat": lat + 0.001, "matched": True}
        cache[f"{c['label']} {emd}"] = {"lon": lon + 0.002, "lat": lat + 0.001, "matched": True}

    intensity = {k: float(rng.gamma(0.7, 0.8)) for k in combos}

    # ---- 화재: _0000 (2014~2020, 실제 시각) / _2021 (2021, 시각 000000) ----
    def fire_rows(years, with_time):
        rows = []
        for y in years:
            for k in combos:
                sgg, emd, road = k
                for _ in range(int(rng.poisson(0.1 + 1.1 * intensity[k]))):
                    hh = int(rng.choice(range(24), p=_hour_weights())) if with_time else 0
                    mi = int(rng.integers(0, 60)) if with_time else 0
                    # 도로명은 자주 빈다 — 실제 데이터의 핵심 특성
                    has_road = rng.random() < (0.35 if with_time else 0.8)
                    rows.append({
                        "SN": len(rows) + 1,
                        "RCPT_PATH_NM": "이동전화",
                        "RCPT_DT": f"{y}{rng.integers(1,13):02d}{rng.integers(1,29):02d}"
                                   f"{hh:02d}{mi:02d}00",
                        "EMRG_RSCU_CTPV_NM": c["short"],
                        "EMRG_RSCU_GUGUN_NM": sgg or None,
                        "EMRG_RSCU_EMD_NM": emd,
                        "LI_NM": None,
                        "ROAD_NM": road if has_road else None,
                        "EMRG_RSCU_KND_NM": "화재",
                        "EMRG_RSCU_CLSF_NM": rng.choice(["기타화재", "일반화재(주택)", "산불"]),
                        "PLCSCN_NM": f"{sgg or c['short']}소방서",
                        "CNTR_NM": f"{emd}119안전센터",
                    })
        return pd.DataFrame(rows)

    fire_dir = raw_dir
    fire_rows(range(2014, 2021), True).to_csv(
        fire_dir / f"{c['label']}소방본부_화재발생현황 데이터셋__화재발생_0000.csv",
        index=False, encoding="utf-8-sig")
    fire_rows([2021], False).to_csv(
        fire_dir / f"{c['label']}소방본부_화재발생현황 데이터셋__화재발생_2021.csv",
        index=False, encoding="utf-8-sig")

    # ---- 특정소방대상물 (스냅샷 두 벌: _0000 은 옛 스냅샷) ----
    def target_rows(scale):
        rows = []
        for k in combos:
            sgg, emd, road = k
            for i in range(int(rng.poisson(scale * (1 + 9 * intensity[k])))):
                rows.append({"SN": len(rows) + 1, "PLCSCN_NM": f"{sgg or c['short']}소방서",
                             "CNTR_NM": f"{emd}119안전센터",
                             "MUSES_NM": rng.choice(USAGE),
                             "CTPV_NM": c["short"], "GUGUN_NM": sgg or None,
                             "DONG_NM": emd, "ROAD_NM": road if rng.random() < 0.94 else None,
                             "TRGTOBJ_CLSF_NM": rng.choice(FACIL, p=[.7, .12, .08, .07, .03]),
                             "TRGTOBJ_NM": f"{road}대상물{i}"})
        return pd.DataFrame(rows)
    base = f"{c['label']}소방본부_특정소방대상물 현황 데이터셋__특정소방대상물"
    target_rows(0.9).to_csv(raw_dir / f"{base}_0000.csv", index=False, encoding="utf-8-sig")
    target_rows(1.0).to_csv(raw_dir / f"{base}_2021.csv", index=False, encoding="utf-8-sig")

    # ---- 다중이용업소 ----
    rows = []
    for k in combos:
        sgg, emd, road = k
        for i in range(int(rng.poisson(0.5 + 4 * intensity[k]))):
            rows.append({"SN": len(rows) + 1, "PLCSCN_NM": f"{sgg or c['short']}소방서",
                         "CNTR_NM": f"{emd}119안전센터", "BSSH_NM": f"{road}업소{i}",
                         "CTPV_NM": c["short"], "GUGUN_NM": sgg or None,
                         "DONG_NM": emd, "LI_NM": None, "ROAD_NM": road,
                         "TPBIZ_NM": rng.choice(BIZ)})
    pd.DataFrame(rows).to_csv(
        raw_dir / f"{c['label']}소방본부_다중이용업소 현황 데이터셋__다중이용업소_2021.csv",
        index=False, encoding="utf-8-sig")

    # ---- 소방용수시설 (세종은 좌표가 틀린 실제 상황을 재현) ----
    rows = []
    broken = (city == "sejong")
    for k in combos:
        sgg, emd, road = k
        if rng.random() < 0.45:
            continue
        lon, lat = coords[k]
        for _ in range(int(rng.integers(1, 4))):
            rows.append({"SN": len(rows) + 1, "PLCSCN_NM": f"{sgg or c['short']}소방서",
                         "CNTR_NM": f"{emd}119안전센터",
                         "FRUSWTR_NM": rng.choice(HYD, p=[.85, .08, .05, .02]),
                         "CTPV_NM": c["short"], "GUGUN_NM": sgg or None,
                         "DONG_NM": emd, "LI_NM": None,
                         "ROAD_NM": road if not broken else None,
                         # 세종: 경계 밖 좌표 -> bbox 검증에 걸려 버려져야 한다
                         "FRUSWTR_LAT": (lat - 1.0) if broken else lat + rng.normal(0, 0.001),
                         "FRUSWTR_LOT": (lon + 1.3) if broken else lon + rng.normal(0, 0.001)})
    pd.DataFrame(rows).to_csv(
        raw_dir / f"{c['label']}소방본부_소방용수시설운영현황 데이터셋__소방용수시설_2021.csv",
        index=False, encoding="utf-8-sig")
    return cache


def main(data_root: Path) -> None:
    cache: dict = {}
    for city in CITY:
        cache |= generate(city, data_root / "data/raw" / city, seed=11 if city == "ulsan" else 22)
    cpath = data_root / "data/cache/geocode_cache.json"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"합성 원본 생성 완료 -> {data_root}")
    print(f"지오코딩 캐시 {len(cache)}건 심음 (API 불필요)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
