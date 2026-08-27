"""합성 원본 CSV 생성기 — 파이프라인 스모크 테스트용.

실제 도시가 아니다. 성능 수치의 근거로 쓰면 안 되고, 오직
'스크립트 4개가 처음부터 끝까지 도는가'를 확인하는 데만 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SGGS = {"ulsan": ["남구", "중구", "동구", "북구", "울주군"],
        "sejong": ["세종시"]}
ROADS = ["삼산로", "번영로", "태화로", "왕생로", "문수로", "옥동로", "산업로", "학성로"]
BIZ = ["일반음식점", "노래연습장", "유흥주점", "인터넷컴퓨터게임시설제공업", "고시원", "학원"]
GRADES = ["특급", "1급", "2급", "3급"]


def coord_table(city: str, lon0: float, lat0: float) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    for i, s in enumerate(SGGS[city]):
        for j, r in enumerate(ROADS):
            k = i * len(ROADS) + j
            out[(s, r)] = (lon0 + k * 0.008, lat0 + k * 0.0055)
    return out


def generate(city: str, raw_dir: Path, label: str, lon0: float, lat0: float,
             seed: int, n_fire: int) -> dict:
    rng = np.random.default_rng(seed)
    coords = coord_table(city, lon0, lat0)
    combos = list(coords)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 잠재 위험도: 격자마다 다르게. 화재/대상물/업소가 모두 여기에 연동된다.
    intensity = {c: float(rng.gamma(0.7, 0.8)) for c in combos}

    fires = []
    for year in range(2014, 2022):
        for c in combos:
            lam = 0.15 + 1.6 * intensity[c]
            for _ in range(int(rng.poisson(lam))):
                sgg, road = c
                hour = int(rng.choice(range(24), p=_hour_weights()))
                fires.append({
                    "화재발생일시": f"{year}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d} "
                                 f"{hour:02d}:{rng.integers(0,60):02d}:00",
                    "도로명주소": f"{label} {sgg} {road} {rng.integers(1,400)}",
                    "시군구": sgg,
                    "화재종별": rng.choice(["건축·구조물", "차량", "임야", "기타"]),
                })
    fires = fires[:n_fire] if n_fire and len(fires) > n_fire else fires
    pd.DataFrame(fires).to_csv(raw_dir / f"{label}소방본부_화재발생현황.csv",
                               index=False, encoding="cp949")

    rows = []
    for c in combos:
        sgg, road = c
        for i in range(int(rng.poisson(2 + 14 * intensity[c]))):
            rows.append({"대상물명": f"{road}대상물{i}",
                         "소재지도로명주소": f"{label} {sgg} {road} {rng.integers(1,400)}",
                         "시군구명": sgg,
                         "특정소방대상물등급": rng.choice(GRADES, p=[.05, .15, .4, .4])})
    pd.DataFrame(rows).to_csv(raw_dir / f"{label}소방본부_특정소방대상물현황.csv",
                              index=False, encoding="utf-8-sig")

    rows = []
    for c in combos:
        sgg, road = c
        for i in range(int(rng.poisson(1 + 7 * intensity[c]))):
            rows.append({"업소명": f"{road}업소{i}",
                         "영업장주소": f"{label} {sgg} {road} {rng.integers(1,400)}",
                         "관할구역": sgg, "영업의종류": rng.choice(BIZ)})
    pd.DataFrame(rows).to_csv(raw_dir / f"{label}소방본부_다중이용업소현황.csv",
                              index=False, encoding="cp949")

    rows = []
    for c in combos:
        sgg, road = c
        if rng.random() < 0.55:                       # 절반쯤은 소화전이 없다
            continue
        for _ in range(int(rng.integers(1, 5))):
            rows.append({"설치장소": f"{label} {sgg} {road} {rng.integers(1,400)}",
                         "소방용수시설구분": rng.choice(["소화전", "저수조", "급수탑"]),
                         "시군구": sgg})
    pd.DataFrame(rows).to_csv(raw_dir / f"{label}소방본부_소방용수시설운영현황.csv",
                              index=False, encoding="cp949")
    return {f"{label} {s} {r}": {"lon": lo, "lat": la, "matched": True}
            for (s, r), (lo, la) in coords.items()}


def _hour_weights():
    """오후·심야에 화재가 몰리게 한다 (기획서의 시간대 패턴 재현)."""
    w = np.ones(24)
    w[13:19] = 2.4
    w[0:4] = 1.8
    return w / w.sum()


def main(data_root: Path) -> None:
    cache = {}
    cache |= generate("ulsan", data_root / "data/raw/ulsan", "울산광역시",
                      129.30, 35.50, seed=11, n_fire=0)
    cache |= generate("sejong", data_root / "data/raw/sejong", "세종특별자치시",
                      127.25, 36.45, seed=22, n_fire=0)
    cpath = data_root / "data/cache/geocode_cache.json"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"합성 원본 생성 완료 -> {data_root}")
    print(f"지오코딩 캐시 {len(cache)}건 심음 (API 불필요)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
