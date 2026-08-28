#!/usr/bin/env python
"""읍면동별 건축물대장 수집 — 연도별 노후도 피처의 재료.

**왜 필요한가**: 지금 쓰는 대상물·업소·소화전은 전부 현재 시점 스냅샷이라
2014년 행에도 2021년 값이 들어간다. 데이터로는 없앨 수 없는 한계라고
적어 두었는데, 건축물대장에는 **사용승인일**이 있다. 사용승인일이 t년 이전인
건물만 세면 't년에 실제로 서 있던 건물'이 되고, 그때부터는 스냅샷이 아니라
시계열이다.

**왜 읍면동 단위인가**: 격자에 붙이려면 건물마다 주소를 좌표로 바꿔야 하는데
울산 전체가 25만 동 규모다. 무료 지오코딩 한도와 시간을 넘는다. 읍면동 단위는
지오코딩이 필요 없고(대장이 이미 법정동별로 온다), 연도 변화라는 핵심 성질은
그대로 남는다. 격자 단위는 상세주소가 확보되면 그때 간다.

한 번 받은 법정동은 다시 받지 않는다. 중간에 끊겨도 이어서 돌리면 된다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                              # noqa: E402

from firebird import buildings as BD             # noqa: E402
from firebird import dataset as D                # noqa: E402
from firebird import geocode as GC               # noqa: E402
from firebird import grid as G                   # noqa: E402
from firebird.config import load_config          # noqa: E402

CODE_CACHE = "bjdong_codes.parquet"


def region_codes(cfg, cur: pd.DataFrame, *, refresh: bool = False) -> pd.DataFrame:
    """읍면동마다 대표 격자 하나를 골라 법정동코드를 얻는다.

    읍면동당 한 번만 물으면 되므로 83번이면 끝난다. 격자마다 물으면 1,459번이다.
    """
    cache = cfg.paths.cache / CODE_CACHE
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    kakao = GC.load_api_key()
    if not kakao:
        print("카카오 키가 없어 법정동코드를 못 얻습니다.")
        return pd.DataFrame()

    rows = []
    # 읍면동마다 대상물이 가장 많은 격자를 대표로 쓴다 — 그 동의 중심일 가능성이 높다.
    reps = (cur.sort_values("target_total", ascending=False)
              .drop_duplicates(subset=["sgg", "emd"]))
    centers = G.grid_centers(reps["grid_id"], cfg).set_index("grid_id")
    for r in reps.itertuples():
        if r.grid_id not in centers.index:
            continue
        c = centers.loc[r.grid_id]
        rc = BD.region_code(float(c["lon"]), float(c["lat"]), kakao)
        if rc:
            rows.append({"sgg": r.sgg, "emd": r.emd,
                         "sigungu_cd": rc[0], "bjdong_cd": rc[1]})
        time.sleep(0.12)
    out = pd.DataFrame(rows).drop_duplicates(subset=["sigungu_cd", "bjdong_cd"])
    if not out.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


def main() -> int:
    cfg = load_config()
    city = sys.argv[1] if len(sys.argv) > 1 else "ulsan"
    key = BD.load_key()
    if not key:
        print("DATA_GO_KR_KEY 가 없습니다. .env 를 확인하십시오.")
        return 1

    panel = D.load_panel(cfg, city)
    cur = panel[panel["year"] == int(cfg.holdout_year)]
    codes = region_codes(cfg, cur)
    if codes.empty:
        print("법정동코드를 얻지 못했습니다.")
        return 1
    print(f"법정동 {len(codes)}개")

    done = fail = 0
    for i, r in enumerate(codes.itertuples(), 1):
        cache = (cfg.paths.cache / BD.CACHE_NAME
                 / f"{r.sigungu_cd}_{r.bjdong_cd}.parquet")
        if cache.exists():
            done += 1
            continue
        t = time.time()
        df = BD.fetch_title(r.sigungu_cd, r.bjdong_cd, key, cfg=cfg)
        ok = cache.exists()
        done += ok
        fail += (not ok)
        print(f"  [{i:3d}/{len(codes)}] {r.sgg} {r.emd:8s} "
              f"{len(df):6,}건 {'저장' if ok else '미완(다시 시도 필요)'} "
              f"({time.time() - t:.0f}초)", flush=True)

    print(f"\n완료 {done}/{len(codes)} · 미완 {fail}")
    print("미완이 있으면 같은 명령을 다시 실행하면 이어 받습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
