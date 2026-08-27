#!/usr/bin/env python
"""지오코딩 캐시를 미리 채운다.

파이프라인에서 유일하게 외부 API 에 의존하고, 유일하게 느린 단계다.
따로 떼어 두면 이후 단계는 네트워크 없이 몇 번이든 다시 돌릴 수 있다.
중단해도 캐시는 남으므로 다시 실행하면 이어서 진행한다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from firebird import addresses, geocode  # noqa: E402
from firebird.config import load_config  # noqa: E402
from firebird.io_utils import load_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("geocode")


def collect_keys(cfg, city: str) -> list[str]:
    default_sido = cfg.city(city)["label"]
    keys: list[str] = []
    for kind in ("fire", "target", "business", "hydrant", "inspection"):
        try:
            df = load_dataset(cfg, city, kind)
        except (FileNotFoundError, KeyError) as exc:
            log.warning("[%s/%s] 건너뜀: %s", city, kind, str(exc).splitlines()[0])
            continue
        if "address" not in df.columns:
            continue
        keyed = addresses.add_address_columns(df, "address", default_sido)
        got = keyed["geo_key"]
        n_empty = int((got == "").sum())
        log.info("[%s/%s] %d행 -> 키 %d개(고유 %d), 도로 추출 실패 %d행",
                 city, kind, len(df), len(got) - n_empty, got[got != ""].nunique(), n_empty)
        keys += got[got != ""].tolist()
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", nargs="*", default=None)
    ap.add_argument("--dry", action="store_true", help="API 호출 없이 대상 건수만 센다")
    args = ap.parse_args()

    cfg = load_config()
    cities = args.city or list(cfg["cities"])

    all_keys: list[str] = []
    for city in cities:
        all_keys += collect_keys(cfg, city)

    unique = list(dict.fromkeys(all_keys))
    cache = geocode.GeocodeCache(cfg.paths.cache / cfg["geocode"]["cache_file"])
    todo = [k for k in unique if k not in cache]
    print(f"\n고유 지오코딩 키 {len(unique):,}개 · 캐시 적중 {len(unique)-len(todo):,}개 "
          f"· 신규 질의 {len(todo):,}건")

    if args.dry:
        print("--dry: 호출하지 않음")
        return 0
    if not todo:
        print("전부 캐시에 있다. 할 일 없음.")
        return 0
    if not geocode.load_api_key():
        print("!! KAKAO_REST_API_KEY 가 없다. .env 에 넣어라 (.env.example 참고).")
        return 1

    coords = geocode.geocode_keys(unique, cfg)
    rate = float(coords["matched"].mean()) if len(coords) else 0.0
    print(f"좌표 확보: {int(coords['matched'].sum()):,}/{len(coords):,} ({rate:.1%})")

    failed = coords[~coords["matched"]]["geo_key"]
    if len(failed):
        out = cfg.paths.outputs / "geocode_failed.csv"
        failed.to_frame().to_csv(out, index=False, encoding="utf-8-sig")
        print(f"실패 키 {len(failed):,}건 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
