#!/usr/bin/env python
"""원본 CSV -> 격자x연도 패널(parquet) + manifest(json)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firebird import dataset as D  # noqa: E402
from firebird.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", nargs="*", default=None)
    ap.add_argument("--no-api", action="store_true",
                    help="지오코딩 API 를 부르지 않고 캐시만 쓴다")
    args = ap.parse_args()

    cfg = load_config()
    cities = args.city or list(cfg["cities"])

    for city in cities:
        print(f"\n{'='*78}\n[{city}] 패널 구축")
        try:
            data = D.load_city(cfg, city, use_api=not args.no_api)
        except FileNotFoundError as exc:
            print(f"  건너뜀: {str(exc).splitlines()[0]}")
            continue
        saved = D.save_city(data, cfg)
        m = data.manifest
        print(json.dumps(m, ensure_ascii=False, indent=2)[:1800])
        print(f"  저장: {saved['panel']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
