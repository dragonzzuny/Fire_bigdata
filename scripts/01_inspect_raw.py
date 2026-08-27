#!/usr/bin/env python
"""원본 CSV 진단: 무엇이 있고, 어떤 컬럼이 매칭되고, 무엇이 빈다.

CSV 를 data/raw/ 에 넣은 직후 가장 먼저 돌린다.
매칭 실패한 컬럼은 configs/schema.yaml 에 실제 컬럼명을 한 줄 추가하면 끝난다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from firebird.config import load_config  # noqa: E402
from firebird.io_utils import find_files, read_csv_any, resolve_columns  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def inspect_city(cfg, city: str, n_preview: int) -> int:
    raw_dir = cfg.raw_dir(city)
    print(f"\n{'='*78}\n[{city}] {cfg.city(city)['label']}  ->  {raw_dir}")
    if not raw_dir.exists():
        print(f"  !! 폴더가 없다. 만들고 CSV 를 넣어라: {raw_dir}")
        return 1

    all_csv = sorted(raw_dir.rglob("*.csv")) + sorted(raw_dir.rglob("*.CSV"))
    print(f"  CSV 파일 {len(all_csv)}개")
    for p in all_csv:
        print(f"    - {p.relative_to(raw_dir)}  ({p.stat().st_size/1e6:.1f} MB)")

    problems = 0
    for kind, spec in cfg.schema["datasets"].items():
        files = find_files(raw_dir, spec.get("file_glob", []))
        print(f"\n  [{kind}] 패턴 {spec.get('file_glob')} -> {len(files)}개 매칭")
        if not files:
            required = bool(spec.get("required"))
            print(f"      {'!! 필수 데이터셋을 못 찾음' if required else '(선택 데이터셋 — 없어도 됨)'}")
            problems += int(required)
            continue

        for path in files:
            print(f"      · {path.name}")
            try:
                df = read_csv_any(path, nrows=200)
            except Exception as exc:                       # noqa: BLE001
                print(f"          !! 읽기 실패: {exc}")
                problems += 1
                continue
            print(f"          컬럼 {len(df.columns)}개: {list(df.columns)}")

            for label, aliases in (("필수", spec.get("required", {})),
                                   ("선택", spec.get("optional", {}))):
                if not aliases:
                    continue
                try:
                    got = resolve_columns(df, aliases, required=False, context=path.name)
                except Exception as exc:                   # noqa: BLE001
                    print(f"          !! {label} 해석 오류: {exc}")
                    problems += 1
                    continue
                missing = [k for k in aliases if k not in got]
                for std, actual in got.items():
                    print(f"          {label} {std:<14} <- '{actual}'")
                for std in missing:
                    mark = "!!" if label == "필수" else "  "
                    print(f"          {mark} {label} {std:<14} <- (못 찾음) "
                          f"후보: {aliases[std][:4]}")
                    problems += int(label == "필수")

            if n_preview:
                with pd.option_context("display.max_columns", None, "display.width", 200):
                    print(df.head(n_preview).to_string(max_colwidth=22))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", nargs="*", default=None, help="기본: 설정의 모든 도시")
    ap.add_argument("--preview", type=int, default=0, help="상위 N행 미리보기")
    args = ap.parse_args()

    cfg = load_config()
    cities = args.city or list(cfg["cities"])
    problems = sum(inspect_city(cfg, c, args.preview) for c in cities)

    print(f"\n{'='*78}")
    if problems:
        print(f"필수 항목 문제 {problems}건. configs/schema.yaml 의 후보 목록에 "
              f"위에서 출력된 실제 컬럼명을 추가하라.")
    else:
        print("모든 필수 컬럼이 해석된다. 다음: python scripts/03_build_dataset.py")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
