#!/usr/bin/env python
"""법정 서식 원본을 내려받아 이미지로 만든다 — 화면·발표자료 대조용.

'양식대로 나온다'는 말은 빈 양식과 채워진 문서를 나란히 놓고 보여 줘야
증명이 된다. 원본은 법제처 것을 그대로 쓰고, 우리가 만든 문서는 그 위에
얹지 않는다 — 둘을 따로 두고 비교하게 한다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firebird.config import load_config      # noqa: E402
from firebird import lawdata as LW           # noqa: E402

#: 대조에 쓰는 서식. 우리 자료로 실제 칸을 채울 수 있는 것만 고른다.
WANTED = {"11": "화재예방강화지구 관리대장"}


def main() -> int:
    cfg = load_config()
    out = cfg.paths.processed / "forms"
    out.mkdir(parents=True, exist_ok=True)

    forms = LW.collect_forms(cfg)
    if forms.empty:
        print("서식 목록을 가져오지 못했습니다.")
        return 1

    made = 0
    for no, title in WANTED.items():
        hit = forms[(forms["kind"] == "서식") & (forms["no"] == no)]
        if hit.empty or not hit.iloc[0]["pdf_url"]:
            print(f"  서식{no} PDF 링크 없음 — 건너뜀")
            continue
        r = hit.iloc[0]
        pdf = out / f"서식{no}_{title}.pdf"
        try:
            resp = requests.get(r["pdf_url"], timeout=90)
            resp.raise_for_status()
            pdf.write_bytes(resp.content)
        except requests.RequestException as exc:
            print(f"  서식{no} 내려받기 실패: {exc}")
            continue

        stem = out / f"서식{no}_빈양식"
        try:
            subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1",
                            str(pdf), str(stem)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  서식{no} PDF→PNG 변환 실패 (poppler-utils 필요)")
            continue
        # pdftoppm 은 접미사(-1)를 붙인다. 화면에서 찾기 쉽게 이름을 고정한다.
        for p in out.glob(f"서식{no}_빈양식*.png"):
            p.rename(out / f"서식{no}_빈양식.png")
        print(f"  서식{no} {title} → {out / f'서식{no}_빈양식.png'}")
        made += 1

    print(f"완료: {made}건")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
