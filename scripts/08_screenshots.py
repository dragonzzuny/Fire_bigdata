#!/usr/bin/env python
"""대시보드 실제 화면 캡처 — 발표자료용.

말로 설명한 기능과 화면에 실제로 있는 기능이 다르면 시연에서 바로 드러난다.
장표에 쓰는 그림은 지금 도는 화면을 그대로 찍은 것이어야 한다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from firebird.config import load_config  # noqa: E402

URL = "http://localhost:8601"

#: (파일명, 탭 이름, 캡처 전 대기 초). 탭 이름은 화면의 탭 라벨과 같아야 한다.
SHOTS = [
    ("shot_allocation.png", "예방점검 배분", 6),
    ("shot_reason.png", "위험요인·점검계획서", 5),
    ("shot_patrol.png", "예방순찰 계획", 14),
    ("shot_plan_doc.png", "순찰·점검 계획서", 4),
    ("shot_hydrant.png", "대응취약 구역", 5),
    ("shot_validation.png", "모델 검증", 5),
    ("shot_assistant.png", "업무 도우미", 6),
]


def main() -> int:
    cfg = load_config()
    out = cfg.paths.figures
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1050},
                                device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(12_000)          # 첫 로드는 패널·모델을 읽느라 느리다

        for name, tab, wait in SHOTS:
            try:
                page.get_by_role("tab", name=tab).click(timeout=20_000)
            except Exception as exc:           # noqa: BLE001
                print(f"  건너뜀 {tab}: {type(exc).__name__}")
                continue
            page.wait_for_timeout(wait * 1000)
            page.screenshot(path=str(out / name))
            print(f"  {name}  ({tab})")

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
