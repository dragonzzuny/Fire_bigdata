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

#: (파일명, 탭 이름, 캡처 전 대기 초, 사이드바 제외 여부)
#:
#: 장표에 넣을 때 사이드바까지 들어가면 본문 글씨가 읽히지 않는다.
#: 발표장 뒷자리에서 안 보이는 화면은 없는 것과 같으므로, 본문만 잘라 낸다.
SHOTS = [
    ("shot_allocation.png", "예방점검 배분", 7, True),
    ("shot_reason.png", "위험요인·점검계획서", 6, True),
    ("shot_patrol.png", "예방순찰 계획", 15, True),
    ("shot_plan_doc.png", "순찰·점검 계획서", 5, True),
    ("shot_hydrant.png", "대응취약 구역", 6, True),
    ("shot_validation.png", "모델 검증", 6, True),
    ("shot_assistant.png", "업무 도우미", 7, True),
    ("shot_full_patrol.png", "예방순찰 계획", 3, False),   # 전체 화면(사이드바 포함)
]


def main() -> int:
    cfg = load_config()
    out = cfg.paths.figures
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1050},
                                device_scale_factor=2)
        # networkidle 은 지도 타일이 계속 오가면 끝나지 않는다.
        # 문서만 뜨면 되므로 domcontentloaded 로 기다리고, 이후는 시간으로 준다.
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(20_000)          # 첫 로드는 패널·모델을 읽느라 느리다

        for name, tab, wait, crop in SHOTS:
            try:
                page.get_by_role("tab", name=tab).click(timeout=20_000)
            except Exception as exc:           # noqa: BLE001
                print(f"  건너뜀 {tab}: {type(exc).__name__}")
                continue
            page.wait_for_timeout(wait * 1000)
            if crop:
                # 본문 영역만. 사이드바를 빼면 같은 폭에 글씨가 1.4배로 커진다.
                main = page.locator("section.main, .main, [data-testid='stMain']").first
                try:
                    main.screenshot(path=str(out / name))
                except Exception:              # noqa: BLE001
                    page.screenshot(path=str(out / name),
                                    clip={"x": 360, "y": 0, "width": 1320, "height": 1050})
            else:
                page.screenshot(path=str(out / name))
            print(f"  {name}  ({tab})")

        # 계획서는 생성 버튼을 눌러야 문서가 보인다.
        # 입력 폼만 찍힌 화면은 '무엇이 나오는지'를 보여주지 못한다.
        try:
            page.get_by_role("tab", name="예방순찰 계획").click(timeout=20_000)
            page.wait_for_timeout(12_000)
            page.get_by_text("계획서용 지도").first.click(timeout=15_000)
            page.wait_for_timeout(1_500)
            page.get_by_role("button", name="지도 만들기").click(timeout=15_000)
            print("  동선도 생성 중…")
            page.wait_for_timeout(25_000)

            page.get_by_role("tab", name="순찰·점검 계획서").click(timeout=20_000)
            page.wait_for_timeout(5_000)
            # AI 다듬기를 끄고 표준 서식으로 — 빠르고 결과가 일정하다
            try:
                page.get_by_text("AI로 문체 다듬기").click(timeout=8_000)
                page.wait_for_timeout(2_000)
            except Exception:                             # noqa: BLE001
                pass
            page.get_by_role("button", name="계획서 생성").click(timeout=20_000)
            print("  계획서 생성 중…")
            page.wait_for_timeout(20_000)
            # 문서는 화면 아래에 나온다. 미리보기까지 내려가야 결과가 보인다.
            page.get_by_text("미리보기").first.scroll_into_view_if_needed(timeout=15_000)
            page.wait_for_timeout(2_500)
            page.screenshot(path=str(out / "shot_plan_result.png"),
                            clip={"x": 360, "y": 0, "width": 1320, "height": 1050})
            print("  shot_plan_result.png  (생성된 계획서 본문)")
        except Exception as exc:                          # noqa: BLE001
            print(f"  계획서 캡처 건너뜀: {type(exc).__name__}")

        # 업무 도우미는 질문을 실제로 눌러야 답변이 보인다.
        # 입력창만 찍힌 화면은 '무엇을 해 주는지'를 전혀 보여주지 못한다.
        try:
            # 예시 질문이 기본으로 채워져 있으므로 바로 누르면 된다.
            page.get_by_role("tab", name="업무 도우미").click(timeout=20_000)
            page.wait_for_timeout(5_000)
            page.get_by_role("button", name="질문하기").click(timeout=20_000)
            print("  업무 도우미 질의 실행 — 답변 대기(최대 90초)")
            page.wait_for_timeout(75_000)
            main = page.locator("section.main, .main, [data-testid='stMain']").first
            main.screenshot(path=str(out / "shot_assistant_answer.png"))
            print("  shot_assistant_answer.png  (답변 포함)")
        except Exception as exc:                          # noqa: BLE001
            print(f"  답변 캡처 건너뜀: {type(exc).__name__}")

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
