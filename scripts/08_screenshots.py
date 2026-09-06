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


def paint_through(page, selector: str, *, step: int = 700) -> None:
    """요소를 끝까지 훑어 내려 브라우저가 실제로 그리게 한다.

    요소 캡처의 높이는 문서 전체와 맞는데 아래쪽이 백지로 나온 적이 있다
    (7,356px 중 2,799px 뒤가 전부 비었다). 화면 밖은 아직 칠해지지 않은
    상태였고, 캡처는 칠해진 것만 가져간다. 한 번 훑어 주면 채워진다.
    """
    # window.scrollTo 는 Streamlit 에서 듣지 않는다 — 본문은 내부 컨테이너가
    # 스크롤한다. 휠은 커서 아래의 스크롤 주체에 그대로 간다.
    h = page.evaluate(
        "sel => { const d = document.querySelector(sel);"
        " return d ? d.scrollHeight : 0; }", selector) or 0
    page.mouse.move(page.viewport_size["width"] // 2,
                    page.viewport_size["height"] // 2)
    for _ in range(int(h / step) + 4):
        page.mouse.wheel(0, step)
        page.wait_for_timeout(140)
    page.wait_for_timeout(600)
    try:
        page.locator(selector).first.scroll_into_view_if_needed(timeout=10_000)
    except Exception:                                     # noqa: BLE001
        pass
    page.wait_for_timeout(900)


def trim_blank(path: Path, *, pad: int = 24) -> None:
    """요소 캡처 아래에 붙은 백지를 잘라낸다.

    .docview 를 요소로 찍으면 컨테이너 높이만큼 찍혀 문서가 끝난 뒤로
    빈 공간이 길게 붙는다. 실제로 7,356px 중 2,799px 뒤가 전부 백지였고,
    장표에서는 이걸 keep 비율로 눈대중해 잘라 쓰고 있었다. 여기서 자른다.
    """
    try:
        from PIL import Image
        import numpy as np
    except Exception:                                     # noqa: BLE001
        return
    im = Image.open(path)
    a = np.asarray(im.convert("L"))
    ink = np.nonzero((a < 200).sum(axis=1))[0]
    if not len(ink):
        return
    bottom = min(a.shape[0], int(ink[-1]) + pad)
    top = max(0, int(ink[0]) - pad)
    if bottom - top < a.shape[0] * 0.98:
        im.crop((0, top, im.width, bottom)).save(path)


def main() -> int:
    cfg = load_config()
    out = cfg.paths.figures
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1400},
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
            # 문서는 화면 아래에 나온다. 화면 위쪽을 찍으면 입력 폼만 잡히고
            # '무엇이 나오는지'가 안 보인다. 미리보기 제목의 좌표에서 자른다.
            prev = page.get_by_text("미리보기").first
            prev.scroll_into_view_if_needed(timeout=15_000)
            page.wait_for_timeout(3_000)
            # 좌표를 계산해 자르려다 316px 짜리 조각을 얻은 적이 있다.
            # 문서 자체를 요소로 찍으면 좌표 계산이 필요 없다.
            doc = page.locator(".docview").first
            doc.wait_for(state="visible", timeout=20_000)
            # 결재란까지 그려졌는지 확인하고 찍는다. 기다리지 않고 찍었더니
            # 문서 앞 38%만 들어오고 나머지가 백지였다.
            try:
                doc.get_by_text("결재").first.wait_for(timeout=20_000)
            except Exception:                             # noqa: BLE001
                page.wait_for_timeout(5_000)
            paint_through(page, ".docview")
            # Streamlit 은 본문을 내부 컨테이너로 스크롤한다. 그래서 문서가
            # 화면보다 길면 한 장으로는 못 찍는다 — 요소 캡처는 칠해지지 않은
            # 쪽이 백지로 남고, full_page 도 화면 높이까지만 잡힌다.
            # 여기서는 훑어 내린 끝(붙임·끝.·발신명의·결재란이 있는 뒷부분)을
            # 온전히 찍는다. 문서 머리는 shot_plan_head.png 가 따로 맡는다.
            bb = doc.bounding_box()
            page.screenshot(path=str(out / "_full_page.png"), full_page=True)
            if bb:
                from PIL import Image as _Im
                sc = page.evaluate("() => window.devicePixelRatio") or 1
                fp = _Im.open(out / "_full_page.png")
                sy = page.evaluate("() => window.scrollY") or 0
                box = (int(bb["x"] * sc), int((bb["y"] + sy) * sc),
                       int((bb["x"] + bb["width"]) * sc),
                       int((bb["y"] + sy + bb["height"]) * sc))
                box = (max(0, box[0]), max(0, box[1]),
                       min(fp.width, box[2]), min(fp.height, box[3]))
                fp.crop(box).save(out / "shot_plan_result.png")
                fp.close()
            else:
                doc.screenshot(path=str(out / "shot_plan_result.png"))
            (out / "_full_page.png").unlink(missing_ok=True)
            trim_blank(out / "shot_plan_result.png")
            print("  shot_plan_result.png  (생성된 계획서 본문)")
            # 결재란은 문서 맨 끝이라 본문 캡처에 들어오지 않는다. 그런데
            # '이대로 결재가 됩니다' 의 증거는 바로 그 칸이다. 따로 찍는다.
            try:
                appr = doc.locator("table").last
                appr.scroll_into_view_if_needed(timeout=15_000)
                page.wait_for_timeout(1_200)
                appr.screenshot(path=str(out / "shot_plan_approval.png"))
                print("  shot_plan_approval.png  (결재란)")
            except Exception as exc:                      # noqa: BLE001
                print(f"  결재란 캡처 건너뜀: {type(exc).__name__}")
            # 제목·수신·시행일이 있는 머리도 따로. 장표에서 확대해 쓴다.
            try:
                head = doc.locator("p").first
                head.scroll_into_view_if_needed(timeout=10_000)
                page.wait_for_timeout(800)
                bb = doc.bounding_box()
                if bb:
                    page.screenshot(path=str(out / "shot_plan_head.png"),
                                    clip={"x": bb["x"], "y": max(bb["y"], 0),
                                          "width": bb["width"],
                                          "height": min(560, bb["height"])})
                    print("  shot_plan_head.png  (기관·수신·제목·시행일)")
            except Exception as exc:                      # noqa: BLE001
                print(f"  머리 캡처 건너뜀: {type(exc).__name__}")
        except Exception as exc:                          # noqa: BLE001
            print(f"  계획서 캡처 건너뜀: {type(exc).__name__}")

        # 순찰 동선 지도는 화면 아래쪽에 있다. 위에서 자른 화면에는 지도가
        # 손톱만 하게 들어가는데, 이 화면의 결과물은 지도다. 따로 찍는다.
        try:
            page.get_by_role("tab", name="예방순찰 계획").click(timeout=20_000)
            page.wait_for_timeout(12_000)
            h = page.get_by_text("관서별 순찰 구역").first
            h.scroll_into_view_if_needed(timeout=15_000)
            page.wait_for_timeout(6_000)
            box = h.bounding_box()
            top = max((box["y"] if box else 0) - 60, 0)
            page.screenshot(path=str(out / "shot_patrol_map.png"),
                            clip={"x": 360, "y": top, "width": 1320,
                                  "height": min(1050 - top, 900)})
            print("  shot_patrol_map.png  (동선 지도 중심)")
        except Exception as exc:                          # noqa: BLE001
            print(f"  동선 지도 캡처 건너뜀: {type(exc).__name__}")

        # 조건을 바꾸면 계획이 달라진다는 것은 전후를 나란히 놓아야 보인다.
        try:
            box_sel = page.locator("[data-testid='stMain']").get_by_role(
                "combobox").first
            box_sel.click(timeout=20_000)
            page.wait_for_timeout(1_500)
            opts = page.get_by_role("option")
            label = opts.nth(1).inner_text().strip()
            opts.nth(1).click()
            print(f"  순찰 목적 변경 → {label}, 재계산 대기")
            page.wait_for_timeout(35_000)
            # 지도 두 장만 찍는다. 좌표를 손으로 잘라 내면 지도가 중간에서
            # 끊긴다 — 화면이 길어지거나 짧아지면 그 값이 바로 틀어지기 때문이다.
            # 두 지도를 담고 있는 블록 자체를 찍으면 잘릴 일이 없다.
            block = page.locator("[data-testid='stHorizontalBlock']").filter(
                has_text="바꾸기 전").first
            block.scroll_into_view_if_needed(timeout=15_000)
            page.wait_for_timeout(8_000)          # 지도 타일이 다시 그려질 시간
            block.screenshot(path=str(out / "shot_patrol_compare.png"))
            print("  shot_patrol_compare.png  (전후 지도)")

            # 달라진 수치는 따로 찍어 둔다. 지도와 한 장에 넣으면 둘 다 작아진다.
            try:
                mblock = page.locator("[data-testid='stHorizontalBlock']").filter(
                    has_text="가장 먼 순찰조").first
                mblock.scroll_into_view_if_needed(timeout=10_000)
                page.wait_for_timeout(2_000)
                mblock.screenshot(path=str(out / "shot_patrol_delta.png"))
                print("  shot_patrol_delta.png  (전후 수치)")
            except Exception:                             # noqa: BLE001
                pass
        except Exception as exc:                          # noqa: BLE001
            print(f"  전후 비교 캡처 건너뜀: {type(exc).__name__}")

        # 법정 서식은 빈 양식과 채운 문서를 나란히 놓아야 뜻이 통한다.
        try:
            page.get_by_role("tab", name="순찰·점검 계획서").click(timeout=20_000)
            page.wait_for_timeout(6_000)
            page.get_by_text("법정 서식으로 내보내기").first.scroll_into_view_if_needed()
            page.wait_for_timeout(1_500)
            page.get_by_role("button", name="서식 채우기").click(timeout=20_000)
            print("  법정 서식 채우는 중…")
            page.wait_for_timeout(20_000)
            block = page.locator("[data-testid='stHorizontalBlock']").filter(
                has_text="별지 제11호서식").first
            block.scroll_into_view_if_needed(timeout=15_000)
            page.wait_for_timeout(5_000)
            block.screenshot(path=str(out / "shot_form_compare.png"))
            print("  shot_form_compare.png  (빈 양식 ↔ 채운 대장)")
        except Exception as exc:                          # noqa: BLE001
            print(f"  법정 서식 캡처 건너뜀: {type(exc).__name__}")

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
