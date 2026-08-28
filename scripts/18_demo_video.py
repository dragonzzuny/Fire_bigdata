#!/usr/bin/env python
"""시연 녹화 — 리허설·백업용.

리허설은 말이 아니라 손이 밀린다. 어느 버튼이 몇 초 걸리는지 몸으로 알아야
발표장에서 침묵이 생기지 않는다. 그래서 발표에서 밟을 순서 그대로 녹화한다.

부수 효과가 하나 더 있다. 발표장 네트워크가 막혀 지도가 안 뜨면 이 영상이
그대로 대체본이 된다.

전제: 대시보드가 localhost:8601 에 떠 있어야 한다.
    .venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8601
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from firebird.config import load_config  # noqa: E402

URL = "http://localhost:8601"

#: 발표에서 이 배속으로 말한다. 1.0 이면 실제 시연 속도.
#: 리허설용은 1.0, 확인만 할 때는 0.4 정도로 줄여 쓴다.
#: 지도 타일이 뜨는 데 시간이 걸린다. 배속을 낮추면 흰 화면이 찍힌다.
DEFAULT_PACE = 1.0


class Recorder:
    """장면 하나하나에 이름과 시간을 붙여 진행 상황을 찍어 준다."""

    def __init__(self, page, pace: float):
        self.page, self.pace = page, pace
        self.elapsed = 0.0

    def beat(self, seconds: float, label: str = "") -> None:
        ms = int(seconds * self.pace * 1000)
        self.page.wait_for_timeout(ms)
        self.elapsed += seconds * self.pace
        if label:
            print(f"  [{int(self.elapsed // 60)}:{int(self.elapsed % 60):02d}] {label}")

    def tab(self, name: str, settle: float, label: str) -> None:
        self.page.get_by_role("tab", name=name).click(timeout=25_000)
        self.settle(settle, label)

    def scroll(self, ratio: float, seconds: float = 1.5) -> None:
        """화면을 천천히 내린다. 뚝 끊기면 무엇을 보는지 알 수 없다."""
        steps = 12
        for i in range(steps):
            self.page.mouse.wheel(0, int(900 * ratio / steps))
            self.page.wait_for_timeout(int(seconds * self.pace * 1000 / steps))
        self.elapsed += seconds * self.pace

    def to(self, text: str, seconds: float, label: str = "") -> bool:
        """특정 문구가 보이는 곳으로 올린다/내린다.

        휠을 몇 번 굴릴지로 정하면 화면 길이가 조금만 달라져도 엉뚱한 곳을
        비춘다. 실제로 계획서를 만들기도 전에 지나가 버린 적이 있다.
        """
        try:
            self.page.get_by_text(text).first.scroll_into_view_if_needed(
                timeout=20_000)
        except Exception:                                 # noqa: BLE001
            print(f"  '{text}' 를 찾지 못해 그 장면은 건너뛴다")
            return False
        self.beat(seconds, label)
        return True

    def settle(self, extra: float = 0.0, label: str = "") -> None:
        """Streamlit 이 다 그릴 때까지 기다린 뒤 지도에 시간을 더 준다.

        재계산 중에는 pydeck 이 세계 지도로 돌아가 있다. 그 상태로 녹화되면
        울산 자료를 보여준다면서 유럽·아프리카가 나온다. 실제로 그랬다.
        """
        try:                                              # 실행 표시가 사라질 때까지
            self.page.locator("[data-testid='stStatusWidget']").wait_for(
                state="detached", timeout=90_000)
        except Exception:                                 # noqa: BLE001
            pass
        if extra:
            self.beat(extra, label)
        elif label:
            print(f"  [{int(self.elapsed // 60)}:{int(self.elapsed % 60):02d}] {label}")


def record(out_dir: Path, pace: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            # 세로가 짧으면 화면이 접혀 스크롤이 늘고, 지도가 자꾸 잘린다.
            viewport={"width": 1600, "height": 1000},
            record_video_dir=str(out_dir),
            record_video_size={"width": 1600, "height": 1000})
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        r = Recorder(page, pace)
        r.beat(6, "대시보드 로드")

        # --- 1. 예방점검 배분 -------------------------------------------
        r.tab("예방점검 배분", 6, "① 예방점검 배분 — 인력을 먼저 넣는다")
        r.to("위험도 상위", 8, "   위험 상위 20% 그대로 가면 1위 구역도 못 끝낸다")
        r.to("점검 순위표", 4)
        r.settle(10, "   같은 인력으로 갈 수 있는 조합을 고른 결과 (지도)")

        # --- 2. 순찰 동선 (핵심) ----------------------------------------
        r.tab("예방순찰 계획", 13, "② 순찰 동선 — 관서에서 출발해 관할을 돌고 복귀")
        if r.to("관서별 순찰 구역", 4):
            r.settle(14, "   관서별 색, 실제 도로 주행거리")
        r.scroll(0.7, 3)
        r.beat(6, "   관서별 순찰 구역표")

        # --- 3. 조건을 바꾸면 계획이 다시 짜인다 (핵심) -----------------
        try:
            r.to("순찰 목적", 2, "③ 순찰 목적을 바꾼다")
            sel = page.locator("[data-testid='stMain']").get_by_role("combobox").first
            sel.wait_for(state="visible", timeout=30_000)
            # 한 번의 클릭으로 목록이 안 열리는 경우가 있다(스크롤 직후 특히).
            opts = page.get_by_role("option")
            for attempt in range(4):
                sel.click(timeout=20_000)
                page.wait_for_timeout(1_800)
                if opts.count() > 1:
                    break
                page.keyboard.press("Escape")
                page.wait_for_timeout(600)
            else:
                raise TimeoutError("순찰 목적 목록이 열리지 않았다")
            label = opts.nth(1).inner_text().strip()
            opts.nth(1).click()
            # 재계산이 끝나기 전에는 지도가 세계 지도로 돌아가 있다. 기다린다.
            r.settle(6, f"   → {label} · 대상 구역과 동선이 다시 계산된다")
            if r.to("바꾸기 전", 4):
                r.settle(12, "   바꾸기 전 / 바꾼 뒤 — 같은 범위, 같은 배율")
            r.to("가장 먼 순찰조", 8, "   달라진 구역 수와 이동거리")
        except Exception as exc:                          # noqa: BLE001
            print(f"  목적 변경 장면 건너뜀: {type(exc).__name__} — {str(exc)[:120]}")

        # --- 4. 계획서 (결과물) -----------------------------------------
        r.tab("순찰·점검 계획서", 5, "④ 계획서 — 결재 올릴 문서를 만든다")
        try:
            page.get_by_text("AI로 문체 다듬기").click(timeout=10_000)
            r.beat(2, "   AI 문체 다듬기는 끈다(빠르고 결과가 일정하다)")
        except Exception:                                 # noqa: BLE001
            pass
        r.to("계획서 생성", 3)
        page.get_by_role("button", name="계획서 생성").click(timeout=25_000)
        r.settle(4, "   생성 완료")
        try:
            page.locator(".docview").first.scroll_into_view_if_needed(timeout=25_000)
            r.beat(6, "   공문 서식 그대로 나온 문서")
            r.scroll(2.2, 9)
            r.beat(5, "   관서별 순찰 구역·중점 확인사항·법령 근거")
        except Exception as exc:                          # noqa: BLE001
            print(f"  문서 장면 건너뜀: {type(exc).__name__}")

        # --- 5. 대응취약 · 업무 도우미 (밀리면 버리는 순서) --------------
        r.tab("대응취약 구역", 8, "⑤ 대응취약 — 고위험인데 소화전이 없는 구역")
        r.to("소방용수 사각지대", 10)
        r.tab("업무 도우미", 6, "⑥ 업무 도우미 — 법령을 조문 번호와 함께")
        try:
            # 자주 찾는 질문 첫 칸을 누른다. 문구가 바뀌어도 깨지지 않게
            # 이름이 아니라 자리로 집는다.
            page.get_by_text("자주 찾는 질문").first.scroll_into_view_if_needed(
                timeout=15_000)
            page.wait_for_timeout(1_500)
            btn = page.locator("[data-testid='stMain']").get_by_role(
                "button").filter(has_text="화재예방강화지구").first
            btn.click(timeout=15_000)
            # 자주 찾는 질문을 누르면 그대로 답변까지 실행된다(qa_run).
            # 답변은 15~40초 걸린다. 상태 표시만 보고 넘어가면 아직 계산 중인
            # 화면이 찍힌다. 결과가 실제로 뜰 때까지 기다린다.
            r.beat(6, "   자주 찾는 질문을 눌러 답을 받는다 (15~40초)")
            page.get_by_text("근거 자료").first.wait_for(state="visible",
                                                      timeout=90_000)
            r.to("근거 자료", 10, "   법령명과 조문 번호가 함께 나온다")
        except Exception as exc:                          # noqa: BLE001
            print(f"  질의응답 장면 건너뜀: {type(exc).__name__} — {str(exc)[:100]}")
            r.scroll(0.8, 3)
            r.beat(6)

        print(f"\n  총 길이 약 {int(r.elapsed // 60)}분 {int(r.elapsed % 60)}초")
        video = page.video
        ctx.close()          # 닫아야 파일이 완성된다
        browser.close()
        return Path(video.path())


def to_mp4(webm: Path, mp4: Path) -> bool:
    """발표용 노트북에서 바로 열리도록 mp4 로도 남긴다."""
    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm),
           "-c:v", "libx264", "-preset", "medium", "-crf", "24",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)]
    return subprocess.run(cmd, check=False).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pace", type=float, default=DEFAULT_PACE,
                    help="1.0 = 실제 시연 속도, 0.4 = 빠르게 확인만")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = cfg.paths.outputs / "demo_video"
    print(f"녹화 시작 — {URL}  (배속 {args.pace})")
    webm = record(out_dir, args.pace)

    final_webm = out_dir / "시연_불씨예보.webm"
    if webm != final_webm:
        webm.replace(final_webm)
    # 중간에 실패한 녹화 조각(page@....webm)이 남으면 어느 것이 최신인지
    # 헷갈린다. 이번에 만든 것만 남긴다.
    for stale in out_dir.glob("page@*.webm"):
        stale.unlink()
    print(f"\n저장: {final_webm}  ({final_webm.stat().st_size / 1e6:.1f} MB)")

    mp4 = out_dir / "시연_불씨예보.mp4"
    if to_mp4(final_webm, mp4):
        print(f"저장: {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")
    else:
        print("mp4 변환 건너뜀 (ffmpeg 없음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
