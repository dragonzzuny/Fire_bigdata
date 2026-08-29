#!/usr/bin/env python
"""시연 녹화 — 리허설용 전체본과, 핵심만 추린 소개본.

리허설은 말이 아니라 손이 밀린다. 어느 버튼이 몇 초 걸리는지 몸으로 알아야
발표장에서 침묵이 생기지 않는다. 그래서 발표에서 밟을 순서 그대로 녹화한다.

두 벌이 나온다.
  · 전체본  — 여섯 화면을 끊지 않고. 리허설용이자 시연이 막혔을 때의 대체본
  · 핵심본  — 순찰 동선과 계획서만 골라 제목 카드로 이어 붙인 소개용

자막은 녹화하면서 찍은 실제 시각으로 만든다. 손으로 맞추면 화면이 조금만
느려져도 어긋난다.

전제: 대시보드가 localhost:8601 에 떠 있어야 한다.
    .venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8601
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from firebird.config import load_config  # noqa: E402

URL = "http://localhost:8601"

#: 발표에서 이 배속으로 말한다. 1.0 이면 실제 시연 속도.
#: 지도 타일이 뜨는 데 시간이 걸린다. 배속을 낮추면 흰 화면이 찍힌다.
DEFAULT_PACE = 1.0

W, H = 1600, 1000
#: 제목 카드용 한글 글꼴. 없는 장비에서도 돌도록 후보를 여러 개 둔다.
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
)

#: 핵심본 맨 앞 카드.
CARD_LEAD = ("불씨예보  K-FIREBIRD", "화재예방 점검·순찰 의사결정 시스템")


@dataclass
class Mark:
    """자막 한 줄과 그 시각. key 면 핵심본에도 들어간다."""
    t: float
    text: str
    key: bool = False
    card: str = ""          # 핵심본에서 이 장면 앞에 세울 제목
    fast: float = 1.0       # 이 장면부터 다음 장면까지의 배속
    endkey: bool = False    # 여기서부터는 핵심본에 넣지 않는다


@dataclass
class Recorder:
    """장면마다 실제 시각을 찍어 둔다. 자막과 편집이 그 시각을 쓴다."""
    page: object
    pace: float
    t0: float = field(default_factory=time.monotonic)
    marks: list = field(default_factory=list)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def mark(self, text: str, key: bool = False, card: str = "",
             fast: float = 1.0, endkey: bool = False) -> None:
        t = self.now()
        self.marks.append(Mark(t, text.strip(), key, card, fast, endkey))
        tag = f"  ({fast:g}배속)" if fast != 1.0 else ""
        print(f"  [{int(t // 60)}:{int(t % 60):02d}] {text}{tag}")

    def beat(self, seconds: float, text: str = "", **kw) -> None:
        self.page.wait_for_timeout(int(seconds * self.pace * 1000))
        if text:
            self.mark(text, **kw)

    def tab(self, name: str, settle: float, text: str, **kw) -> None:
        self.page.get_by_role("tab", name=name).click(timeout=25_000)
        self.settle(settle, text, **kw)

    def scroll(self, ratio: float, seconds: float = 1.5) -> None:
        """화면을 천천히 내린다. 뚝 끊기면 무엇을 보는지 알 수 없다."""
        steps = 12
        for _ in range(steps):
            self.page.mouse.wheel(0, int(900 * ratio / steps))
            self.page.wait_for_timeout(int(seconds * self.pace * 1000 / steps))

    def to(self, needle: str, seconds: float, text: str = "", **kw) -> bool:
        """특정 문구가 보이는 곳으로 옮긴다.

        휠을 몇 번 굴릴지로 정하면 화면 길이가 조금만 달라져도 엉뚱한 곳을
        비춘다. 실제로 계획서를 만들기도 전에 지나가 버린 적이 있다.
        """
        try:
            self.page.get_by_text(needle).first.scroll_into_view_if_needed(
                timeout=20_000)
        except Exception:                                 # noqa: BLE001
            print(f"  '{needle}' 를 찾지 못해 그 장면은 건너뛴다")
            return False
        self.beat(seconds, text, **kw)
        return True

    def settle(self, extra: float = 0.0, text: str = "", **kw) -> None:
        """Streamlit 이 다 그릴 때까지 기다린 뒤 지도에 시간을 더 준다.

        재계산 중에는 pydeck 이 세계 지도로 돌아가 있다. 그 상태로 녹화되면
        울산 자료를 보여준다면서 유럽·아프리카가 나온다. 실제로 그랬다.
        """
        try:
            self.page.locator("[data-testid='stStatusWidget']").wait_for(
                state="detached", timeout=90_000)
        except Exception:                                 # noqa: BLE001
            pass
        self.beat(extra, text, **kw)


# ------------------------------------------------------------------ 녹화

def play(page, pace: float) -> list:
    r = Recorder(page, pace)
    r.beat(6, "실제 울산 자료로 지금 도는 화면입니다")

    # --- 1. 예방점검 배분 -----------------------------------------------
    r.tab("예방점검 배분", 6, "점검 가용 인력을 먼저 넣습니다",
          key=True, card="인력에 맞춘 예방점검 배분")
    r.to("위험도 상위", 8,
         "위험한 곳일수록 점검할 건물이 많습니다. 1위 구역 한 곳도 못 끝냅니다")
    r.to("점검 순위표", 4,
         "그래서 갈 수 있는 조합 중 화재를 가장 많이 잡는 쪽을 고릅니다")
    r.settle(10, "같은 인력으로 186개 구역, 실제 화재 17.8% 포착")

    # --- 2. 순찰 동선 (핵심) --------------------------------------------
    r.tab("예방순찰 계획", 13, "119안전센터에서 출발해 관할을 돌고 복귀합니다",
          key=True, card="관서별 순찰 동선")
    if r.to("관서별 순찰 구역", 4):
        r.settle(14, "색깔이 관서, 검은 점이 출동 관서. 선은 실제 도로 주행거리입니다")
    r.scroll(0.7, 3)
    r.beat(6, "관서마다 몇 구역을 몇 km 도는지 표로 나옵니다")

    # --- 3. 조건을 바꾸면 계획이 다시 짜인다 (핵심) ----------------------
    try:
        r.to("순찰 목적", 2, "순찰 목적을 바꿔 보겠습니다",
             key=True, card="조건을 바꾸면 계획이 다시 짜입니다")
        sel = page.locator("[data-testid='stMain']").get_by_role("combobox").first
        sel.wait_for(state="visible", timeout=30_000)
        # 한 번의 클릭으로 목록이 안 열리는 경우가 있다(스크롤 직후 특히).
        opts = page.get_by_role("option")
        for _ in range(4):
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
        r.mark(f"{label}로 바꾸면 대상 구역도 시간대도 다시 계산됩니다", fast=3.0)
        # 재계산이 끝나기 전에는 지도가 세계 지도로 돌아가 있다. 기다린다.
        r.settle(6, "다시 계산한 결과입니다")
        if r.to("바꾸기 전", 4):
            r.settle(12, "바꾸기 전과 바꾼 뒤를 같은 범위·같은 배율로 남깁니다")
        r.to("가장 먼 순찰조", 8, "빠진 구역과 새로 들어온 구역까지 적어 둡니다")
    except Exception as exc:                              # noqa: BLE001
        print(f"  목적 변경 장면 건너뜀: {type(exc).__name__} — {str(exc)[:120]}")

    # --- 4. 계획서 (핵심) ------------------------------------------------
    r.tab("순찰·점검 계획서", 5, "이 동선을 그대로 계획서로 만듭니다",
          key=True, card="결재 올릴 순찰 계획서")
    try:
        page.get_by_text("AI로 문체 다듬기").click(timeout=10_000)
        r.beat(2, "숫자와 법령은 시스템이 확정하고, AI는 문장만 다듬습니다")
    except Exception:                                     # noqa: BLE001
        pass
    r.to("계획서 생성", 3)
    page.get_by_role("button", name="계획서 생성").click(timeout=25_000)
    r.mark("계획서를 만드는 중입니다 (약 25초)", fast=5.0)
    r.settle(4, "만들어진 문서입니다")
    try:
        page.locator(".docview").first.scroll_into_view_if_needed(timeout=25_000)
        r.beat(6, "기관·수신·경유·시행일·관련 근거까지 공문 서식 그대로입니다")
        # 많이 내리면 문서를 지나쳐 아래 입력 폼이 나온다. 문서 안에서만 움직인다.
        r.mark("관서별 순찰 구역과 중점 확인사항이 이어집니다")
        r.scroll(1.1, 7)
        r.beat(3)
        r.mark("끝에 붙임과 결재란까지 들어갑니다")
        r.scroll(1.1, 7)
        r.beat(4)
    except Exception as exc:                              # noqa: BLE001
        print(f"  문서 장면 건너뜀: {type(exc).__name__}")

    # --- 5. 대응취약 · 업무 도우미 (전체본에만) --------------------------
    r.tab("대응취약 구역", 8, "고위험인데 소화전이 없는 구역을 따로 뽑습니다",
          endkey=True)
    r.to("소방용수 사각지대", 10, "소화전 신설 우선순위의 객관적 근거가 됩니다")
    r.tab("업무 도우미", 6, "법령은 조문 번호와 함께 답합니다")
    try:
        page.get_by_text("자주 찾는 질문").first.scroll_into_view_if_needed(
            timeout=15_000)
        page.wait_for_timeout(1_500)
        btn = page.locator("[data-testid='stMain']").get_by_role(
            "button").filter(has_text="화재예방강화지구").first
        btn.click(timeout=15_000)
        # 답변은 15~40초 걸린다. 상태 표시만 보고 넘어가면 계산 중인 화면이 찍힌다.
        r.mark("법령을 찾아 답변을 만드는 중입니다 (15~40초)", fast=5.0)
        page.get_by_text("근거 자료").first.wait_for(state="visible", timeout=90_000)
        r.to("근거 자료", 10, "인용한 조문을 검색 원문과 대조해 표시합니다")
    except Exception as exc:                              # noqa: BLE001
        print(f"  질의응답 장면 건너뜀: {type(exc).__name__} — {str(exc)[:100]}")
        r.scroll(0.8, 3)
        r.beat(6)

    r.beat(3)
    return r.marks


def record(out_dir: Path, pace: float) -> tuple[Path, list, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(out_dir),
            record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        marks = play(page, pace)
        video = page.video
        end = (marks[-1].t + 3) if marks else 0.0
        ctx.close()          # 닫아야 파일이 완성된다
        browser.close()
        return Path(video.path()), marks, end


# ------------------------------------------------------------------ 편집

def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec_, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec_:02d},{ms:03d}"


def _run(cmd: list) -> bool:
    return subprocess.run(cmd, check=False,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def _style() -> str:
    """자막 모양. 발표장 뒷자리에서 읽혀야 하므로 크고 두껍게."""
    return ("FontName=Noto Sans CJK KR,FontSize=21,Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&HC0000000,"
            "BorderStyle=3,Outline=3,Shadow=0,MarginV=34")


def plan(marks: list, end: float, only_key: bool) -> list:
    """장면을 (시작, 끝, 배속, 자막, 제목카드) 목록으로 편다.

    배속을 건 구간은 길이가 줄어든다. 자막을 원본 시각으로 만들면 그만큼
    어긋나므로, 자른 뒤의 시간표를 다시 계산해 자막을 붙인다.
    """
    out, keep = [], not only_key
    for i, mk in enumerate(marks):
        stop = marks[i + 1].t if i + 1 < len(marks) else end
        if only_key:
            if mk.key:
                keep = True
            elif mk.endkey:
                keep = False
                # tab() 은 눌러 놓고 다 그려진 뒤에 표시를 남긴다. 그래서 전환된
                # 화면이 앞 구간 꼬리에 붙는다. 그 꼬리를 잘라 낸다.
                if out:
                    out[-1]["stop"] = max(out[-1]["start"] + 0.5,
                                          out[-1]["stop"] - 3.0)
        if stop - mk.t < 0.4:                 # 너무 짧은 조각은 버린다
            continue
        if keep:
            out.append({"start": mk.t, "stop": stop, "fast": mk.fast,
                        "text": mk.text,
                        "card": mk.card if (only_key and mk.key) else ""})
    return out


def write_srt(parts: list, path: Path, card_sec: float) -> Path:
    """자른 뒤의 시간표로 자막을 만든다."""
    lines, n, now = [], 0, 0.0
    for pt in parts:
        if pt.get("card"):
            now += card_sec
        dur = (pt["stop"] - pt["start"]) / pt["fast"]
        if pt["text"]:
            n += 1
            lines += [str(n), f"{_ts(now)} --> {_ts(now + max(dur, 1.2))}",
                      pt["text"], ""]
        now += dur
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def title_card(text: str, sub: str, png: Path) -> Path:
    """제목 카드. 영상 사이에 끼워 넣으면 '편집한 것'으로 읽힌다."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    if font_path is None:                    # 글꼴이 없으면 카드는 포기한다
        raise FileNotFoundError("한글 글꼴을 찾지 못했다: " + str(FONT_CANDIDATES))
    img = Image.new("RGB", (W, H), "#12161C")
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(font_path, 62)
    small = ImageFont.truetype(font_path, 28)
    d.rectangle([0, 0, W, 10], fill="#E8452C")
    tw = d.textbbox((0, 0), text, font=big)
    d.text(((W - tw[2]) / 2, H / 2 - 70), text, font=big, fill="#FFFFFF")
    if sub:
        sw = d.textbbox((0, 0), sub, font=small)
        d.text(((W - sw[2]) / 2, H / 2 + 30), sub, font=small, fill="#9AA4B2")
    img.save(png)
    return png


def card_clip(png: Path, mp4: Path, seconds: float) -> bool:
    return _run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
                 "-t", f"{seconds}", "-i", str(png), "-r", "25",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                 "-pix_fmt", "yuv420p", str(mp4)])


def cut(src: Path, start: float, stop: float, fast: float, dst: Path) -> bool:
    """구간 하나를 잘라 내고, 필요하면 빨리 감는다."""
    vf = "setpts=PTS-STARTPTS" if fast == 1.0 else f"setpts=(PTS-STARTPTS)/{fast}"
    return _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-ss", f"{start:.2f}", "-to", f"{stop:.2f}",
                 "-vf", vf, "-r", "25",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                 "-pix_fmt", "yuv420p", str(dst)])


def concat(files: list, dst: Path, work: Path) -> bool:
    # concat 목록의 상대 경로는 목록 파일이 있는 폴더 기준으로 풀린다.
    # 절대 경로로 적지 않으면 _work/_work/... 를 찾다 실패한다.
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{Path(f).resolve()}'\n" for f in files),
                   encoding="utf-8")
    return _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                 "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)])


def burn(src: Path, srt: Path, dst: Path) -> bool:
    esc = str(srt).replace("'", "")
    return _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-vf", f"subtitles='{esc}':force_style='{_style()}'",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])


def render(src: Path, marks: list, end: float, dst: Path, work: Path, *,
           only_key: bool = False, card_sec: float = 2.4) -> float:
    """자르고, 빨리 감고, 이어 붙이고, 자막을 태운다. 최종 길이(초)를 돌려준다."""
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    parts = plan(marks, end, only_key)
    if not parts:
        return 0.0

    files, total = [], 0.0
    if only_key:
        lead = work / "lead.mp4"
        if card_clip(title_card(CARD_LEAD[0], CARD_LEAD[1], work / "lead.png"),
                     lead, 3.0):
            files.append(lead)
            total += 3.0
    for i, pt in enumerate(parts):
        if pt.get("card"):
            cp = work / f"card{i}.mp4"
            if card_clip(title_card(pt["card"], "", work / f"card{i}.png"),
                         cp, card_sec):
                files.append(cp)
                total += card_sec
        vp = work / f"seg{i}.mp4"
        if cut(src, pt["start"], pt["stop"], pt["fast"], vp):
            files.append(vp)
            total += (pt["stop"] - pt["start"]) / pt["fast"]

    joined = work / "joined.mp4"
    if not concat(files, joined, work):
        return 0.0
    srt = write_srt(parts, work / "cap.srt", 3.0 if only_key else card_sec)
    if not burn(joined, srt, dst):
        return 0.0
    sped = [pt for pt in parts if pt["fast"] != 1.0]
    if sped:
        print("  배속 구간: "
              + " · ".join(f"{pt['fast']:g}배 {pt['stop'] - pt['start']:.0f}→"
                           f"{(pt['stop'] - pt['start']) / pt['fast']:.0f}초"
                           for pt in sped))
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pace", type=float, default=DEFAULT_PACE,
                    help="1.0 = 실제 시연 속도, 0.4 = 빠르게 확인만")
    ap.add_argument("--no-edit", action="store_true", help="녹화만 하고 편집은 생략")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.paths.outputs / "demo_video"
    print(f"녹화 시작 — {URL}  (배속 {args.pace})")
    raw, marks, end = record(out, args.pace)

    src = out / "원본.webm"
    if raw != src:
        raw.replace(src)
    # 중간에 실패한 녹화 조각(page@....webm)이 남으면 어느 것이 최신인지
    # 헷갈린다. 이번에 만든 것만 남긴다.
    for stale in out.glob("page@*.webm"):
        stale.unlink()
    print(f"\n  녹화 {int(end // 60)}분 {int(end % 60)}초 · 자막 {len(marks)}줄")

    write_srt(plan(marks, end, False), out / "자막_원본시각.srt", 0.0)
    (out / "장면.json").write_text(
        json.dumps([mk.__dict__ for mk in marks], ensure_ascii=False, indent=2),
        encoding="utf-8")

    if args.no_edit or not shutil.which("ffmpeg"):
        print("편집 생략 (ffmpeg 없음 또는 --no-edit)")
        return 0

    for name, only_key in (("시연_전체.mp4", False), ("시연_핵심.mp4", True)):
        dst = out / name
        secs = render(src, marks, end, dst, out / "_work", only_key=only_key)
        if secs:
            print(f"저장: {dst}  ({int(secs // 60)}분 {int(secs % 60)}초 · "
                  f"{dst.stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(out / "_work", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
