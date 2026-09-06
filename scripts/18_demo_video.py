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
#: 사이드바가 화면의 19% 를 상시 차지한다. 값이 한 번 정해진 뒤로는 계속
#: 같은 상태라, 그 자리를 본문에 주면 표와 지도가 그만큼 커진다.
#: 다만 '인원을 넣으면 갈 수 있는 구역만 남는다' 의 증거가 그 사이드바라,
#: 입력 장면(side=True)에서는 남기고 그 뒤부터 잘라 낸다.
#: 두 창의 폭은 같아야 한다 — 다르면 이어 붙일 때 ffmpeg 이 거부한다.
SIDE_W = 300
OUT_W = W - SIDE_W
#: 자막 전용 띠. 화면 위에 자막을 얹으면 컷마다 다른 것을 가린다 — 점검
#: 순위표, 관서별 표, '빠진 구역 9개 / 새 구역 6개' 가 실제로 가려져 있었다.
#: 컷마다 위치를 맞추는 대신 아래에 자막만 놓는 띠를 붙인다. 구조적으로
#: 가릴 수가 없어지고, 컷이 늘어도 다시 손볼 일이 없다.
BAR_H = 132
OUT_H = H + BAR_H

#: libass 는 SRT 에 화면 크기가 없으면 PlayResY 를 288 로 잡고 그 비율로
#: 글자와 여백을 키운다. 그래서 FontSize·MarginV 를 화소로 착각하면
#: 자막 상자가 띠보다 커진다 — 실제로 80px 띠에 171px 상자가 얹혔다.
#: 아래 두 값은 그 배율을 되돌려 계산한다.
_ASS_RES = 288
_BOX_PER_PT = 2.1                 # 글자 크기 1 당 상자 높이(측정값)


def _sub_metrics() -> tuple[float, int]:
    """(FontSize, MarginV) — 자막 상자가 띠 안에 들어오도록."""
    scale = OUT_H / _ASS_RES
    size = round((BAR_H - 16) / (scale * _BOX_PER_PT), 1)
    margin = max(1, int(8 / scale))
    return size, margin
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
    side: bool = False      # 사이드바(입력값)를 화면에 남긴다


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
             fast: float = 1.0, endkey: bool = False,
             side: bool = False) -> None:
        t = self.now()
        self.marks.append(Mark(t, text.strip(), key, card, fast, endkey, side))
        tag = f"  ({fast:g}배속)" if fast != 1.0 else ""
        print(f"  [{int(t // 60)}:{int(t % 60):02d}] {text}{tag}")

    def beat(self, seconds: float, text: str = "", **kw) -> None:
        self.page.wait_for_timeout(int(seconds * self.pace * 1000))
        if text:
            self.mark(text, **kw)

    def tab(self, name: str, settle: float, text: str, **kw) -> None:  # noqa: D401
        # 표시는 화면이 다 그려진 뒤에 남긴다. 그래서 탭이 바뀐 화면이 앞
        # 장면의 꼬리에 붙는다. 핵심본을 여기서 끊으려면 누르기 '전'에
        # 경계를 찍어야 한다. 실제로 대응취약 화면이 딸려 들어갔었다.
        if kw.pop("endkey", False):
            self.marks.append(Mark(self.now(), "", endkey=True))
        self.page.get_by_role("tab", name=name).click(timeout=25_000)
        self.settle(settle, text, **kw)

    def read(self, label: str, default: str = "") -> str:
        """화면에 떠 있는 지표 값을 그대로 읽어 온다.

        자막에 숫자를 손으로 적으면 화면과 어긋난다. 이 영상은 그때그때
        다시 돌린 결과를 찍는 것이라 값이 바뀔 수 있다. 화면에서 읽으면
        자막과 화면이 어긋날 수가 없다.
        """
        try:
            # st.metric 은 라벨과 값이 각각 제 testid 를 갖는다. 부모를 거슬러
            # 올라가 글자를 긁으면 옆 지표까지 딸려 온다.
            m = self.page.locator("[data-testid='stMetric']").filter(
                has_text=label).first
            return m.locator("[data-testid='stMetricValue']").inner_text(
                timeout=8_000).strip()
        except Exception:                                 # noqa: BLE001
            return default

    def count_in(self, label: str, default: str = "") -> str:
        """'빠진 구역 9개' 처럼 라벨과 수가 붙어 있는 문구를 통째로 읽는다."""
        import re as _re
        try:
            t = self.page.get_by_text(_re.compile(label + r"\s*\d+개")).first
            return t.inner_text(timeout=8_000).strip().splitlines()[0]
        except Exception:                                 # noqa: BLE001
            return default

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

    def to_block(self, needle: str, seconds: float, text: str = "", **kw) -> bool:
        """그 문구를 품은 가로 블록 전체를 화면에 올린다.

        문구만 올리면 그 문구가 가운데로 오고 아래 그림이 잘린다.
        전후 비교 지도가 실제로 그렇게 잘렸다.
        """
        try:
            self.page.locator("[data-testid='stHorizontalBlock']").filter(
                has_text=needle).first.scroll_into_view_if_needed(timeout=20_000)
        except Exception:                                 # noqa: BLE001
            return self.to(needle, seconds, text, **kw)
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
    r.beat(6, "실제 울산 자료로 지금 도는 화면입니다", side=True)

    # --- 1. 예방점검 배분 -----------------------------------------------
    r.tab("예방점검 배분", 6, "점검 가용 인력을 먼저 넣습니다",
          key=True, card="인력에 맞춘 예방점검 배분", side=True)
    r.to("위험도 상위", 8,
         "위험한 곳일수록 점검할 건물이 많습니다")
    r.to("점검 순위표", 4,
         "갈 수 있는 조합 중 가장 많이 잡는 쪽으로")
    r.beat(5)
    r.mark("같은 인력으로 186개 구역 · 화재 17.8%")
    r.beat(4)
    r.settle(6, "1위 구역 하나가 713건 — 가용 640건을 넘습니다", fast=2.0)

    # --- 2. 순찰 동선 (핵심) --------------------------------------------
    r.tab("예방순찰 계획", 13, "순찰 목적과 출동 단위를 고릅니다",
          key=True, card="관서별 순찰 동선")
    r.beat(3)
    _st = r.read("출동 관서", "8개")
    _km = r.read("총 이동", "")
    r.mark(f"관서 {_st} · 총 이동 {_km}".strip(" ·"))
    r.beat(3)
    if r.to("관서별 순찰 구역", 4):
        # 지도가 실제로 보이는 자리에서 동선 이야기를 한다.
        r.settle(7, "관서마다 자기 관할만 돕니다")
        r.beat(7, "색깔이 관서 · 선은 실제 도로 주행거리")
    r.scroll(0.7, 3)
    r.beat(6)

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
        # 목적 이름이 길어 자막이 두 줄로 넘어간다. 짧게 쓴다.
        # 재계산이 끝나기 전에는 지도가 세계 지도로 돌아가 있다.
        # settle 은 이미 멈춰 있으면 바로 돌아오므로 표시가 붙어 버린다.
        # 눈에 보이는 시간을 확실히 주려면 beat 로 벌린다.
        r.mark("야간순찰로 바꿔 다시 계산합니다", fast=8.0)
        r.beat(4)
        r.settle(0)
        r.mark("", fast=8.0)          # 계산이 끝날 때까지는 자막 없이 지나간다
        r.beat(6)
        if r.to_block("바꾸기 전", 4):
            # '같은 범위 · 같은 배율' 은 화면에도 글자로 있고 발표자도
            # 얹어 말한다. 자막까지 하면 세 번이다. 자막은 결과를 말한다.
            r.mark("조건을 바꾸면 계획이 통째로 다시 나옵니다")
            r.settle(6)
            _gone = r.count_in("빠진 구역", "")
            _new = r.count_in("새로 들어온 구역", "")
            if _gone and _new:
                r.mark(f"{_gone} · {_new}")
            r.beat(6)
        r.to("가장 먼 순찰조", 8)
    except Exception as exc:                              # noqa: BLE001
        print(f"  목적 변경 장면 건너뜀: {type(exc).__name__} — {str(exc)[:120]}")

    # --- 4. 계획서 (핵심) ------------------------------------------------
    r.tab("순찰·점검 계획서", 5, "이 동선을 계획서로",
          key=True, card="결재 올릴 순찰 계획서")
    try:
        page.get_by_text("AI로 문체 다듬기").click(timeout=10_000)
        r.beat(2, "숫자와 법령은 시스템이, AI는 문장만")
    except Exception:                                     # noqa: BLE001
        pass
    r.to("계획서 생성", 3)
    page.get_by_role("button", name="계획서 생성").click(timeout=25_000)
    r.mark("", fast=12.0)             # 생성 대기는 보여 주지 않는다
    r.settle(4)   # 문서는 아래에 있다. 스크롤한 뒤에 자막을 띄운다
    try:
        page.locator(".docview").first.scroll_into_view_if_needed(timeout=25_000)
        r.beat(6, "기관·수신·경유·시행일까지 공문 서식 그대로")
        r.beat(5)
        # 많이 내리면 문서를 지나쳐 아래 입력 폼이 나온다. 문서 안에서만 움직인다.
        r.mark("격자 · 순찰 구간 · 조치 기준까지 문서 안에 정의됩니다")
        r.scroll(1.1, 7)
        r.beat(3)
        r.mark("끝에 붙임과 결재란까지 들어갑니다")
        r.scroll(0.55, 5)
        r.beat(4)
        r.mark("이대로 결재에 올립니다")
        page.locator(".docview").first.scroll_into_view_if_needed(timeout=15_000)
        r.beat(5)
    except Exception as exc:                              # noqa: BLE001
        print(f"  문서 장면 건너뜀: {type(exc).__name__}")

    # --- 5. 대응취약 · 업무 도우미 (전체본에만) --------------------------
    r.tab("대응취약 구역", 8, "고위험 · 소화전 없는 구역",
          endkey=True)
    r.to("소방용수 사각지대", 10, "소화전 신설 우선순위의 근거")
    r.tab("업무 도우미", 6, "법령은 조문 번호와 함께 답합니다")
    try:
        page.get_by_text("자주 찾는 질문").first.scroll_into_view_if_needed(
            timeout=15_000)
        page.wait_for_timeout(1_500)
        btn = page.locator("[data-testid='stMain']").get_by_role(
            "button").filter(has_text="화재예방강화지구").first
        btn.click(timeout=15_000)
        # 답변은 15~40초 걸린다. 상태 표시만 보고 넘어가면 계산 중인 화면이 찍힌다.
        r.mark("", fast=12.0)         # 답변 대기도 보여 주지 않는다
        page.get_by_text("근거 자료").first.wait_for(state="visible", timeout=90_000)
        r.to("근거 자료", 10, "인용 조문을 원문과 대조")
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


def _duration(path: Path) -> float:
    """만들어진 파일의 실제 길이(초). 계산값을 믿지 않기 위해 잰다."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _run(cmd: list) -> bool:
    return subprocess.run(cmd, check=False,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def _style() -> str:
    """자막 모양. 발표장 뒷자리에서 읽혀야 하므로 크고 두껍게."""
    size, margin = _sub_metrics()
    return (f"FontName=Noto Sans CJK KR,FontSize={size},Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00121612,"
            f"BorderStyle=3,Outline=3,Shadow=0,MarginV={margin}")


def plan(marks: list, end: float, only_key: bool, base: float = 1.0) -> list:
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
        if stop - mk.t < 0.4:                 # 너무 짧은 조각은 버린다
            if mk.text:
                print(f"  ※ 자막이 너무 짧아 버립니다: '{mk.text[:30]}' "
                      f"({stop - mk.t:.1f}초). 표시를 연달아 찍지 마십시오")
            continue
        if keep:
            fast = mk.fast if mk.fast != 1.0 else base
            # 자막이 지나가 버리면 없느니만 못하다. 한국어는 초당 6자쯤
            # 읽는다. 그만큼도 안 뜨면 그 구간만 천천히 돌린다.
            if mk.text:
                need = len(mk.text) / 6.0
                if (stop - mk.t) / fast < need:
                    fast = max(1.0, (stop - mk.t) / need)
            out.append({"start": mk.t, "stop": stop, "fast": fast,
                        "text": mk.text, "side": mk.side,
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
    img = Image.new("RGB", (OUT_W, OUT_H), "#12161C")
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(font_path, 62)
    small = ImageFont.truetype(font_path, 28)
    d.rectangle([0, 0, OUT_W, 10], fill="#E8452C")
    tw = d.textbbox((0, 0), text, font=big)
    d.text(((OUT_W - tw[2]) / 2, OUT_H / 2 - 70), text, font=big, fill="#FFFFFF")
    if sub:
        sw = d.textbbox((0, 0), sub, font=small)
        d.text(((OUT_W - sw[2]) / 2, OUT_H / 2 + 30), sub, font=small, fill="#9AA4B2")
    img.save(png)
    return png


def card_clip(png: Path, mp4: Path, seconds: float) -> bool:
    return _run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
                 "-t", f"{seconds}", "-i", str(png), "-r", "25",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                 "-pix_fmt", "yuv420p", str(mp4)])


def cut(src: Path, start: float, stop: float, fast: float, dst: Path,
        *, side: bool = False) -> bool:
    """구간 하나를 잘라 내고, 필요하면 빨리 감는다.

    -ss 를 -i 뒤에 두면 setpts 가 먹지 않는다. 20초 구간에 1.4배를 걸어도
    20초가 그대로 나왔다. 입력 앞으로 옮겨야 한다.
    """
    # 사이드바를 남길 때는 왼쪽 창, 아닐 때는 오른쪽 창. 폭은 같다.
    crop = f"crop={OUT_W}:{H}:{0 if side else SIDE_W}:0"
    bar = f"pad={OUT_W}:{OUT_H}:0:0:color=#12161C"
    speed = "setpts=PTS-STARTPTS" if fast == 1.0 else f"setpts=(PTS-STARTPTS)/{fast}"
    vf = f"{crop},{bar},{speed}"
    return _run(["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{start:.2f}", "-to", f"{stop:.2f}", "-i", str(src),
                 "-vf", vf, "-r", "25",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                 "-pix_fmt", "yuv420p", str(dst)])


def to_169(src: Path, dst: Path) -> bool:
    """16:10 화면을 16:9 로 맞춘다. 위아래에 같은 색 띠를 넣어 가운데 둔다.

    발표장은 발표자가 화면비를 맞출 수 있지만, 제출본은 심사위원 PC 에서
    그냥 재생된다. 그때 좌우에 검은 띠가 생기면 화면이 작아진다.
    """
    if not src.exists() or not shutil.which("ffmpeg"):
        return False
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf",
         # 원본이 16:10 이라 폭을 1920 에 맞추면 세로가 1200 이 되어
         # 1080 프레임보다 커진다. 세로를 먼저 맞추고 좌우를 채운다.
         "scale=-2:1080,pad=1920:1080:(ow-iw)/2:0:color=#12161C",
         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
         "-pix_fmt", "yuv420p", str(dst)],
        capture_output=True, text=True)
    return r.returncode == 0 and dst.exists()


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
           only_key: bool = False, card_sec: float = 2.0,
           subs: bool = True, base: float = 1.0, lead: bool = None) -> float:
    """자르고, 빨리 감고, 이어 붙이고, 자막을 태운다. 최종 길이(초)를 돌려준다."""
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    parts = plan(marks, end, only_key, base)
    if not parts:
        return 0.0

    files, total = [], 0.0
    if only_key if lead is None else lead:
        lead = work / "lead.mp4"
        if card_clip(title_card(CARD_LEAD[0], CARD_LEAD[1], work / "lead.png"),
                     lead, 2.5):
            files.append(lead)
            total += 2.5
    for i, pt in enumerate(parts):
        if pt.get("card"):
            cp = work / f"card{i}.mp4"
            if card_clip(title_card(pt["card"], "", work / f"card{i}.png"),
                         cp, card_sec):
                files.append(cp)
                total += card_sec
        vp = work / f"seg{i}.mp4"
        if cut(src, pt["start"], pt["stop"], pt["fast"], vp,
               side=pt.get("side", False)):
            files.append(vp)
            total += (pt["stop"] - pt["start"]) / pt["fast"]

    joined = work / "joined.mp4"
    if not concat(files, joined, work):
        return 0.0
    if subs:
        srt = write_srt(parts, work / "cap.srt", 2.5 if only_key else card_sec)
        if not burn(joined, srt, dst):
            return 0.0
    else:
        joined.replace(dst)
    real = _duration(dst)
    if real and abs(real - total) > 3:
        print(f"  ※ 계산 {total:.0f}초와 실제 {real:.0f}초가 다릅니다")
    total = real or total
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
    ap.add_argument("--speed", type=float, default=1.25,
                    help="보통 구간 배속. 자막을 읽을 수 있는 한도가 1.4 근처다")
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

    # 자막본은 그대로 틀 때, 무자막본은 발표자가 얹어 말할 때 쓴다.
    made = {}
    for name, only_key, subs, lead in (
            ("시연_전체.mp4", False, True, True),
            ("시연_핵심.mp4", True, True, True),
            ("시연_핵심_무자막.mp4", True, False, True)):
        dst = out / name
        secs = render(src, marks, end, dst, out / "_work",
                      only_key=only_key, subs=subs, base=args.speed,
                      lead=lead)
        if secs:
            made[name] = secs
            print(f"저장: {dst}  ({int(secs // 60)}분 {int(secs % 60)}초 · "
                  f"{dst.stat().st_size / 1e6:.1f} MB)")

    if "시연_핵심.mp4" in made:
        sub_dst = out / "불씨예보_시연영상_박용준.mp4"
        if to_169(out / "시연_핵심.mp4", sub_dst):
            print(f"저장: {sub_dst}  (제출용 · 발표와 같은 영상 · 16:9)")

    shutil.rmtree(out / "_work", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
