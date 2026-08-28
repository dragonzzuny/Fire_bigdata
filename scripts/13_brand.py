#!/usr/bin/env python
"""브랜드 자산(로고) 생성.

로고를 그림 파일로만 들고 있으면 색을 한 번 바꿀 때마다 손으로 다시 그려야
한다. 이 저장소의 다른 산출물과 같이, 로고도 명령 한 번으로 다시 만들어진다.

**마크의 뜻**: 4×4 격자 안에서 가운데 칸들만 뜨겁게 타오른다.
이 서비스가 하는 일이 정확히 그것이다 — 도시를 500m 격자로 나누고,
어느 칸이 위험한지를 가려낸다. 불꽃만 그리면 소화기 상표와 구분되지 않고,
격자만 그리면 무슨 서비스인지 알 수 없다. 둘이 겹쳐야 이 서비스가 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firebird.config import load_config      # noqa: E402

# 발표자료·대시보드와 같은 색을 쓴다. 여기서 어긋나면 화면과 장표가 따로 논다.
INK = "#1A1F2B"
INK_SOFT = "#2E3542"
RED = "#E34A31"
DEEP = "#C0492F"
AMBER = "#F0A63C"
AMBER_LIGHT = "#F6C86B"
MUTED = "#6B7484"
PAPER = "#FFFFFF"

#: 지도 마커. 불꽃을 담는 그릇이자 '여기'를 가리키는 손가락이다.
PIN = ("M256 78 C 174 78, 108 144, 108 226 C 108 330, 256 448, 256 448 "
       "C 256 448, 404 330, 404 226 C 404 144, 338 78, 256 78 Z")

#: 불꽃. 오른쪽에 굽이를 한 번 넣었다 — 좌우 대칭인 물방울 모양으로 그리면
#: 소방 서비스에서 정반대인 '물'로 읽힌다. 이 굽이 하나가 불과 물을 가른다.
FLAME = ("M256 66 C 248 132, 210 164, 184 196 C 152 236, 138 272, 138 312 "
         "C 138 378, 190 430, 256 430 C 322 430, 374 378, 374 312 "
         "C 374 266, 354 230, 326 200 C 322 232, 304 244, 292 232 "
         "C 276 216, 302 168, 256 66 Z")


def mark_svg(*, on_dark: bool = False, framed: bool = True) -> str:
    """정사각 마크. 앱 아이콘·장표 모서리에 쓴다.

    핀 안을 불꽃 모양으로 도려낸다. 색을 덧칠하지 않고 구멍을 내면
    작은 크기에서도 두 형태가 서로를 잡아먹지 않는다.
    """
    bg = PAPER if on_dark else INK
    pin = DEEP if on_dark else RED
    hole = bg
    frame = (f'<rect width="512" height="512" rx="112" fill="{bg}"/>'
             if framed else "")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  {frame}
  <path d="{PIN}" fill="{pin}"/>
  <g transform="translate(256,212) scale(0.54) translate(-256,-256)">
    <path d="{FLAME}" fill="{hole}"/>
  </g>
</svg>'''


def lockup_svg(*, on_dark: bool = False, tagline: bool = True) -> str:
    """마크 + 이름. 표지와 사이드바에 쓴다.

    밝은 배경용에는 바탕을 깔지 않는다. 흰 사각형을 깔면 흰 장표 위에서는
    안 보이다가 색이 있는 자리에 얹는 순간 네모가 드러난다.
    폭도 글자 길이에 맞춘다 — 남는 여백이 그대로 빈 띠로 보인다.
    """
    ink = PAPER if on_dark else INK
    muted = "#AAB3C0" if on_dark else MUTED
    m = 0.72
    size = int(512 * m)
    tx = size + 44
    w = (tx + 640) if tagline else (tx + 500)
    h = 380
    my = (h - size) // 2
    bg = f'<rect width="{w}" height="{h}" fill="{INK}"/>' if on_dark else ""
    sub = (f'<text x="{tx}" y="{my + 300}" font-family="Noto Sans CJK KR, sans-serif" '
           f'font-size="34" fill="{muted}" letter-spacing="1">'
           f'화재예방 점검·순찰 의사결정 시스템</text>') if tagline else ""
    inner = mark_svg(on_dark=on_dark, framed=False)
    inner = inner[inner.index(">", inner.index("<svg")) + 1:inner.rindex("</svg>")]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  {bg}
  <g transform="translate(-14,{my}) scale({m})">{inner}</g>
  <text x="{tx}" y="{my + 150}" font-family="Noto Sans CJK KR, sans-serif"
        font-size="118" font-weight="700" fill="{ink}" letter-spacing="-2">불씨예보</text>
  <text x="{tx}" y="{my + 222}" font-family="Noto Sans CJK KR, sans-serif"
        font-size="52" font-weight="500" fill="{DEEP}" letter-spacing="7">K-FIREBIRD</text>
  {sub}
</svg>'''


def render(svg: str, png: Path, width: int, height: int, scale: int = 2) -> None:
    """SVG 를 PNG 로. 발표자료·PPTX 는 SVG 를 못 넣으므로 래스터가 필요하다."""
    from playwright.sync_api import sync_playwright

    tmp = png.with_suffix(".tmp.html")
    tmp.write_text(
        "<html><head><meta charset='utf-8'><style>"
        "html,body{margin:0;padding:0;background:transparent}</style></head>"
        f"<body>{svg}</body></html>", encoding="utf-8")
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": width, "height": height},
                        device_scale_factor=scale)
        pg.goto(tmp.as_uri())
        pg.wait_for_timeout(400)
        pg.screenshot(path=str(png), omit_background=True)
        b.close()
    tmp.unlink(missing_ok=True)


def main() -> int:
    cfg = load_config()
    out = Path(__file__).resolve().parents[1] / "reports" / "brand"
    out.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("logo_mark.svg", "logo_mark.png", mark_svg(), 512, 512, 2),
        ("logo_mark_light.svg", "logo_mark_light.png", mark_svg(on_dark=True), 512, 512, 2),
        ("logo_lockup.svg", "logo_lockup.png", lockup_svg(), 1064, 380, 2),
        ("logo_lockup_light.svg", "logo_lockup_light.png",
         lockup_svg(on_dark=True), 1064, 380, 2),
        ("logo_lockup_plain.svg", "logo_lockup_plain.png",
         lockup_svg(tagline=False), 924, 380, 2),
    ]
    from PIL import Image

    for svg_name, png_name, svg, w, h, sc in jobs:
        (out / svg_name).write_text(svg, encoding="utf-8")
        path = out / png_name
        render(svg, path, w, h, sc)
        # 남는 투명 여백을 잘라 낸다. 그대로 두면 넣는 쪽에서 로고가
        # 실제보다 작게 앉고, 위치를 맞추기도 어렵다.
        im = Image.open(path)
        if im.mode == "RGBA":
            box = im.getbbox()
            if box and box != (0, 0, im.width, im.height):
                im.crop(box).save(path)
                im = Image.open(path)
        print(f"  {png_name}  ({im.width}×{im.height})")

    # 발표자료가 바로 집어 갈 수 있게 그림 폴더에도 둔다.
    figs = cfg.paths.figures
    figs.mkdir(parents=True, exist_ok=True)
    for name in ("logo_mark.png", "logo_mark_light.png",
                 "logo_lockup.png", "logo_lockup_plain.png"):
        (figs / name).write_bytes((out / name).read_bytes())
    print(f"완료 -> {out}  (사본: {figs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
