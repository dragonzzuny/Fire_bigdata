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

#: 4×4 격자에서 불꽃이 차지하는 칸과 그 온도.
#: (행, 열): 색. 행 0 이 위, 행 3 이 아래.
#: 위로 갈수록 옅어지는 것은 실제 불꽃의 끝이 그렇기 때문이다.
FLAME = {
    (0, 2): AMBER_LIGHT,
    (1, 1): AMBER,
    (1, 2): AMBER,
    (2, 1): RED,
    (2, 2): RED,
    (3, 1): DEEP,
    (3, 2): DEEP,
}

CELL, GAP, N = 72, 16, 4
SPAN = N * CELL + (N - 1) * GAP          # 336
PAD = (512 - SPAN) // 2                  # 88


def _cells(dark: str, x0: int = PAD, y0: int = PAD, *,
           dim_opacity: float = 1.0) -> str:
    """격자 16칸. 불꽃이 아닌 칸은 어둡게 깔린다."""
    out = []
    for r in range(N):
        for c in range(N):
            x = x0 + c * (CELL + GAP)
            y = y0 + r * (CELL + GAP)
            fill = FLAME.get((r, c))
            op = "" if fill else f' opacity="{dim_opacity}"'
            out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                       f'rx="18" fill="{fill or dark}"{op}/>')
    return "\n    ".join(out)


def mark_svg(*, on_dark: bool = False) -> str:
    """정사각 마크. 앱 아이콘·장표 모서리에 쓴다."""
    bg = PAPER if on_dark else INK
    dark_cell = "#E3E7EC" if on_dark else INK_SOFT
    dim = 0.55 if on_dark else 1.0
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <rect width="512" height="512" rx="112" fill="{bg}"/>
  <g>
    {_cells(dark_cell, dim_opacity=dim)}
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
    m = 0.72                                  # 마크 축소 비율
    size = int(512 * m)                       # 368
    tx = size + 56                            # 글자 시작
    w = (tx + 640) if tagline else (tx + 500)
    h = 380
    my = (h - size) // 2
    bg = f'<rect width="{w}" height="{h}" fill="{INK}"/>' if on_dark else ""
    sub = (f'<text x="{tx}" y="{my + 300}" font-family="Noto Sans CJK KR, sans-serif" '
           f'font-size="34" fill="{muted}" letter-spacing="1">'
           f'화재예방 점검·순찰 의사결정 시스템</text>') if tagline else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  {bg}
  <g transform="translate(0,{my}) scale({m})">
    <rect width="512" height="512" rx="112" fill="{ink if on_dark else INK}"/>
    {_cells("#E3E7EC" if on_dark else INK_SOFT, dim_opacity=0.55 if on_dark else 1.0)}
  </g>
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
