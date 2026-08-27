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

    if made:
        _compare_image(cfg)
    print(f"완료: {made}건")
    return 0 if made else 1


def _compare_image(cfg) -> None:
    """빈 양식과 채운 대장을 한 장에 나란히 놓는다 — 발표자료용.

    화면 캡처는 스크롤 때문에 양식의 윗부분만 들어간다. 정작 보여 줘야 하는 것은
    '어느 칸이 채워졌는가'이므로, 두 문서를 통째로 줄여 나란히 붙인다.
    """
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont

    from firebird import forms as FM, stations as ST
    from firebird import dataset as D

    blank = cfg.paths.processed / "forms" / "서식11_빈양식.png"
    if not blank.exists():
        return

    panel = D.load_panel(cfg, cfg.default_city if hasattr(cfg, "default_city")
                         else "ulsan")
    year = int(cfg.holdout_year)
    cur = panel[panel["year"] == year]
    if cur.empty:
        return
    row = cur.nlargest(1, "fires_cum").iloc[0]
    label = cfg.city("ulsan")["label"]
    sta = pd.concat([ST.station_table(cur, cfg, level=lv, city_label=label)
                     for lv in ("station", "center")], ignore_index=True)
    led = FM.zone_ledger(row, city_label=label, year=year, stations=sta,
                         grid_m=int(cfg.grid_size_m))
    html = FM.render_ledger_html(led, city_label=label, year=year)

    tmp = cfg.paths.figures / "_ledger.html"
    tmp.write_text("<html><head><meta charset='utf-8'></head>"
                   "<body style='width:900px;margin:18px auto;background:#fff'>"
                   + html + "</body></html>", encoding="utf-8")
    filled = cfg.paths.figures / "_ledger.png"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 940, "height": 1330},
                            device_scale_factor=2)
            pg.goto(tmp.as_uri())
            pg.wait_for_timeout(1200)
            pg.screenshot(path=str(filled), clip={"x": 0, "y": 0,
                                                  "width": 940, "height": 1330})
            b.close()
    except Exception as exc:                        # noqa: BLE001
        print(f"  채운 대장 이미지 생성 실패: {type(exc).__name__}")
        return

    a, bimg = Image.open(blank).convert("RGB"), Image.open(filled).convert("RGB")
    h = 1500
    a = a.resize((int(a.width * h / a.height), h), Image.LANCZOS)
    bimg = bimg.resize((int(bimg.width * h / bimg.height), h), Image.LANCZOS)
    gap, top, pad = 46, 62, 24
    out = Image.new("RGB", (a.width + bimg.width + gap + pad * 2,
                            h + top + pad), "white")
    out.paste(a, (pad, top))
    out.paste(bimg, (pad + a.width + gap, top))
    d = ImageDraw.Draw(out)
    for x0, x1 in ((pad, pad + a.width),
                   (pad + a.width + gap, pad + a.width + gap + bimg.width)):
        d.rectangle([x0, top, x1 - 1, top + h - 1], outline=(200, 205, 210))
    font = None
    for cand in ("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"):
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 34)
            break
    d.text((pad, 16), "법제처 원본 서식 (빈 양식)", fill=(40, 44, 52), font=font)
    d.text((pad + a.width + gap, 16),
           f"불씨예보가 채운 대장 ({led['n_filled']}/{led['n_fields']}칸)",
           fill=(192, 73, 47), font=font)
    dest = cfg.paths.figures / "fig_form_compare.png"
    out.save(dest)
    tmp.unlink(missing_ok=True)
    filled.unlink(missing_ok=True)
    print(f"  대조 이미지 → {dest}  {out.size}")


if __name__ == "__main__":
    raise SystemExit(main())
