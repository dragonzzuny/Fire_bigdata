"""소방 법령 수집 — 국가법령정보센터 OPEN API.

화재예방 업무는 법령이 근거다. 점검 주기도, 화재예방강화지구 지정도,
특별경계근무 편성도 조문에 있다. 담당자가 "이 대상물 점검 주기가 얼마죠"를
물었을 때 시스템이 조문을 근거로 답할 수 있어야 실제로 쓰인다.

주의: 조회는 `MST`(법령일련번호) 파라미터로 해야 본문이 온다.
검색 결과의 `ID` 를 그대로 넣으면 "일치하는 법령이 없습니다"가 돌아온다.
"""
from __future__ import annotations

import html
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"

#: 화재예방 업무에 실제로 쓰이는 법령. 이름은 검색어로 그대로 쓴다.
FIRE_LAWS = [
    "화재의 예방 및 안전관리에 관한 법률",
    "화재의 예방 및 안전관리에 관한 법률 시행령",
    "화재의 예방 및 안전관리에 관한 법률 시행규칙",
    "소방시설 설치 및 관리에 관한 법률",
    "소방시설 설치 및 관리에 관한 법률 시행령",
    "다중이용업소의 안전관리에 관한 특별법",
    "소방기본법",
]


def _clean(text: str) -> str:
    """XML/CDATA/태그를 걷어내고 공백을 정리한다."""
    if not text:
        return ""
    t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    return re.sub(r"[ \t]+", " ", t).strip()


def _norm(s: str) -> str:
    """법령명 비교용 정규화. 공백만 다른 이름을 같은 것으로 본다."""
    return re.sub(r"\s+", "", str(s))


def search_law(name: str, *, oc: str = "test", timeout: int = 45) -> dict | None:
    """법령명으로 검색해 (법령일련번호, 정식명칭)을 얻는다.

    검색어에 공백이 있으면 엉뚱한 법령이 걸린다. '화재의 예방 및 안전관리에
    관한 법률' 로 검색하면 '소방시설 설치 및 관리에 관한 법률' 이 1순위로 온다.
    공백을 뗀 질의라야 정확히 잡힌다.
    """
    try:
        r = requests.get(SEARCH_URL, timeout=timeout,
                         params={"OC": oc, "target": "law", "type": "XML",
                                 "query": _norm(name), "display": 100})
        r.raise_for_status()
        t = r.text
    except requests.RequestException as exc:
        log.warning("법령 검색 실패(%s): %s", name, exc)
        return None
    if "사용자 정보 검증에 실패" in t:
        log.warning("법령 API 인증 실패 — OC 값을 확인하라")
        return None

    # 결과 블록을 순회하며 **정확히 일치**하는 것만 채택한다.
    # 첫 결과로 물러나면 다른 법을 가져와 놓고 이름만 맞는 척하게 된다.
    # 응답 블록 태그가 <law> 가 아닌 경우가 있어 두 방식으로 쪼갠다.
    blocks = re.findall(r"<law>(.*?)</law>", t, re.S)
    if not blocks:
        blocks = re.split(r"(?=<법령일련번호>)", t)[1:] or [t]
    for block in blocks:
        mst = re.search(r"<법령일련번호>(\d+)</법령일련번호>", block)
        nm = re.search(r"<법령명한글>(.*?)</법령명한글>", block, re.S)
        if mst and nm and _norm(_clean(nm.group(1))) == _norm(name):
            return {"mst": mst.group(1), "name": _clean(nm.group(1))}
    log.warning("법령명이 정확히 일치하지 않는다: %s", name)
    return None


def fetch_articles(mst: str, law_name: str, *, oc: str = "test",
                   timeout: int = 60) -> list[dict]:
    """조문 단위로 쪼갠다. 조문이 검색 단위가 되어야 근거를 조문으로 댈 수 있다."""
    try:
        r = requests.get(SERVICE_URL, timeout=timeout,
                         params={"OC": oc, "target": "law", "type": "XML", "MST": mst})
        r.raise_for_status()
        t = r.text
    except requests.RequestException as exc:
        log.warning("법령 본문 실패(%s): %s", law_name, exc)
        return []

    out = []
    for block in re.findall(r"<조문단위[^>]*>(.*?)</조문단위>", t, re.S):
        num = re.search(r"<조문번호>(\d+)</조문번호>", block)
        sub = re.search(r"<조문가지번호>(\d+)</조문가지번호>", block)
        title = re.search(r"<조문제목>(.*?)</조문제목>", block, re.S)
        body = re.search(r"<조문내용>(.*?)</조문내용>", block, re.S)
        if not num:
            continue
        label = f"제{num.group(1)}조"
        if sub and sub.group(1) not in ("0", ""):
            label += f"의{sub.group(1)}"
        text = _clean(body.group(1) if body else "")
        # 항·호까지 붙여야 '점검 주기' 같은 세부가 빠지지 않는다.
        for part in re.findall(r"<항내용>(.*?)</항내용>", block, re.S):
            text += "\n" + _clean(part)
        for part in re.findall(r"<호내용>(.*?)</호내용>", block, re.S):
            text += "\n" + _clean(part)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        if not text:
            continue
        out.append({"law": law_name, "article": label,
                    "title": _clean(title.group(1)) if title else "",
                    "text": text})
    return out


def collect(cfg, laws: list[str] | None = None, *, oc: str = "test",
            refresh: bool = False) -> pd.DataFrame:
    """법령 전체를 조문 단위로 모아 캐시한다."""
    cache = cfg.paths.cache / "law_articles.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    rows: list[dict] = []
    for name in (laws or FIRE_LAWS):
        found = search_law(name, oc=oc)
        if not found:
            log.warning("법령을 찾지 못함: %s", name)
            continue
        arts = fetch_articles(found["mst"], found["name"], oc=oc)
        log.info("%s — 조문 %d개", found["name"], len(arts))
        rows += arts
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


# ---------------------------------------------------------------- 법정 서식

FORM_DIR_NAME = "forms"


def _tag(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.S)
    return _clean(m.group(1)) if m else ""


def list_forms(mst: str, law_name: str, *, oc: str = "test",
               timeout: int = 90) -> list[dict]:
    """법령의 별표·서식 목록과 HWP/PDF 내려받기 링크.

    계획서를 우리 마음대로 만들면 결재에서 되돌아온다. 법정 서식이 있는 업무는
    그 서식을 써야 한다. 서식 파일 링크가 응답에 들어 있어 그대로 받을 수 있다.
    """
    try:
        r = requests.get(SERVICE_URL, timeout=timeout,
                         params={"OC": oc, "target": "law", "type": "XML", "MST": mst})
        r.raise_for_status()
        t = r.text
    except requests.RequestException as exc:
        log.warning("별표·서식 조회 실패(%s): %s", law_name, exc)
        return []

    out = []
    for block in re.findall(r"<별표단위[^>]*>(.*?)</별표단위>", t, re.S):
        hwp = _tag(block, "별표서식파일링크")
        pdf = _tag(block, "별표서식PDF파일링크")
        if not (hwp or pdf):
            continue
        out.append({
            "law": law_name,
            "kind": _tag(block, "별표구분"),
            "no": _tag(block, "별표번호").lstrip("0") or "0",
            "title": _tag(block, "별표제목"),
            "hwp_url": f"https://www.law.go.kr{hwp}" if hwp else "",
            "pdf_url": f"https://www.law.go.kr{pdf}" if pdf else "",
            "hwp_name": _tag(block, "별표HWP파일명"),
        })
    return out


def collect_forms(cfg, laws: list[str] | None = None, *, oc: str = "test",
                  refresh: bool = False) -> pd.DataFrame:
    """모든 대상 법령의 별표·서식 목록."""
    cache = cfg.paths.cache / "law_forms.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows: list[dict] = []
    for name in (laws or FIRE_LAWS):
        found = search_law(name, oc=oc)
        if not found:
            continue
        got = list_forms(found["mst"], found["name"], oc=oc)
        log.info("%s — 별표·서식 %d개", found["name"], len(got))
        rows += got
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def download_forms(cfg, forms: pd.DataFrame, *, only_forms: bool = True,
                   limit: int | None = None, timeout: int = 90) -> pd.DataFrame:
    """서식 파일을 내려받아 저장한다.

    only_forms=True 면 '별표'(기준표)는 건너뛰고 '서식'(작성 양식)만 받는다.
    실제로 담당자가 채워 넣는 것은 서식이다.
    """
    if forms.empty:
        return forms
    target = forms[forms["kind"] == "서식"] if only_forms else forms
    if limit:
        target = target.head(limit)

    out_dir = cfg.paths.processed / FORM_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for r in target.itertuples():
        safe = re.sub(r"[^\w가-힣ㆍ()\-]+", "_", f"{r.law}_{r.kind}{r.no}_{r.title}")[:120]
        path = out_dir / f"{safe}.hwp"
        if path.exists():
            saved.append({"title": r.title, "path": str(path), "status": "캐시"})
            continue
        url = r.hwp_url or r.pdf_url
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            saved.append({"title": r.title, "path": str(path),
                          "status": f"{len(resp.content)/1024:.0f} KB"})
        except requests.RequestException as exc:
            log.warning("서식 내려받기 실패 %s: %s", r.title, exc)
            saved.append({"title": r.title, "path": "", "status": f"실패: {exc}"})
        time.sleep(0.25)
    return pd.DataFrame(saved)
