"""원본 CSV 적재와 컬럼 별칭 해석.

플랫폼 CSV는 인코딩(cp949/euc-kr/utf-8-sig)과 컬럼명이 시도별로 다르다.
여기서 그 차이를 한 번만 흡수하고, 이후 모듈은 표준 이름만 다룬다.
"""
from __future__ import annotations

import glob
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

ENCODINGS = ("utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1")


class SchemaError(RuntimeError):
    """필수 컬럼을 원본에서 찾지 못했다. 조용히 넘어가지 않는다."""


def read_csv_any(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """인코딩을 순서대로 시도해서 읽는다. 전부 실패하면 마지막 예외를 올린다."""
    path = Path(path)
    last: Exception | None = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
            if enc != ENCODINGS[0]:
                log.info("%s: %s 인코딩으로 읽음", path.name, enc)
            return df
        except UnicodeDecodeError as exc:
            last = exc
        except pd.errors.ParserError as exc:
            last = exc
    raise RuntimeError(f"{path} 를 읽지 못했다 (시도한 인코딩: {ENCODINGS})") from last


def normalize_colname(name: str) -> str:
    """비교용 정규화: 공백/괄호/언더스코어/점 제거, 소문자화."""
    return re.sub(r"[\s_.\-()\[\]/]", "", str(name)).lower()


def resolve_columns(df: pd.DataFrame,
                    aliases: dict[str, list[str]],
                    *,
                    required: bool,
                    context: str) -> dict[str, str]:
    """표준이름 -> 실제 컬럼명 매핑.

    완전일치(정규화 후) 우선, 없으면 부분일치. 찾지 못한 항목은
    required 면 SchemaError, optional 이면 로그를 남기고 빠진다.
    """
    norm_to_actual: dict[str, str] = {}
    for col in df.columns:
        norm_to_actual.setdefault(normalize_colname(col), col)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for std_name, candidates in (aliases or {}).items():
        hit = None
        for cand in candidates:
            n = normalize_colname(cand)
            if n in norm_to_actual:
                hit = norm_to_actual[n]
                break
        if hit is None:                      # 완전일치 실패 -> 부분일치
            for cand in candidates:
                n = normalize_colname(cand)
                for norm, actual in norm_to_actual.items():
                    if n and n in norm:
                        hit = actual
                        break
                if hit:
                    break
        if hit is None:
            missing.append(std_name)
        else:
            resolved[std_name] = hit

    if missing:
        if required:
            raise SchemaError(
                f"[{context}] 필수 컬럼을 찾지 못했다: {missing}\n"
                f"  실제 컬럼: {list(df.columns)}\n"
                f"  configs/schema.yaml 의 해당 항목에 실제 컬럼명을 후보로 추가하라."
            )
        log.warning("[%s] 선택 컬럼 없음(해당 피처 비활성): %s", context, missing)
    return resolved


def find_files(raw_dir: Path, patterns: list[str]) -> list[Path]:
    """glob 패턴 목록으로 CSV 를 찾는다. 하위 디렉터리도 훑는다."""
    found: list[Path] = []
    for pat in patterns:
        for ext in ("csv", "CSV"):
            found += [Path(p) for p in glob.glob(str(raw_dir / "**" / f"{pat}.{ext}"), recursive=True)]
            found += [Path(p) for p in glob.glob(str(raw_dir / f"{pat}.{ext}"))]
    # 중복 제거, 순서 유지
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def load_dataset(cfg, city: str, kind: str) -> pd.DataFrame:
    """schema.yaml 의 한 dataset 을 읽어 표준 컬럼명으로 바꿔 돌려준다.

    같은 종류의 CSV 가 여러 개면 세로로 붙인다(연도별 분할 파일 대응).
    """
    spec = cfg.schema["datasets"][kind]
    raw_dir = cfg.raw_dir(city)
    files = find_files(raw_dir, spec.get("file_glob", []))
    if not files:
        raise FileNotFoundError(
            f"[{city}/{kind}] 원본 CSV 를 찾지 못했다.\n"
            f"  찾은 위치: {raw_dir}\n"
            f"  패턴: {spec.get('file_glob')}\n"
            f"  docs/DATA_SOURCES.md 의 링크에서 받아 이 폴더에 넣어라."
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        df = read_csv_any(path)
        # 플랫폼 CSV 의 첫 컬럼에는 BOM 이 붙어 오고, 값에는 자리맞춤 공백이 붙어 온다.
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        ctx = f"{city}/{kind}/{path.name}"
        cols = resolve_columns(df, spec.get("required", {}), required=True, context=ctx)
        cols |= resolve_columns(df, spec.get("optional", {}), required=False, context=ctx)
        sub = df[list(cols.values())].copy()
        sub.columns = list(cols.keys())
        sub = strip_object_columns(sub)
        sub["_source_file"] = path.name
        frames.append(sub)
        log.info("%s: %d행 %d컬럼", ctx, len(sub), len(cols))

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["city"] = city
    return out


def strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """문자열 컬럼의 앞뒤 공백을 떼고 빈 문자열을 결측으로 바꾼다.

    원본은 고정폭으로 채워져 있어 '울산 ' 과 '울산' 이 다른 값이 된다.
    이걸 그대로 두면 격자 집계가 조용히 두 갈래로 갈라진다.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            cleaned = out[col].astype(str).str.strip()
            out[col] = cleaned.mask(cleaned.isin(["", "nan", "None", "NaN"]), pd.NA)
    return out
