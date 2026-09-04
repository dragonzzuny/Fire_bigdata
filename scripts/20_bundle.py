#!/usr/bin/env python
"""제출·발표용 묶음 만들기.

발표장에 들고 갈 것과 심사에 낼 것을 한 폴더로 모은다. 손으로 모으면
빠뜨리거나 옛 파일이 섞인다. 실제로 리허설 카드가 이름이 바뀐 옛 영상
파일을 가리키고 있던 적이 있다.

중간 산물(원본 녹화, _crop 이미지, 학습된 모델)은 넣지 않는다.
받는 사람이 열어 볼 것만 넣는다.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firebird.config import load_config  # noqa: E402

#: (묶음 안 폴더, 원본 경로 목록). 없는 파일은 건너뛰고 끝에 알린다.
def layout(cfg) -> list[tuple[str, list[Path]]]:
    out = cfg.paths.outputs
    figs = cfg.paths.figures
    docs = ROOT / "docs"
    vid = out / "demo_video"
    return [
        ("01_발표자료", [
            *out.glob("불씨예보_발표자료_*.pptx"),
            *out.glob("불씨예보_발표자료_*.pdf"),
        ]),
        ("02_영상", [
            vid / "시연_핵심.mp4",
            vid / "시연_핵심_무자막.mp4",
            vid / "시연_전체.mp4",
        ]),
        ("03_대본", [
            docs / "DEMO_SCRIPT.md",
            docs / "REHEARSAL_CARD.md",
        ]),
        ("04_기획서", [
            *ROOT.glob("*경진대회 기획서*.hwp"),
            *ROOT.glob("*경진대회 기획서*.pdf"),
        ]),
        ("05_데이터제안", [
            docs / "DATA_PROPOSAL.md",
        ]),
        ("06_그림", [p for p in sorted(figs.glob("*.png"))
                   if "_crop" not in p.name]),
        ("07_산출물", [
            out / "evaluation.json",
            *out.glob("artifacts_summary_*.json"),
            *out.glob("backtest_patrol_*.json"),
            *out.glob("building_feature_eval_*.json"),
            *out.glob("*.csv"),
        ]),
        ("08_계획서_예시", sorted((out / "inspection_plans").glob("*.txt"))[:5]),
        ("09_설명", [
            ROOT / "README.md",
            docs / "DATA_REALITY.md",
            docs / "REPRODUCTION.md",
        ]),
    ]


def readme(cfg, counts: dict, video_secs: dict) -> str:
    def mmss(s: float) -> str:
        return f"{int(s // 60)}분 {int(s % 60):02d}초"
    lines = [
        "# 불씨예보(K-Firebird) 발표 묶음",
        "",
        "제6회 소방안전 빅데이터 활용 및 아이디어 경진대회 · 서비스 개발 부문",
        "박용준 (아주대학교 산업공학과 석사과정)",
        "",
        f"만든 날: {date.today():%Y-%m-%d}",
        "",
        "## 발표장에서 쓰는 것",
        "",
        "| 폴더 | 무엇 |",
        "|---|---|",
        "| `01_발표자료` | 장표 19장. PPTX 와 PDF 둘 다 |",
        f"| `02_영상` | 시연 대체 영상. **핵심 {mmss(video_secs.get('시연_핵심.mp4', 0))}** 를 튼다 |",
        "| `03_대본` | 「실제로 말할 대본」 과 리허설 카드 |",
        "",
        "## 심사·검토용",
        "",
        "| 폴더 | 무엇 |",
        "|---|---|",
        "| `04_기획서` | 1차 제출 기획서 원본 |",
        "| `05_데이터제안` | 데이터 제안 부문 제출 원고 |",
        "| `06_그림` | 장표에 쓴 그림과 화면 캡처 |",
        "| `07_산출물` | 발표에 나온 수치의 출처가 되는 파일 |",
        "| `08_계획서_예시` | 서비스가 만든 순찰·점검 계획서 실물 |",
        "| `09_설명` | 저장소 안내, 자료의 실제 상태, 재현 절차 |",
        "",
        "## 영상 세 벌의 차이",
        "",
        "| 파일 | 길이 | 쓰임 |",
        "|---|---|---|",
    ]
    for name, use in (("시연_핵심.mp4", "발표에서 트는 것"),
                      ("시연_핵심_무자막.mp4", "발표자가 직접 말로 끌 때"),
                      ("시연_전체.mp4", "여섯 화면 전부")):
        if name in video_secs:
            lines.append(f"| `{name}` | {mmss(video_secs[name])} | {use} |")
    lines += [
        "",
        "## 유의할 점",
        "",
        "- 영상 화면비는 16:10 입니다. 16:9 화면에서는 좌우에 검은 띠가 생깁니다.",
        "- 소리는 없습니다. 자막이 설명을 맡습니다.",
        "- `07_산출물` 의 수치는 모두 저장소의 스크립트가 낸 값입니다.",
        "  `09_설명/REPRODUCTION.md` 의 절차로 다시 만들 수 있습니다.",
        "",
        "## 담긴 파일 수",
        "",
    ]
    for folder, n in counts.items():
        lines.append(f"- `{folder}` {n}개")
    return "\n".join(lines) + "\n"


def video_lengths(paths: list[Path]) -> dict:
    out = {}
    for p in paths:
        if p.suffix != ".mp4" or not p.exists():
            continue
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True)
        try:
            out[p.name] = float(r.stdout.strip())
        except ValueError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="불씨예보_발표묶음")
    args = ap.parse_args()

    cfg = load_config()
    plan = layout(cfg)
    stage = cfg.paths.outputs / "_bundle" / args.name
    if stage.parent.exists():
        shutil.rmtree(stage.parent)
    stage.mkdir(parents=True)

    counts, missing = {}, []
    vids = []
    for folder, paths in plan:
        dst = stage / folder
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in paths:
            if not p.exists():
                missing.append(str(p.relative_to(ROOT)))
                continue
            shutil.copy2(p, dst / p.name)
            n += 1
            if p.suffix == ".mp4":
                vids.append(p)
        counts[folder] = n

    (stage / "00_먼저읽기.md").write_text(
        readme(cfg, counts, video_lengths(vids)), encoding="utf-8")

    zip_path = cfg.paths.outputs / f"{args.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage.parent))

    print(f"묶음: {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)\n")
    for folder, n in counts.items():
        print(f"  {folder:<16} {n:>3}개")
    if missing:
        print(f"\n  ※ 없어서 넣지 못한 것 {len(missing)}개")
        for m in missing:
            print("     ·", m)
    shutil.rmtree(stage.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
