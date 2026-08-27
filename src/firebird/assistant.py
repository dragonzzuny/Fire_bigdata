"""화재예방 업무 질의응답 — 법령 조문과 우리 데이터를 근거로 답한다.

두 종류의 질문이 온다.

  "3급 대상물 자체점검 주기가 어떻게 되죠?"      -> 법령 조문
  "삼산119안전센터 관할 이번 달 어디부터 돌죠?"   -> 우리 데이터

그래서 검색 대상을 둘 다 둔다. 그리고 **근거를 반드시 함께 돌려준다.**
조문 번호 없이 "연 1회입니다"라고만 답하는 시스템은 행정에서 쓸 수 없다.
담당자가 조문을 열어 확인할 수 있어야 한다.

검색은 BM25 로 한다. 임베딩 모델을 붙이면 의미 검색이 되지만, 이 저장소는
파이썬 표준 라이브러리와 몇 개 패키지만으로 도는 것을 원칙으로 한다.
법령 질의는 용어가 그대로 등장하는 경우가 많아 어휘 검색으로 충분하다.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import llm as L

log = logging.getLogger(__name__)

#: 한국어 조사·접미어. 붙은 채로는 '점검주기가' 와 '점검주기' 가 다른 낱말이 된다.
_JOSA = ("으로써", "에서는", "에게서", "이라도", "까지도", "부터는", "에서의",
         "으로는", "이라는", "라는", "에서", "에게", "으로", "부터", "까지",
         "보다", "처럼", "이나", "거나", "이며", "하고", "과의", "와의",
         "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로", "와", "과")

_TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """어절에서 조사를 떼고, 긴 낱말은 앞부분도 함께 색인한다."""
    out: list[str] = []
    for w in _TOKEN.findall(str(text).lower()):
        out.append(w)
        for j in _JOSA:
            if len(w) > len(j) + 1 and w.endswith(j):
                out.append(w[: -len(j)])
                break
        # 복합명사 대응: '소방안전관리자' 로 물어도 '소방안전관리' 가 걸리게
        if len(w) >= 5:
            out.append(w[:3])
            out.append(w[:4])
    return out


@dataclass
class Doc:
    doc_id: str
    source: str          # '법령' | '데이터' | '업무규칙'
    title: str
    text: str
    ref: str = ""        # 조문 번호 등 근거 표기


class BM25:
    """Okapi BM25. 문서 수가 수백~수천 규모라 순수 파이썬으로 충분하다."""

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tokens = [tokenize(f"{d.title} {d.text}") for d in docs]
        self.lengths = np.array([len(t) for t in self.tokens], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.tf = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[Doc, float]]:
        q = tokenize(query)
        if not q or not self.docs:
            return []
        scores = np.zeros(len(self.docs))
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg_len, 1))
                scores[i] += idf * f * (self.k1 + 1) / denom
        order = np.argsort(-scores)[:top_k]
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > 0]


# ---------------------------------------------------------------- 문서 구성

def law_docs(articles: pd.DataFrame) -> list[Doc]:
    if articles is None or articles.empty:
        return []
    return [Doc(doc_id=f"law:{r.law}:{r.article}", source="법령",
                title=f"{r.law} {r.article}" + (f"({r.title})" if r.title else ""),
                text=str(r.text), ref=f"{r.law} {r.article}")
            for r in articles.itertuples()]


def annex_docs(forms: pd.DataFrame) -> list[Doc]:
    """별표 본문도 검색 대상에 넣는다.

    소방안전관리자 선임 대상 범위, 점검 주기 같은 실무 기준은 조문이 아니라
    별표에 적혀 있다. 조문만 색인하면 "별표에 있는데 자료에 없습니다"라는
    답이 반복되어 정작 필요한 답을 못 준다.
    """
    if forms is None or forms.empty or "content" not in forms.columns:
        return []
    out = []
    for r in forms.itertuples():
        text = str(getattr(r, "content", "") or "").strip()
        if len(text) < 40:
            continue
        ref = f"{r.law} [{r.kind} {r.no}]"
        out.append(Doc(doc_id=f"annex:{ref}", source="법령",
                       title=f"{ref} {r.title}", text=text, ref=ref))
    return out


def rule_docs() -> list[Doc]:
    """우리가 코드로 갖고 있는 업무 규칙도 검색 대상에 넣는다."""
    from .patrol_modes import MODES
    from .rules import BUSINESS_RULES, FACILITY_RULES, USAGE_RULES

    docs: list[Doc] = []
    for biz, r in BUSINESS_RULES.items():
        docs.append(Doc(doc_id=f"rule:biz:{biz}", source="업무규칙",
                        title=f"{biz} 점검 항목",
                        text=f"{r['근거']}\n" + "\n".join(r["항목"]),
                        ref=f"업종별 점검 규칙 · {biz}"))
    for fac, items in FACILITY_RULES.items():
        docs.append(Doc(doc_id=f"rule:fac:{fac}", source="업무규칙",
                        title=f"{fac} 설치대상 확인 항목",
                        text="\n".join(items), ref=f"소방시설별 점검 규칙 · {fac}"))
    for usage, items in USAGE_RULES.items():
        docs.append(Doc(doc_id=f"rule:usage:{usage}", source="업무규칙",
                        title=f"{usage} 용도 확인 항목",
                        text="\n".join(items), ref=f"용도별 점검 규칙 · {usage}"))
    for key, mode in MODES.items():
        docs.append(Doc(doc_id=f"rule:patrol:{key}", source="업무규칙",
                        title=f"{mode.label}",
                        text=f"{mode.purpose}\n" + "\n".join(mode.checks),
                        ref=f"순찰 유형 · {mode.label}"))
    return docs


def data_docs(panel_year: pd.DataFrame, *, city_label: str = "", year: int | None = None,
              level: str = "center", top_n: int = 40) -> list[Doc]:
    """우리 분석 결과를 문장으로 만들어 검색 대상에 넣는다.

    "삼산119안전센터 관할에서 어디가 위험하죠" 같은 질문에 답하려면
    데이터도 검색 가능한 형태여야 한다.
    """
    if panel_year.empty:
        return []
    docs: list[Doc] = []
    df = panel_year.copy()
    if "pred" not in df.columns:
        return []

    # 관서별 요약
    if level in df.columns:
        for name, g in df.groupby(level):
            if not str(name).strip():
                continue
            top = g.nlargest(5, "pred")
            lines = [f"{city_label} {name} 관할 {year}년 화재위험 요약",
                     f"관할 격자 {g['grid_id'].nunique()}개, "
                     f"예측 화재 {g['pred'].sum():.1f}건"]
            if "target_total" in g:
                lines.append(f"점검 대상물 {g['target_total'].sum():.0f}개소")
            if "n_hydrant" in g:
                none_h = int((g["n_hydrant"] <= 0).sum())
                lines.append(f"소화전 없는 격자 {none_h}개")
            lines.append("위험 상위 격자: " + ", ".join(
                f"{r.grid_id}(위험 {r.pred:.2f})" for r in top.itertuples()))
            docs.append(Doc(doc_id=f"data:{level}:{name}", source="데이터",
                            title=f"{name} 관할 현황", text="\n".join(lines),
                            ref=f"{city_label} {year}년 분석 결과"))

    # 읍면동별 요약
    if "emd" in df.columns:
        for name, g in df.groupby("emd"):
            if not str(name).strip():
                continue
            docs.append(Doc(
                doc_id=f"data:emd:{name}", source="데이터",
                title=f"{name} 화재위험",
                text=(f"{city_label} {name} {year}년 예측 화재 {g['pred'].sum():.1f}건, "
                      f"격자 {g['grid_id'].nunique()}개. "
                      f"관할 소방서 {g['station'].mode().iloc[0] if 'station' in g and len(g['station'].mode()) else ''}. "
                      f"위험 상위 격자 {', '.join(g.nlargest(3,'pred')['grid_id'])}"),
                ref=f"{city_label} {year}년 분석 결과"))

    # 최상위 위험 격자
    for r in df.nlargest(top_n, "pred").itertuples():
        parts = [f"격자 {r.grid_id} 위험점수 상위. 예측 화재 {r.pred:.2f}건"]
        for col, label in (("sgg", "시군구"), ("emd", "읍면동"),
                           ("station", "소방서"), ("center", "119안전센터")):
            v = getattr(r, col, "")
            if v:
                parts.append(f"{label} {v}")
        if getattr(r, "target_total", None):
            parts.append(f"대상물 {r.target_total:.0f}개소")
        if getattr(r, "biz_total", None):
            parts.append(f"다중이용업소 {r.biz_total:.0f}개소")
        docs.append(Doc(doc_id=f"data:grid:{r.grid_id}", source="데이터",
                        title=f"격자 {r.grid_id}", text=", ".join(parts),
                        ref=f"{city_label} {year}년 분석 결과"))
    return docs


def build_index(articles: pd.DataFrame | None = None,
                panel_year: pd.DataFrame | None = None,
                annexes: pd.DataFrame | None = None, **kw) -> BM25:
    docs = law_docs(articles) + annex_docs(annexes) + rule_docs()
    if panel_year is not None:
        docs += data_docs(panel_year, **kw)
    return BM25(docs)


# ---------------------------------------------------------------- 답변

SYSTEM = (
    "당신은 소방서 화재예방 담당자를 돕는 업무 보조자입니다. "
    "아래 '참고 자료' 안의 내용만으로 답하십시오. "
    "자료에 없는 내용은 지어내지 말고 '제공된 자료에서 확인되지 않습니다'라고 하십시오. "
    "법령을 인용할 때는 반드시 법령명과 조문 번호를 함께 쓰십시오. "
    "숫자(점검 주기, 기준값, 과태료 등)는 자료에 적힌 그대로만 쓰십시오. "
    "간결한 실무 문체로 답하십시오."
)


def build_prompt(question: str, hits: list[tuple[Doc, float]]) -> str:
    lines = [SYSTEM, "", "[참고 자료]"]
    for i, (d, score) in enumerate(hits, 1):
        lines.append(f"\n({i}) [{d.source}] {d.title}")
        if d.ref:
            lines.append(f"    근거: {d.ref}")
        body = d.text if len(d.text) <= 1800 else d.text[:1800] + " …"
        lines.append("    " + body.replace("\n", "\n    "))
    lines += ["", f"[질문] {question}", "",
              "위 자료만 근거로 답하고, 사용한 자료의 번호와 근거를 문장 안에 밝히십시오."]
    return "\n".join(lines)


#: 답변에 나온 법령 인용을 골라내는 규칙. 「법령명」 제N조(의N) 제N항 제N호, 별표 N.
_CITE_RE = re.compile(r"「([^」]{2,60})」|제\s*(\d+)\s*조(?:\s*의\s*(\d+))?|별표\s*(\d+)")


def _citations(text: str) -> set[str]:
    """인용된 법령명과 조문 번호를 표준 표기로 모은다."""
    out: set[str] = set()
    for law, art, sub, annex in _CITE_RE.findall(str(text)):
        if law:
            out.add("법:" + re.sub(r"\s+", "", law))
        elif art:
            out.add(f"조:{art}의{sub}" if sub else f"조:{art}")
        elif annex:
            out.add(f"별표:{annex}")
    return out


def check_grounding(answer_text: str, hits: list[tuple["Doc", float]]) -> dict:
    """답변이 인용한 조문이 실제로 검색된 자료 안에 있는지 대조한다.

    RAG 라도 모델은 '있을 법한' 조문을 만들어 낸다. 소방 실무에서 조문 번호가
    한 자리 틀리면 그 답변은 쓸모가 없는 정도가 아니라 위험하다. 그래서 답변에
    등장한 법령명·조문 번호를 전부 뽑아, 검색해 온 원문에 그 표기가 없으면
    **근거 없는 인용**으로 표시한다. 지우지는 않는다 — 사람이 보고 판단해야 한다.

    대조는 공백을 지운 원문에 대한 포함 검사로 한다. 원문은 법령명을 「」 없이
    적고 답변은 「」로 감싸는 등 표기가 다를 뿐인데, 그 차이를 위반으로 세면
    경고가 쏟아져 정작 진짜 위반이 묻힌다.
    """
    corpus = " ".join(f"{d.title} {d.ref or ''} {d.text}" for d, _ in hits)
    flat = re.sub(r"\s+", "", corpus)
    used = _citations(answer_text)

    unsupported = []
    for tok in sorted(used):
        kind, _, val = tok.partition(":")
        if kind == "법":
            ok = val in flat
        elif kind == "조":
            base, _, sub = val.partition("의")
            ok = (f"제{base}조의{sub}" in flat) if sub else (f"제{base}조" in flat)
        else:                                   # 별표
            ok = f"별표{val}" in flat
        if not ok:
            unsupported.append(tok)
    return {
        "ok": not unsupported,
        "cited": sorted(used),
        "unsupported": unsupported,
        "n_sources": len(hits),
    }


def answer(cfg, question: str, index: BM25, *, top_k: int = 5,
           use_llm: bool = True) -> dict:
    """질문 -> {answer, sources, prompt, source_backend}.

    LLM 이 없으면 검색 결과만 돌려준다. 근거 조문을 보여주는 것만으로도
    쓸모가 있고, 없는 답을 지어내는 것보다 낫다.
    """
    hits = index.search(question, top_k=top_k)
    sources = [{"source": d.source, "title": d.title, "ref": d.ref,
                "score": round(s, 2),
                "excerpt": d.text[:280] + ("…" if len(d.text) > 280 else "")}
               for d, s in hits]
    if not hits:
        return {"answer": "관련 자료를 찾지 못했습니다. 질문을 다르게 표현해 보십시오.",
                "sources": [], "prompt": None, "source_backend": "no_match"}

    if not use_llm or not L.is_available(cfg):
        summary = "\n\n".join(f"· {d.title}\n  {d.text[:400]}" for d, _ in hits[:3])
        return {"answer": "관련 근거를 찾았습니다. 아래 자료를 확인하십시오.\n\n" + summary,
                "sources": sources, "prompt": None, "source_backend": "search_only"}

    prompt = build_prompt(question, hits)
    text, backend, tried = L.generate(cfg, prompt)
    if not text:
        summary = "\n\n".join(f"· {d.title}\n  {d.text[:400]}" for d, _ in hits[:3])
        return {"answer": "관련 근거를 찾았습니다. 아래 자료를 확인하십시오.\n\n" + summary,
                "sources": sources, "prompt": prompt, "source_backend": "search_only",
                "tried": tried}
    ground = check_grounding(text, hits)
    if not ground["ok"]:
        log.warning("자료에 없는 인용: %s", ground["unsupported"])
    return {"answer": text, "sources": sources, "prompt": prompt,
            "source_backend": backend, "grounding": ground}
