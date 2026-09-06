"""점검계획서 문서화용 LLM 백엔드.

원칙은 하나다. **LLM 은 문서화만 한다.** 위험 판단(모델)과 점검 항목(규칙)은
이미 확정된 입력으로 넘기고, 그 밖의 사실을 새로 만들지 말라고 지시한다.
LLM 이 점검 항목이나 법령 조문을 지어내면 그건 행정 문서의 오류가 된다.

백엔드는 설정 순서대로 시도하고 되는 첫 번째를 쓴다:
    cli(codex 등) -> api(OpenAI 호환) -> ollama(로컬) -> rule_based(무의존)
어느 경로로 만들어졌는지는 산출물에 항상 함께 기록한다. 사람이 쓴 문서와
기계가 쓴 문서를 나중에 구분할 수 없으면 검수 자체가 불가능하기 때문이다.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any

import requests

log = logging.getLogger(__name__)

SYSTEM_RULE = (
    "당신은 소방서 예방과의 문서 작성 보조자다. "
    "주어진 '위험요인'과 '점검항목' 밖의 사실을 새로 만들지 마라. "
    "법령 조문 번호, 통계 수치, 건물명을 임의로 지어내지 마라. "
    "공문 문체로, 항목을 빠뜨리지 말고 정리하라."
)


# ------------------------------------------------------------------ 가용성

def _cli_argv(cfg) -> list[str] | None:
    cmd = cfg["llm"].get("cli", {}).get("command") or []
    if not cmd:
        return None
    return list(cmd)


def cli_available(cfg) -> bool:
    argv = _cli_argv(cfg)
    return bool(argv) and shutil.which(argv[0]) is not None


def api_available(cfg) -> bool:
    env = cfg["llm"].get("api", {}).get("key_env")
    return bool(env) and bool(os.environ.get(env))


def ollama_available(cfg) -> bool:
    ep = cfg["llm"].get("ollama", {}).get("endpoint")
    if not ep:
        return False
    base = str(ep).rsplit("/api/", 1)[0]
    try:
        return requests.get(f"{base}/api/tags", timeout=3).status_code == 200
    except requests.RequestException:
        return False


#: 이름 -> 가용성 검사. 람다로 감싸 이름을 호출 시점에 찾게 한다.
#: 함수 객체를 직접 담으면 임포트 시점에 묶여, 나중에 이 모듈의 함수를
#: 바꿔치기해도(테스트·확장) 이 표는 옛 함수를 계속 가리킨다.
CHECKS = {"cli": lambda cfg: cli_available(cfg),
          "api": lambda cfg: api_available(cfg),
          "ollama": lambda cfg: ollama_available(cfg),
          "rule_based": lambda cfg: True}


def available_backends(cfg) -> list[str]:
    """설정 순서 그대로, 지금 실제로 쓸 수 있는 백엔드."""
    return [b for b in cfg["llm"].get("backends", ["rule_based"])
            if CHECKS.get(b, lambda _c: False)(cfg)]


def is_available(cfg) -> bool:
    """규칙기반 말고 진짜 LLM 이 하나라도 있는가."""
    return any(b != "rule_based" for b in available_backends(cfg))


# ------------------------------------------------------------------ 호출

def _call_cli(cfg, prompt: str) -> tuple[str, str]:
    conf = cfg["llm"]["cli"]
    argv = _cli_argv(cfg) or []
    full = str(conf.get("preamble", "")) + prompt
    stdin_text = None

    if conf.get("prompt_as_stdin"):
        stdin_text = full
    elif any("{prompt}" in a for a in argv):
        argv = [a.replace("{prompt}", full) for a in argv]
    else:
        argv = argv + [full]

    try:
        r = subprocess.run(argv, input=stdin_text, capture_output=True, text=True,
                           timeout=int(cfg["llm"].get("timeout_s", 180)))
    except (subprocess.TimeoutExpired, OSError) as exc:
        return "", f"cli_error:{type(exc).__name__}"
    if r.returncode != 0:
        log.warning("CLI(%s) 실패 rc=%d: %s", argv[0], r.returncode, r.stderr[-300:])
        return "", f"cli_rc{r.returncode}"
    text = r.stdout.strip()
    return (text, f"cli:{argv[0]}") if text else ("", "cli_empty")


def _call_api(cfg, prompt: str) -> tuple[str, str]:
    conf = cfg["llm"]["api"]
    key = os.environ.get(str(conf.get("key_env", "")), "")
    if not key:
        return "", "api_no_key"
    try:
        r = requests.post(conf["endpoint"],
                          headers={"Authorization": f"Bearer {key}"},
                          timeout=int(cfg["llm"].get("timeout_s", 180)),
                          json={"model": conf["model"],
                                "messages": [{"role": "system", "content": SYSTEM_RULE},
                                             {"role": "user", "content": prompt}]})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        log.warning("API 호출 실패: %s", exc)
        return "", f"api_error:{type(exc).__name__}"
    return (text, f"api:{conf['model']}") if text else ("", "api_empty")


def _call_ollama(cfg, prompt: str) -> tuple[str, str]:
    conf = cfg["llm"]["ollama"]
    try:
        r = requests.post(conf["endpoint"], timeout=int(cfg["llm"].get("timeout_s", 180)),
                          json={"model": conf["model"], "prompt": prompt,
                                "system": SYSTEM_RULE, "stream": False})
        r.raise_for_status()
        text = str(r.json().get("response", "")).strip()
    except (requests.RequestException, ValueError) as exc:
        log.warning("Ollama 호출 실패: %s", exc)
        return "", f"ollama_error:{type(exc).__name__}"
    return (text, f"ollama:{conf['model']}") if text else ("", "ollama_empty")


GENERATORS = {"cli": lambda cfg, p: _call_cli(cfg, p),
              "api": lambda cfg, p: _call_api(cfg, p),
              "ollama": lambda cfg, p: _call_ollama(cfg, p)}


def generate(cfg, prompt: str) -> tuple[str, str, list[str]]:
    """(본문, 생성경로, 시도기록). 전부 실패하면 본문은 빈 문자열."""
    tried: list[str] = []
    for backend in cfg["llm"].get("backends", []):
        if backend == "rule_based":
            break
        if not CHECKS.get(backend, lambda _c: False)(cfg):
            tried.append(f"{backend}:unavailable")
            continue
        text, source = GENERATORS[backend](cfg, prompt)
        if text:
            return text, source, tried
        tried.append(source)
    return "", "", tried


# ------------------------------------------------------------------ 프롬프트

def build_prompt(grid_id: str, risk: dict, drivers: list[dict], checklist: dict) -> str:
    lines = [
        SYSTEM_RULE,
        "",
        f"[대상 구역] 격자 {grid_id}",
        f"[위험도] 위험점수 {risk.get('score_0_100', 0):.0f}/100, "
        f"관내 순위 {risk.get('rank', '?')}위 (상위 {risk.get('percentile', 0):.1f}%)",
        "",
        "[모델이 짚은 위험 상승 요인]",
    ]
    for d in drivers or []:
        val = d.get("value_text") or f"{d['value']:.4g}"
        lines.append(f"  - {d['label']}: 현재값 {val} (기여도 {d['contribution']:+.3f})")
    if not drivers:
        lines.append("  - (제공된 요인 없음)")
    lines += ["", "[점검 항목]"]
    for sec in checklist.get("sections", []):
        lines.append(f"  · {sec['구분']} — {sec['근거']}")
        for item in sec["항목"]:
            lines.append(f"      - {item}")
    lines += [
        "",
        "위 내용만 사용해서 다음 형식의 점검계획서 초안을 작성하라:",
        "1) 점검 개요 (대상 구역, 선정 사유) 2) 중점 확인 사항 3) 점검 항목 체크리스트 "
        "4) 유의사항. 각 항목은 위 입력에 근거해야 한다. 다른 설명은 붙이지 마라.",
    ]
    return "\n".join(lines)


def rule_based_draft(grid_id: str, risk: dict, drivers: list[dict], checklist: dict) -> str:
    """LLM 없이도 바로 쓸 수 있는 초안. 폴백이자 LLM 출력의 기준선."""
    out = [
        f"■ 화재예방 점검계획서(초안) — 격자 {grid_id}",
        "",
        "1) 점검 개요",
        f"   - 대상: 격자 {grid_id} (500m 단위 구역)",
        f"   - 위험도: 위험점수 {risk.get('score_0_100', 0):.0f}/100, "
        f"상위 {risk.get('percentile', 0):.1f}% 구역",
        "   - 선정 사유: 화재위험 예측 모델의 상위 위험 구역으로 산출됨",
        "",
        "2) 중점 확인 사항 (모델이 짚은 위험 상승 요인)",
    ]
    for d in drivers or []:
        val = d.get("value_text") or f"{d['value']:.4g}"
        out.append(f"   - {d['label']} (현재값 {val})")
    if not drivers:
        out.append("   - (해당 없음)")
    out += ["", "3) 점검 항목"]
    for sec in checklist.get("sections", []):
        out.append(f"   [{sec['구분']}] {sec['근거']}")
        for item in sec["항목"]:
            out.append(f"      □ {item}")
    out += [
        "",
        "4) 유의사항",
        "   - 본 초안은 공개 데이터 기반 예측 결과이며, 법정 점검주기·관할 판단을 대체하지 않는다.",
        "   - 격자 단위 산출물로 개별 건물을 특정하지 않는다.",
    ]
    return "\n".join(out)


def inspection_plan(cfg, grid_id: str, risk: dict, drivers: list[dict],
                    checklist: dict, *, use_llm: bool | None = None) -> dict[str, Any]:
    """점검계획서 한 건. 어느 경로로 만들어졌는지 항상 함께 돌려준다."""
    fallback = rule_based_draft(grid_id, risk, drivers, checklist)
    if use_llm is False:
        return {"grid_id": grid_id, "text": fallback, "source": "rule_based",
                "prompt": None, "tried": ["disabled"]}

    prompt = build_prompt(grid_id, risk, drivers, checklist)
    text, source, tried = generate(cfg, prompt)
    if not text:
        return {"grid_id": grid_id, "text": fallback,
                "source": "rule_based", "prompt": prompt, "tried": tried}
    return {"grid_id": grid_id, "text": text, "source": source,
            "prompt": prompt, "tried": tried}
