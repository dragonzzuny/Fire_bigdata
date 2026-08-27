# docs/ 안내

이 폴더에는 **두 종류의 문서**가 섞여 있습니다.
`AGENTS.md` 와 `evals/`, `.claude/skills/` 가 아래 하네스 문서를 경로로 참조하므로
옮기지 않고 여기에 표시만 해 둡니다.

## 불씨예보(K-Firebird) 프로젝트 문서

| 파일 | 내용 |
|---|---|
| [DATA_SOURCES.md](DATA_SOURCES.md) | 원본 데이터 내려받는 곳과 배치 위치 |
| [DATA_REALITY.md](DATA_REALITY.md) | 기획서가 가정한 것과 원본 실측이 다른 지점 10건 |
| [REPRODUCTION.md](REPRODUCTION.md) | 재현 상태와 실행 결과 |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | 10분 발표·시연 대본과 예상 질문 |
| [refs/](refs/) | 인용 문헌 (애틀랜타 Firebird, KDD 2016) |

## dobby 하네스 문서 (이 프로젝트와 무관)

저장소 루트에 함께 설치된 에이전트 하네스(`dobby/`, `evals/`, `mcp/`, `tests/`)의
문서입니다. 불씨예보의 성능·설계와는 관계가 없습니다.

`PROJECT.md` · `RUNTIME.md` · `OPERATING_MANUAL.md` · `EVAL_DESIGN.md` ·
`BENCHMARK_LANDSCAPE.md` · `RESEARCH_EVIDENCE_MATRIX.md` · `FAILURE_CATALOG.md` ·
`THREAT_MODEL.md`

> 이 문서들 안의 수치(예: GAIA, SWE-Bench 점수)는 하네스 평가에 관한 것이며
> 불씨예보의 검증 결과가 아닙니다. 불씨예보의 수치는 `outputs/evaluation.json`
> 하나가 원본입니다.
