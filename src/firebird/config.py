"""설정 로딩. YAML 한 곳에서 전 파이프라인 파라미터를 읽는다."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """repo 루트. 이 파일 기준 src/firebird/config.py -> 두 단계 위."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    cache: Path
    outputs: Path
    figures: Path

    def ensure(self) -> "Paths":
        for p in (self.interim, self.processed, self.cache, self.outputs, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    schema: dict[str, Any]
    paths: Paths

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # --- 자주 쓰는 값은 이름을 붙여둔다 ---
    @property
    def grid_size_m(self) -> int:
        return int(self.raw["grid"]["size_m"])

    @property
    def crs_metric(self) -> str:
        return self.raw["grid"]["crs_metric"]

    @property
    def crs_geographic(self) -> str:
        return self.raw["grid"]["crs_geographic"]

    @property
    def split_year(self) -> int:
        return int(self.raw["years"]["split_year"])

    @property
    def holdout_year(self) -> int:
        return int(self.raw["years"]["holdout_year"])

    @property
    def year_min(self) -> int:
        return int(self.raw["years"]["min"])

    @property
    def year_max(self) -> int:
        return int(self.raw["years"]["max"])

    @property
    def headline_k(self) -> int:
        return int(self.raw["evaluation"]["headline_k"])

    def city(self, name: str) -> dict[str, Any]:
        cities = self.raw["cities"]
        if name not in cities:
            raise KeyError(f"알 수 없는 도시 '{name}'. configs/config.yaml 의 cities: {list(cities)}")
        return cities[name]

    def raw_dir(self, city: str) -> Path:
        return self.paths.raw / self.city(city)["raw_subdir"]


def load_config(config_path: str | os.PathLike | None = None,
                schema_path: str | os.PathLike | None = None) -> Config:
    root = project_root()
    config_path = Path(config_path) if config_path else root / "configs" / "config.yaml"
    schema_path = Path(schema_path) if schema_path else root / "configs" / "schema.yaml"

    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    with open(schema_path, encoding="utf-8") as fh:
        schema = yaml.safe_load(fh)

    # FIREBIRD_DATA_ROOT 로 데이터 루트를 통째로 옮길 수 있다.
    # 테스트가 실제 data/ 를 건드리지 않고 파이프라인 전체를 돌리기 위한 통로이며,
    # 운영에서는 데이터를 외장 디스크에 두고 코드만 repo 에 두는 데 쓴다.
    data_root = Path(os.environ.get("FIREBIRD_DATA_ROOT", root))
    p = raw["paths"]
    paths = Paths(
        root=root,
        raw=data_root / p["raw"],
        interim=data_root / p["interim"],
        processed=data_root / p["processed"],
        cache=data_root / p["cache"],
        outputs=data_root / p["outputs"],
        figures=data_root / p["figures"],
    ).ensure()

    return Config(raw=raw, schema=schema, paths=paths)
