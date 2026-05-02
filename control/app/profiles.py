"""Load and validate the profiles.yaml config."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


PROFILES_PATH = Path(os.environ.get("QWEN_CONTROL_PROFILES", "/etc/qwen-control/profiles.yaml"))


@dataclass(frozen=True)
class Profile:
    name: str
    display_name: str
    served_model_id: str
    upstream: str
    launcher: str
    container_names: tuple[str, ...]
    ready_url: str
    expected_load_seconds: int
    description: str


@dataclass(frozen=True)
class Config:
    profiles: dict[str, Profile]
    gpu_competitors: tuple[str, ...]


def load_config(path: Optional[Path] = None) -> Config:
    p = path or PROFILES_PATH
    if not p.exists():
        raise RuntimeError(f"profiles.yaml not found at {p}")
    raw = yaml.safe_load(p.read_text()) or {}

    raw_profiles = raw.get("profiles") or {}
    profiles: dict[str, Profile] = {}
    for name, body in raw_profiles.items():
        if not isinstance(body, dict):
            raise ValueError(f"profile {name!r}: expected mapping, got {type(body).__name__}")
        try:
            profiles[name] = Profile(
                name=name,
                display_name=str(body["display_name"]),
                served_model_id=str(body["served_model_id"]),
                upstream=str(body["upstream"]).rstrip("/"),
                launcher=str(body["launcher"]),
                container_names=tuple(body.get("container_names") or ()),
                ready_url=str(body["ready_url"]),
                expected_load_seconds=int(body.get("expected_load_seconds") or 90),
                description=str(body.get("description") or "").strip(),
            )
        except KeyError as e:
            raise ValueError(f"profile {name!r}: missing required field {e}")

    competitors = tuple(raw.get("gpu_competitors") or ())
    return Config(profiles=profiles, gpu_competitors=competitors)
