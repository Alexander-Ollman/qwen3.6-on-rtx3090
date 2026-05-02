"""State machine for the qwen-control plane.

States: off | switching | active(profile_name) | error(msg)

The orchestrator is the single source of truth for which model is up. The
proxy layer reads `state.active` to know where to forward /v1/* requests.
During a switch, requests get a 503 with Retry-After.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from . import db, docker_ops
from .profiles import Config, Profile


log = logging.getLogger("qwen-control.orchestrator")


class Status(str, enum.Enum):
    OFF = "off"
    SWITCHING = "switching"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class SwitchProgress:
    """Live progress of a switch; updated by the orchestrator, read by the UI."""
    target: Optional[str] = None
    started_at: Optional[float] = None
    expected_seconds: int = 0
    current_step: str = ""
    log_lines: list[str] = field(default_factory=list)

    def add_step(self, msg: str) -> None:
        self.current_step = msg
        self.log_lines.append(f"{int(time.time())} {msg}")
        # cap log buffer
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[-200:]


@dataclass
class State:
    status: Status = Status.OFF
    active_profile: Optional[str] = None
    started_at: Optional[float] = None
    error_message: Optional[str] = None
    progress: SwitchProgress = field(default_factory=SwitchProgress)


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.state = State()
        self._lock = asyncio.Lock()

    # --- introspection -----------------------------------------------------

    def upstream_for_active(self) -> Optional[str]:
        if self.state.status == Status.ACTIVE and self.state.active_profile:
            p = self.config.profiles.get(self.state.active_profile)
            return p.upstream if p else None
        return None

    def served_model_id(self) -> Optional[str]:
        if self.state.status == Status.ACTIVE and self.state.active_profile:
            p = self.config.profiles.get(self.state.active_profile)
            return p.served_model_id if p else None
        return None

    # --- public actions ----------------------------------------------------

    async def stop(self, *, reason: str = "user request") -> None:
        async with self._lock:
            await self._stop_inner(reason=reason)

    async def switch_to(self, profile_name: str, *, reason: str = "user request") -> None:
        if profile_name not in self.config.profiles:
            raise ValueError(f"unknown profile: {profile_name}")
        # Background task — switch is long; lock prevents overlapping switches.
        asyncio.create_task(self._switch_locked(profile_name, reason))

    async def _switch_locked(self, profile_name: str, reason: str) -> None:
        async with self._lock:
            await self._switch_inner(profile_name, reason)

    # --- inner state-modifying ops (must hold self._lock) ------------------

    async def _stop_inner(self, *, reason: str) -> None:
        prev = self.state.active_profile or self.state.status.value
        log.info("stopping (was=%s, reason=%s)", prev, reason)
        # Stop containers from every profile we know about — best effort.
        all_containers: list[str] = []
        for prof in self.config.profiles.values():
            all_containers.extend(prof.container_names)
        await docker_ops.stop_and_remove(all_containers)

        self.state = State(status=Status.OFF)
        db.set_state("last_active", "off")
        db.log_transition(prev, "off", reason)

    async def _switch_inner(self, profile_name: str, reason: str) -> None:
        prof = self.config.profiles[profile_name]
        prev = self.state.active_profile or self.state.status.value
        log.info("switching from %s → %s (reason=%s)", prev, profile_name, reason)

        progress = SwitchProgress(
            target=profile_name,
            started_at=time.time(),
            expected_seconds=prof.expected_load_seconds,
        )
        self.state = State(status=Status.SWITCHING, active_profile=None, progress=progress)
        db.log_transition(prev, f"switching:{profile_name}", reason)

        try:
            # 1. Stop everything belonging to ANY profile (drain).
            progress.add_step("Stopping previous containers")
            all_containers: list[str] = []
            for p in self.config.profiles.values():
                all_containers.extend(p.container_names)
            await docker_ops.stop_and_remove(all_containers)

            # 2. Launch the target profile via its launcher script.
            progress.add_step(f"Running launcher: {prof.launcher}")
            rc, out, err = await docker_ops.run_launcher(prof.launcher)
            if rc != 0:
                progress.add_step(f"launcher exited rc={rc}: {err.strip()[:300]}")
                self._fail(f"launcher failed (rc={rc}): {err.strip()[:300]}")
                db.log_transition(f"switching:{profile_name}", "error", "launcher failed")
                return

            # 3. Poll readiness.
            progress.add_step("Waiting for ready_url to return 200")
            ready = await self._wait_until_ready(prof, progress)
            if not ready:
                progress.add_step("ready_url never became healthy; rolling back")
                # Tear down the half-started containers so the next switch is clean.
                await docker_ops.stop_and_remove(prof.container_names)
                self._fail("model never reached ready state")
                db.log_transition(f"switching:{profile_name}", "error", "readiness timeout")
                return

            # 4. Flip to active.
            progress.add_step("Active")
            self.state = State(
                status=Status.ACTIVE,
                active_profile=profile_name,
                started_at=time.time(),
                progress=progress,
            )
            db.set_state("last_active", profile_name)
            db.log_transition(f"switching:{profile_name}", f"active:{profile_name}", reason)
            log.info("switch complete: active=%s", profile_name)

        except Exception as e:
            log.exception("switch failed")
            progress.add_step(f"unexpected error: {e}")
            self._fail(str(e))
            db.log_transition(f"switching:{profile_name}", "error", str(e))

    # --- helpers -----------------------------------------------------------

    def _fail(self, msg: str) -> None:
        self.state = State(
            status=Status.ERROR,
            error_message=msg,
            progress=self.state.progress,
        )

    async def _wait_until_ready(self, prof: Profile, progress: SwitchProgress) -> bool:
        deadline = time.time() + max(prof.expected_load_seconds * 3, 240)
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(prof.ready_url)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                progress.add_step(
                    f"loading… {int(time.time() - (progress.started_at or time.time()))}s elapsed"
                )
                await asyncio.sleep(2)
        return False

    # --- auto-restore ------------------------------------------------------

    async def maybe_restore_last(self) -> None:
        last = db.get_state("last_active")
        if last and last != "off" and last in self.config.profiles:
            log.info("auto-restoring last active profile: %s", last)
            await asyncio.sleep(15)  # let docker daemon finish other startup work
            await self.switch_to(last, reason="auto-restore")
