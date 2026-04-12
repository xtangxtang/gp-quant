"""Base agent framework with JSON state management."""

import fcntl
import json
import os
import sys
import time
import traceback
from datetime import datetime


class AgentState:
    """Manages agent state persisted to a JSON file."""

    def __init__(self, state_dir: str, agent_name: str):
        self.agent_name = agent_name
        self.state_file = os.path.join(state_dir, f".agent_{agent_name}_state.json")
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._default()

    def _default(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "status": "idle",
            "last_run_at": "",
            "last_success_at": "",
            "last_error": "",
            "progress": {"phase": "", "current": 0, "total": 0, "pct": 0.0},
            "stats": {"files_written": 0, "rows_written": 0, "duration_seconds": 0},
            "run_count": 0,
            "consecutive_failures": 0,
        }

    def save(self):
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.state_file)

    @property
    def status(self) -> str:
        return self._data.get("status", "idle")

    @status.setter
    def status(self, v: str):
        self._data["status"] = v

    def set_running(self):
        self._data["status"] = "running"
        self._data["last_run_at"] = datetime.now().isoformat(timespec="seconds")
        self._data["run_count"] = self._data.get("run_count", 0) + 1
        self._data["last_error"] = ""
        self._data["progress"] = {"phase": "starting", "current": 0, "total": 0, "pct": 0.0}
        self.save()

    def set_success(self, duration: float, stats: dict | None = None):
        self._data["status"] = "success"
        self._data["last_success_at"] = datetime.now().isoformat(timespec="seconds")
        self._data["consecutive_failures"] = 0
        self._data["stats"]["duration_seconds"] = round(duration, 1)
        if stats:
            self._data["stats"].update(stats)
        self.save()

    def set_failed(self, error: str, duration: float):
        self._data["status"] = "failed"
        self._data["last_error"] = error[:500]
        self._data["consecutive_failures"] = self._data.get("consecutive_failures", 0) + 1
        self._data["stats"]["duration_seconds"] = round(duration, 1)
        self.save()

    def update_progress(self, phase: str, current: int = 0, total: int = 0):
        pct = round(current / total * 100, 1) if total > 0 else 0.0
        self._data["progress"] = {"phase": phase, "current": current, "total": total, "pct": pct}
        self.save()

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    @property
    def data(self) -> dict:
        return dict(self._data)


class BaseAgent:
    """Base class for all data agents.

    Subclasses must implement:
      - name: str — unique agent identifier
      - run(ctx: AgentContext) — the main work
    """

    name: str = "base"
    description: str = ""

    def __init__(self, data_dir: str, token: str = ""):
        self.data_dir = data_dir
        self.token = token or os.getenv("TUSHARE_TOKEN", "") or os.getenv("GP_TUSHARE_TOKEN", "")
        self.state = AgentState(data_dir, self.name)
        self._lock_fd = None

    def _lock_file(self) -> str:
        return os.path.join(self.data_dir, f".agent_{self.name}.lock")

    def acquire_lock(self) -> bool:
        try:
            self._lock_fd = open(self._lock_file(), "w")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, IOError):
            return False

    def release_lock(self):
        if self._lock_fd:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None

    def execute(self, **kwargs) -> bool:
        """Run the agent with state management. Returns True on success."""
        if not self.acquire_lock():
            print(f"[{self.name}] Another instance is running, skip.")
            return False

        self.state.set_running()
        t0 = time.time()
        try:
            print(f"[{self.name}] Started at {datetime.now().isoformat(timespec='seconds')}")
            self.run(**kwargs)
            dur = time.time() - t0
            self.state.set_success(dur)
            print(f"[{self.name}] Completed in {dur:.1f}s")
            return True
        except Exception as e:
            dur = time.time() - t0
            tb = traceback.format_exc()
            self.state.set_failed(str(e), dur)
            print(f"[{self.name}] Failed after {dur:.1f}s: {e}")
            print(tb, file=sys.stderr)
            return False
        finally:
            self.release_lock()

    def run(self, **kwargs):
        """Override this in subclass."""
        raise NotImplementedError

    def update_progress(self, phase: str, current: int = 0, total: int = 0):
        self.state.update_progress(phase, current, total)
