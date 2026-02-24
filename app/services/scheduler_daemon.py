import os
import signal
import subprocess
import time
from pathlib import Path

SCHEDULER_PID_PATH = Path("data/scheduler.pid")
SCHEDULER_STDOUT_PATH = Path("scheduler.log")


def _read_pid() -> int | None:
    if not SCHEDULER_PID_PATH.exists():
        return None
    raw = SCHEDULER_PID_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    return True


def _clear_stale_pid() -> None:
    pid = _read_pid()
    if pid is None:
        return
    if not _pid_alive(pid):
        SCHEDULER_PID_PATH.unlink(missing_ok=True)


def scheduler_state() -> dict:
    _clear_stale_pid()
    pid = _read_pid()
    running = bool(pid and _pid_alive(pid))
    return {
        "running": running,
        "pid": pid if running else None,
        "pid_file": str(SCHEDULER_PID_PATH),
        "log_file": str(SCHEDULER_STDOUT_PATH),
    }


def start_scheduler_daemon() -> dict:
    state = scheduler_state()
    if state["running"]:
        return {"status": "already_running", **state}

    SCHEDULER_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with SCHEDULER_STDOUT_PATH.open("a", encoding="utf-8") as out:
        proc = subprocess.Popen(
            ["python", "scheduler.py"],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    SCHEDULER_PID_PATH.write_text(str(proc.pid), encoding="utf-8")
    return {"status": "started", **scheduler_state()}


def stop_scheduler_daemon(wait_seconds: int = 8) -> dict:
    state = scheduler_state()
    pid = state["pid"]
    if not pid:
        SCHEDULER_PID_PATH.unlink(missing_ok=True)
        return {"status": "already_stopped", **scheduler_state()}

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        SCHEDULER_PID_PATH.unlink(missing_ok=True)
        return {"status": "already_stopped", **scheduler_state()}

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not _pid_alive(pid):
            SCHEDULER_PID_PATH.unlink(missing_ok=True)
            return {"status": "stopped", **scheduler_state()}
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    time.sleep(0.2)
    SCHEDULER_PID_PATH.unlink(missing_ok=True)
    return {"status": "killed", **scheduler_state()}
