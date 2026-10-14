"""One supervised episode process owns Habitat and its Core child."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time


def _signal_group(pid, number):
    try:
        os.killpg(pid, number)
        return True
    except ProcessLookupError:
        return False


def supervise(command, directory, seconds, cwd):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (directory / "worker.log").open("a") as log:
        process = subprocess.Popen(command, cwd=cwd, start_new_session=True,
                                   stdout=log, stderr=subprocess.STDOUT)
        reason = "finished"
        try:
            process.wait(timeout=seconds + 15)
        except subprocess.TimeoutExpired:
            reason = "timeout"
            _signal_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        except BaseException:
            _signal_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise
        finally:
            # Reap Core descendants even if native Python crashed before finally.
            _signal_group(process.pid, signal.SIGKILL)
    status = {"pid": process.pid, "exit_code": process.returncode, "reason": reason,
              "elapsed_seconds": time.monotonic() - started}
    (directory / "worker_status.json").write_text(json.dumps(status, indent=2))
    result_path = directory / "result.json"
    try:
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict):
            raise ValueError("Result is not a JSON object")
    except (OSError, ValueError):
        result = {"finish_reason": reason if reason != "finished" else "worker_error", "software_only": False}
    result["worker"] = status
    if process.returncode != 0:
        result["episode_finish_reason"] = result.get("finish_reason")
        result["finish_reason"] = "timeout" if reason == "timeout" else "worker_error"
        result["worker_error"] = f"Episode worker exited with code {process.returncode}"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(temporary, result_path)
    return result
