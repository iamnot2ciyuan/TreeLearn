import argparse
import os
import re
import subprocess
import time
from datetime import datetime


def read_last_matching_lines(path: str, marker: str, max_bytes: int = 1024 * 200) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(size - max_bytes, 0), os.SEEK_SET)
        data = f.read().decode("utf-8", "ignore")
    matches = []
    for line in data.splitlines():
        if marker in line:
            matches.append(line.strip())
    return matches[-1] if matches else ""


def _read_tail_bytes(path: str, max_bytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(size - max_bytes, 0), os.SEEK_SET)
        return f.read()


def get_fast_stage_and_pct(fast_log: str, max_bytes: int = 512 * 1024) -> tuple[str, str]:
    if not fast_log:
        return "", ""
    try:
        data = _read_tail_bytes(fast_log, max_bytes).decode("utf-8", "ignore")
    except OSError:
        return "", ""

    marker_lines = [ln.strip() for ln in data.splitlines() if "####################" in ln]
    stage = marker_lines[-1] if marker_lines else ""

    pct_matches = re.findall(r"[0-9]{1,3}%\|", data)
    pct = pct_matches[-1] if pct_matches else ""
    return stage, pct


def fast_log_path_for_pid(pid: int) -> tuple[str, str]:
    """
    Resolve which file to read for tqdm / fast progress.

    Returns (path_or_empty, hint):
      - If stdout is a regular file (typical nohup redirect), use that path or /proc/pid/fd/1.
      - If stdout is tty/pipe, return ("", "stdout_is_tty_or_pipe") so we do not pick stale work_dirs/*.log.
    """
    fd1 = f"/proc/{pid}/fd/1"
    try:
        target = os.readlink(fd1)
    except OSError:
        return "", "no_proc_fd1"

    if target.startswith("/dev/"):
        return "", "stdout_is_terminal"
    if target.startswith("pipe:") or target.startswith("socket:"):
        return "", "stdout_is_pipe_or_socket"

    # File path (possibly "path (deleted)")
    clean = target.replace(" (deleted)", "").strip()
    if clean and os.path.isfile(clean):
        return clean, ""
    # Open file via proc fd (deleted on disk but still writing)
    try:
        os.stat(fd1)
        return fd1, "via_proc_fd1"
    except OSError:
        return "", "stdout_not_a_file"


def process_start_epoch(pid: int) -> float | None:
    """Boot time + starttime/CLK_TCK from /proc (seconds since epoch)."""
    try:
        boot_line = open("/proc/stat", encoding="utf-8").read()
        btime = None
        for line in boot_line.splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        if btime is None:
            return None
        clk = os.sysconf("SC_CLK_TCK")
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            content = f.read()
        rparen = content.rfind(")")
        rest = content[rparen + 2 :].split()
        starttime = int(rest[19])
        return btime + starttime / float(clk)
    except (OSError, ValueError, IndexError):
        return None


def latest_fast_log_after_start(work_dirs: str, start_epoch: float | None) -> str:
    """Fallback: newest pipeline_gpu_fast_*.log modified at/after process start (with slack)."""
    try:
        entries = os.listdir(work_dirs)
    except FileNotFoundError:
        return ""
    candidates = [e for e in entries if e.startswith("pipeline_gpu_fast_") and e.endswith(".log")]
    if not candidates:
        return ""
    paths = [os.path.join(work_dirs, e) for e in candidates]
    if start_epoch is not None:
        slack = 120.0
        paths = [p for p in paths if os.path.getmtime(p) >= start_epoch - slack]
    if not paths:
        return ""
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def count_items(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    try:
        return len([name for name in os.listdir(dir_path) if name not in {".", ".."}])
    except FileNotFoundError:
        return 0


def ps_status(pid: int) -> str:
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "stat=,wchan=,etime=,%cpu=,%mem=,cmd="],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "pid_not_running"


def gpu_snapshot() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description="Periodic pipeline monitor: GPU, ps, main log stage, stdout/tqdm when available."
    )
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--interval-sec", type=int, default=300)
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument(
        "--work-dirs",
        type=str,
        default="/home/ls/lc/TreeLearn/work_dirs",
        help="Directory containing pipeline_gpu_fast_*.log (fallback only)",
    )
    ap.add_argument(
        "--main-log",
        type=str,
        default="/home/ls/lc/TreeLearn/data/documentation/log_pipeline.txt",
    )
    ap.add_argument(
        "--results-base",
        type=str,
        default="/home/ls/lc/TreeLearn/data/results",
        help="Parent of pointwise_results and full_forest",
    )
    args = ap.parse_args()

    point_dir = os.path.join(args.results_base, "pointwise_results")
    full_dir = os.path.join(args.results_base, "full_forest")

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(f"=== monitor start {datetime.now().isoformat(timespec='seconds')} pid={args.pid} ===\n")
        f.flush()

    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        fast_path, fast_hint = fast_log_path_for_pid(args.pid)
        if fast_path:
            fast_source = fast_hint or "stdout_file"
        else:
            start_ep = process_start_epoch(args.pid)
            fb_path = latest_fast_log_after_start(args.work_dirs, start_ep)
            if fb_path:
                fast_path = fb_path
                fast_source = "fallback_newest_fast_log_after_pid_start"
            else:
                fast_source = ""
            if fast_hint:
                fast_source = (fast_source + ("; " if fast_source else "") + fast_hint)

        fast_stage, fast_pct = get_fast_stage_and_pct(fast_path)

        stage_main = read_last_matching_lines(args.main_log, "####################")
        pid_stat = ps_status(args.pid)
        gpu_snap = gpu_snapshot()
        point_cnt = count_items(point_dir)
        full_cnt = count_items(full_dir)

        with open(args.log, "a", encoding="utf-8") as f:
            f.write(f"--- {ts} ---\n")
            f.write(f"GPU: {gpu_snap}\n")
            f.write(f"PID: {pid_stat}\n")
            f.write(f"MainStage: {stage_main}\n")
            f.write(f"FastLog: {fast_path or '(none)'}\n")
            f.write(f"FastSource: {fast_source or fast_note}\n")
            f.write(f"FastStage: {fast_stage}\n")
            f.write(f"FastPct: {fast_pct}\n")
            f.write(f"pointwise_results items: {point_cnt}\n")
            f.write(f"full_forest items: {full_cnt}\n")
            f.write("\n")
            f.flush()

        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
