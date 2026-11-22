import json
import os
import subprocess
import sys
import time


def start(out_dir, config, git=None):
    os.makedirs(out_dir, exist_ok=True)
    git = git or _git()
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump({"cmd": " ".join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "config": config,
                   "git_sha": git.get("sha"), "git_dirty": bool(git.get("diff"))}, f, indent=2)
    if git.get("diff"):
        with open(os.path.join(out_dir, "dirty.diff"), "w") as f:
            f.write(git["diff"])
    return out_dir


# local runs read git directly; Modal containers have no .git, so modal_app passes this in
def _git():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, timeout=5).stdout
        return {"sha": sha or None, "diff": diff}
    except Exception:
        return {}


