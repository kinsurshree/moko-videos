"""Delete old published videos from the repo so it does not grow forever.
Metricool already copied each video to its own storage when it was scheduled, so the repo copy is only
needed until the post is live and (for episodes) until the Sunday compilation has been built.
Rules: episode videos older than 9 days, weekly compilations older than 3 days.
Usage: python3 cleanup_old.py [--dry]   (run from repo root or engine/; then git commit + push)"""
import os, re, sys, datetime, subprocess
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dry = "--dry" in sys.argv
today = datetime.date.today()
removed = []
for line in open(os.path.join(root, "EPISODE_LOG.md"), encoding="utf-8"):
    m = re.match(r"\s*(ep\d+|week-(\d{4}-\d\d-\d\d))\s*\|\s*(\d{4}-\d\d-\d\d)", line)
    if not m: continue
    name, wk, d = m.group(1), m.group(2), datetime.date.fromisoformat(m.group(3))
    if name == "ep00": continue                       # keep the intro
    if wk: path, limit = f"videos/compilations/moko_{name.replace('-', '_', 1)}.mp4", 3
    else:  path, limit = f"videos/moko_{name}.mp4", 9
    if (today - d).days > limit and os.path.exists(os.path.join(root, path)):
        removed.append(path)
        if not dry: subprocess.run(["git", "rm", "-q", path], cwd=root, check=True)
print(("would remove: " if dry else "removed: "), removed or "nothing")
