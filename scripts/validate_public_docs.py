"""Check published local links, file hashes and metric-preserving JSON export."""
import json
from pathlib import Path
import re

from publish_results import sha, portable


def main():
    root = Path(__file__).resolve().parents[1]
    linked = 0
    for relative in ("README.md", "docs/RESULTS.md", "docs/REPRODUCING.md", "THIRD_PARTY_NOTICES.md"):
        path = root/relative
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target:
                continue
            assert (path.parent/target).is_file(), (relative, target)
            linked += 1
    manifest = json.loads((root/"docs/results/publication-manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        source, target = root/entry["source"], root/entry["published"]
        if "published_sha256" in entry:
            assert sha(target) == entry["published_sha256"]
            assert sha(source) == entry["source_sha256"]
            assert json.loads(target.read_text(encoding="utf-8")) == portable(json.loads(source.read_text(encoding="utf-8")))
        else:
            assert sha(source) == sha(target) == entry["sha256"]
    assert not any(path.stat().st_size > 10*1024*1024 for path in (root/"docs").rglob("*") if path.is_file())
    print(f"Verified {linked} local Markdown links and {len(manifest['files'])} publication entries; exported numeric results unchanged")


if __name__ == "__main__":
    main()
