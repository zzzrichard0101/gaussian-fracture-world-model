"""Download only one official asset, with its resolved Hugging Face commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="watermelon")
    args = parser.parse_args()
    config = json.loads((Path(os.environ["WM_PROJECT_ROOT"]) / "config/assets.json").read_text())
    repo = config["repo"]
    info = HfApi().model_info(repo, revision=config["revision"], files_metadata=True)
    prefixes = (f"model/{args.asset}/",)
    selected = [f for f in info.siblings if f.rfilename.startswith(prefixes)]
    if not selected:
        raise ValueError(f"No released model/{args.asset}/ asset found")
    destination = Path(os.environ["WM_RESEARCH_ROOT"]) / "data/gaussianfluent"
    print(json.dumps({"repo": repo, "commit": info.sha, "asset": args.asset,
                      "download_bytes": sum(f.size or 0 for f in selected), "files": [f.rfilename for f in selected]}, indent=2), flush=True)
    snapshot_download(repo_id=repo, revision=info.sha, local_dir=destination, allow_patterns=[f"model/{args.asset}/**"])
    manifest = {"repo": repo, "commit": info.sha, "asset": args.asset, "files": []}
    for item in selected:
        path = destination / item.rfilename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest["files"].append({"path": item.rfilename, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    output = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts" / f"asset-{args.asset}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Asset saved under {destination / 'model' / args.asset}")


if __name__ == "__main__":
    main()
