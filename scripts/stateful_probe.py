"""One command: simulate, analyze physical separation, and replay transfers."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    project = Path(os.environ["WM_PROJECT_ROOT"])
    scripts = project / "scripts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, help="Reprocess an existing completed trajectory; does not resume physics")
    args = parser.parse_args()
    run = args.run
    if run is None:
        command = [sys.executable, "-u", str(scripts / "first_impact.py"), "--config", str(project / "config/stateful-impact.json")]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("OUTPUT="):
                run = Path(line.strip().split("=", 1)[1])
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    if run is None:
        raise RuntimeError("Simulation did not report its output path")
    for filename in ("analyze_separation.py", "replay_transfer.py"):
        subprocess.run([sys.executable, "-u", str(scripts / filename), "--run", str(run)], check=True)
    separation = json.loads((run / "separation/report.json").read_text())
    replay = json.loads((run / "transfer-replay/report.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if not summary["finite_checks_passed"] or not summary["checks"]["precontact_gravity_passed"]:
        raise RuntimeError("Simulation sanity checks failed")
    if not replay["same_recorded_physics"] or not replay["initial_identity_render_match"]:
        raise RuntimeError("Replay consistency check failed")
    output = project / "artifacts/stateful-probe"
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in [
        (run / "transfer-replay/transfer-replay.mp4", "transfer-replay.mp4"),
        (run / "separation/appearance-vs-particles.mp4", "appearance-vs-particles.mp4"),
        (run / "separation/connectivity.png", "connectivity.png"),
        (run / "separation/report.json", "separation.json"),
        (run / "manifest.json", "manifest.json"),
        (run / "summary.json", "simulation.json"),
    ]:
        shutil.copy2(source, output / destination)
    last_frame = sorted((run / "transfer-replay/frames").glob("*.png"))[-1]
    shutil.copy2(last_frame, output / "transfer-comparison.png")
    report = {"status": "completed", "run_directory": str(run), "simulation": summary,
              "mode": "reprocess-recorded-trajectory" if args.run else "simulate-and-analyze",
              "particle_separation_candidate": separation["robust_particle_separation_candidate"],
              "radius_sensitivity": separation["radius_sensitivity"],
              "same_recorded_physics_replayed": replay["same_recorded_physics"],
              "initial_identity_render_match": replay["initial_identity_render_match"],
              "fracture_accuracy_validated": False,
              "pipeline_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "note": "Kinematic history transfer and independent separation diagnostics; not a validated fracture simulator or resumable checkpoint."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "run_directory": str(run),
                      "particle_separation_candidate": report["particle_separation_candidate"],
                      "same_recorded_physics_replayed": True}, indent=2))


if __name__ == "__main__":
    main()
