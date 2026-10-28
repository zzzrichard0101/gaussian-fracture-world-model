"""Execute stage guards in child processes because Python audit hooks persist."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("stage", ["infer", "compress"])
def test_stage_guard_denies_labels_parent_config_and_held_out_plans(tmp_path, stage):
    for name in ("private/parameters.json", "config.json", "future-plans.json", "allowed.json"):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    scripts = Path(__file__).parents[1]/"scripts"
    program = r'''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from response_diversity import protect_inputs
root = Path(sys.argv[2])
seen = protect_inputs(root, sys.argv[3], {"allowed.json"})
assert (root/"allowed.json").read_text() == "{}"
for name in ("private/parameters.json", "config.json", "future-plans.json"):
    try:
        (root/name).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError("Privileged input was not blocked: "+name)
assert seen == {"allowed.json"}
'''
    result = subprocess.run([sys.executable, "-c", program, str(scripts), str(tmp_path), stage], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
