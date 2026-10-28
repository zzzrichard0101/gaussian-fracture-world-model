#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export WM_PROJECT_ROOT="$PROJECT_ROOT"
export WM_RESEARCH_ROOT="${WM_RESEARCH_ROOT:-$HOME/research/world-model-research}"
export WM_VJEPA_ROOT="$WM_RESEARCH_ROOT/third_party/vjepa2"
export PYTHONPATH="$PROJECT_ROOT:$WM_VJEPA_ROOT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib"
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export HF_HOME="$WM_RESEARCH_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$WM_RESEARCH_ROOT/cache/xdg-world-model"
export UV_CACHE_DIR="$WM_RESEARCH_ROOT/cache/uv"
export PATH="$HOME/.local/bin:$PATH"
PYTHON="$WM_RESEARCH_ROOT/venvs/world-model/bin/python"
mkdir -p "$WM_RESEARCH_ROOT"/{cache,runs,models,venvs,third_party}
command_name="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
  bootstrap)
    uv python install 3.11.12
    if [[ ! -x "$PYTHON" ]]; then uv venv --python 3.11.12 "$WM_RESEARCH_ROOT/venvs/world-model"; fi
    uv pip install --python "$PYTHON" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$PYTHON" -r "$PROJECT_ROOT/requirements/world-model.txt"
    if [[ ! -d "$WM_VJEPA_ROOT/.git" ]]; then
      git clone https://github.com/facebookresearch/vjepa2.git "$WM_VJEPA_ROOT"
      git -C "$WM_VJEPA_ROOT" checkout --detach 204698b45b3712590f06245fbfba32d3be539812
    fi
    uv pip freeze --python "$PYTHON" > "$PROJECT_ROOT/requirements/world-model.lock.txt"
    exec "$PYTHON" "$PROJECT_ROOT/scripts/world_model_probe.py" --stage weights
    ;;
  run|rank|evaluate)
    if [[ "$command_name" == run ]]; then
      exec "$PYTHON" "$PROJECT_ROOT/scripts/world_model_probe.py" "$@"
    fi
    exec "$PYTHON" "$PROJECT_ROOT/scripts/world_model_probe.py" --stage "$command_name" "$@"
    ;;
  action-adapter) exec "$PYTHON" "$PROJECT_ROOT/scripts/action_adapter.py" "$@" ;;
  belief-state) exec "$PYTHON" "$PROJECT_ROOT/scripts/belief_state.py" "$@" ;;
  python) exec "$PYTHON" "$@" ;;
  test) exec "$PYTHON" -m pytest "$PROJECT_ROOT/tests/test_world_model_io.py" "$PROJECT_ROOT/tests/test_action_adapter.py" "$PROJECT_ROOT/tests/test_belief_state.py" "$PROJECT_ROOT/tests/test_history_refinement.py" "$PROJECT_ROOT/tests/test_joint_state.py" "$PROJECT_ROOT/tests/test_hypothesis_selection.py" "$PROJECT_ROOT/tests/test_response_diversity_io.py" "$PROJECT_ROOT/tests/test_active_probe.py" "$PROJECT_ROOT/tests/test_fixture_reference.py" "$PROJECT_ROOT/tests/test_compliant_fixture.py" "$@" ;;
  *) echo "Usage: world-model.sh {bootstrap|run|rank|evaluate|action-adapter|belief-state|python|test}" >&2; exit 2 ;;
esac
