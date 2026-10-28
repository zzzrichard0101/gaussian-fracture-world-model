#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export WM_RESEARCH_ROOT="${WM_RESEARCH_ROOT:-$HOME/research/world-model-research}"
export WM_SOURCE_ROOT="$WM_RESEARCH_ROOT/third_party/GaussianFluent"
export WM_PROJECT_ROOT="$PROJECT_ROOT"
export UV_CACHE_DIR="$WM_RESEARCH_ROOT/cache/uv"
export XDG_CACHE_HOME="$WM_RESEARCH_ROOT/cache/xdg"
export HF_HOME="$WM_RESEARCH_ROOT/cache/huggingface"
export CUDA_HOME=/usr/local/cuda-11.8
# Older Warp loads libcuda.so (without .1). Prefer the WSL driver bridge over
# Linux driver/stub libraries already installed in the distribution.
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLBACKEND=Agg
export PATH="$CUDA_HOME/bin:$WM_RESEARCH_ROOT/venvs/gaussianfluent/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$WM_SOURCE_ROOT/gaussian-splatting:$WM_SOURCE_ROOT/gaussian-splatting/submodules/simple-knn:$WM_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$WM_RESEARCH_ROOT/venvs/gaussianfluent/bin/python"
mkdir -p "$WM_RESEARCH_ROOT"/{cache,runs,data,venvs,third_party}
command_name="${1:-doctor}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
  bootstrap)
    uv python install 3.9.25
    if [[ ! -x "$PYTHON" ]]; then
      uv venv --python 3.9.25 "$WM_RESEARCH_ROOT/venvs/gaussianfluent"
    fi
    if [[ ! -d "$WM_SOURCE_ROOT/.git" ]]; then
      git clone --no-checkout https://github.com/HB-pencil-zero/GaussianFluent.git "$WM_SOURCE_ROOT"
      revision="$($PYTHON -c 'import json,os; print(json.load(open(os.path.join(os.environ["WM_PROJECT_ROOT"],"config/upstream.json")))["GaussianFluent"]["commit"])')"
      git -C "$WM_SOURCE_ROOT" checkout --detach "$revision"
    fi
    expected="$($PYTHON -c 'import json,os; print(json.load(open(os.path.join(os.environ["WM_PROJECT_ROOT"],"config/upstream.json")))["GaussianFluent"]["commit"])')"
    actual="$(git -C "$WM_SOURCE_ROOT" rev-parse HEAD)"
    if [[ "$actual" != "$expected" ]]; then
      echo "Expected upstream $expected; found $actual. Inspect before changing an existing checkout." >&2
      exit 1
    fi
    git -C "$WM_SOURCE_ROOT" submodule update --init gaussian-splatting
    git -C "$WM_SOURCE_ROOT/gaussian-splatting" submodule update --init submodules/simple-knn
    uv pip install --python "$PYTHON" --index-url https://download.pytorch.org/whl/cu118 'torch==2.0.0+cu118' 'torchvision==0.15.1+cu118'
    uv pip install --python "$PYTHON" -r "$PROJECT_ROOT/requirements/sim.txt"
    uv pip install --python "$PYTHON" --no-build-isolation --no-deps -e "$WM_SOURCE_ROOT/gaussian-splatting/submodules/diff-gaussian-rasterization"
    uv pip install --python "$PYTHON" --no-build-isolation --no-deps -e "$WM_SOURCE_ROOT/gaussian-splatting/submodules/simple-knn"
    uv pip freeze --python "$PYTHON" > "$PROJECT_ROOT/requirements/sim.lock.txt"
    ;;
  doctor|smoke|fetch-asset|asset-check|first-impact|impact-report|analyze-separation|replay-transfer|stateful-probe|resume-impact|checkpoint-probe|history-pilot|fracture-probe|cohesive-probe|cohesive-inverse|history-refinement|direct-benchmark|joint-damage-material|response-diversity|active-probe)
    exec "$PYTHON" "$PROJECT_ROOT/scripts/${command_name//-/_}.py" "$@"
    ;;
  python)
    exec "$PYTHON" "$@"
    ;;
  test)
    exec "$PYTHON" -m pytest "$PROJECT_ROOT/tests" "$@"
    ;;
  upstream)
    script_path="${1:?upstream requires a repository-relative Python script}"
    shift
    cd "$WM_SOURCE_ROOT"
    exec "$PYTHON" "$script_path" "$@"
    ;;
  *) echo "Usage: bash scripts/wsl.sh {bootstrap|doctor|smoke|fetch-asset|asset-check|first-impact|impact-report|analyze-separation|replay-transfer|stateful-probe|resume-impact|checkpoint-probe|history-pilot|fracture-probe|cohesive-probe|cohesive-inverse|history-refinement|direct-benchmark|joint-damage-material|response-diversity|active-probe|test|python|upstream} [args...]" >&2; exit 2 ;;
esac
